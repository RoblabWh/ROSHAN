import yaml
import optuna
import os, shutil
import sys
from utils import SimulationBridge, get_project_paths, looks_like_a_notebook
import logging

module_directory = get_project_paths('module_directory')
sys.path.insert(0, module_directory)

# Change the current working directory, so that the python script has the same folder as the c++ executable
os.chdir(module_directory)

# Check Python Version
pythonversion = str(sys.version_info[0]) + str(sys.version_info[1])

# Find Firesim Module in the current directory
firesimverison = [f for f in os.listdir(module_directory) if f[:7] == 'firesim'][0].split('-')[1]

if pythonversion != firesimverison:
    raise ValueError(f"Python Version {pythonversion} does not match Firesim Version {firesimverison}."
                     f"Activate your conda environment before compiling the library.")

import firesim

def assert_config(config):
    """
    Validates that the config dictionary has sane values.
    Raises ValueError with descriptive messages on invalid configurations.
    """
    c_settings = config["settings"]
    hierarchy_type = c_settings["hierarchy_type"]
    c_agent = config["environment"]["agent"][hierarchy_type]
    used_algo = c_agent["algorithm"]
    c_algo = config["algorithm"].get(used_algo, "no_algo")
    rl_mode = c_settings["rl_mode"]

    # --- Basic type/range validation ---
    if c_settings["mode"] not in [0, 2]:
        raise ValueError("Invalid mode in config.yaml. Must be 0 (GUI_RL) or 2 (NoGUI_RL)")
    if hierarchy_type not in ["fly_agent", "explore_agent", "planner_agent"]:
        raise ValueError("Invalid hierarchy_type in config.yaml. Must be 'fly_agent', 'explore_agent', or 'planner_agent'.")
    if rl_mode not in ["train", "eval"]:
        raise ValueError("Invalid rl_mode in config.yaml. Must be 'train' or 'eval'.")
    if c_agent["view_range"] <= 0:
        raise ValueError("view_range must be positive.")
    if c_agent["num_agents"] <= 0:
        raise ValueError("num_agents must be positive.")

    # --- Algorithm-specific numeric bounds ---
    if c_algo != "no_algo" and isinstance(c_algo, dict):
        if c_algo.get("lr", 1) <= 0:
            raise ValueError(f"lr must be positive for {used_algo}.")
        if c_algo.get("batch_size", 1) <= 0:
            raise ValueError(f"batch_size must be positive for {used_algo}.")
    if used_algo == "PPO" and isinstance(c_algo, dict):
        if c_algo["horizon"] < c_algo["batch_size"]:
            raise ValueError(f"PPO horizon ({c_algo['horizon']}) must be >= batch_size ({c_algo['batch_size']}).")
        if not (0 < c_algo["gamma"] <= 1):
            raise ValueError(f"PPO gamma must be in (0, 1], got {c_algo['gamma']}.")
        if not (0 < c_algo["eps_clip"] < 1):
            raise ValueError(f"PPO eps_clip must be in (0, 1), got {c_algo['eps_clip']}.")

    # --- use_heuristic / use_water_limit (independent flags) ---
    # use_heuristic is the new name for eval_fly_policy. Accept the legacy key with a deprecation warning.
    _use_heuristic_key = "use_heuristic" if "use_heuristic" in c_settings else (
        "eval_fly_policy" if "eval_fly_policy" in c_settings else None
    )
    if _use_heuristic_key == "eval_fly_policy":
        logging.warning("Config key 'eval_fly_policy' is deprecated; rename to 'use_heuristic'.")
    use_heuristic = bool(c_settings[_use_heuristic_key]) if _use_heuristic_key else False
    use_water_limit = bool(config["environment"]["agent"]["use_water_limit"])

    # use_heuristic is only meaningful for FlyAgent in eval mode; warn (don't fail) otherwise so it's
    # simply ignored. The heuristic FlyPolicy gating in fly_agent.cpp enforces this at runtime.
    if use_heuristic:
        if hierarchy_type != "fly_agent":
            logging.warning(
                f"use_heuristic=true has no effect when hierarchy_type={hierarchy_type!r}. "
                "The heuristic FlyPolicy only runs for fly_agent. Ignoring use_heuristic."
            )
        elif rl_mode != "eval":
            logging.warning(
                "use_heuristic=true but rl_mode is not 'eval'. The heuristic is a no-learning "
                "controller; it should be used only for evaluation. Ignoring use_heuristic."
            )
        elif not c_settings["log_eval"]:
            raise ValueError("log_eval must be True when running the heuristic (use_heuristic=true).")

    # use_water_limit is independent of use_heuristic — refueling now works for trained planners too.
    # Warn only when it is likely misconfigured (trainable FlyAgent with water pressure but no learned refuel path).
    if use_water_limit and hierarchy_type == "fly_agent" and rl_mode == "train" and not use_heuristic:
        logging.warning(
            "use_water_limit=true with a trainable FlyAgent (rl_mode=train, use_heuristic=false) "
            "enables water depletion but the learned policy has no explicit refuel incentive. "
            "Usually use_water_limit is paired with PlannerAgent training or heuristic eval."
        )

    if rl_mode == "train":
        if not c_settings["log_eval"]:
            raise ValueError("log_eval should be True when rl_mode is 'train'. While this would"
                             " work, it is more likely than not that the user did not intend to do this.")

    # --- Hierarchy-specific constraints ---
    if hierarchy_type == "planner_agent":
        if config["environment"]["agent"]["planner_agent"]["default_model_folder"] == "":
            raise ValueError("default_model_folder for planner_agent cannot be empty in config.yaml when planner_agent is selected.")
        if config["environment"]["agent"]["planner_agent"]["default_model_name"] == "":
            raise ValueError("default_model_name cannot be empty in config.yaml when planner_agent is selected.")
    if hierarchy_type == "fly_agent":
        if not c_agent["use_simple_policy"]:
            if config["environment"]["agent_behaviour"]["fire_goal_percentage"] != 1.0:
                raise ValueError("fire_goal_percentage should be 1.0 when using fly_agent with complex policy.")

    # --- Auto-train constraints ---
    if c_settings["auto_train"]["use_auto_train"]:
        if c_settings["resume"]:
            raise ValueError("resume cannot be True when auto_train is enabled in config.yaml.")
        if rl_mode == "eval":
            raise ValueError("rl_mode cannot be 'eval' when auto_train is enabled in config.yaml. "
                             "While this configuration WOULD work, it is not intended to be used in this way, since it would automatically "
                             "resume training from the last checkpoint, which is not what you want when evaluating a model.")

    # --- Optuna constraints ---
    if c_settings["optuna"]["use_optuna"]:
        if c_settings["optuna"]["objective"] not in ["objective", "reward", "time_to_end"]:
            raise ValueError("Invalid objective in config.yaml for optuna. Must be 'objective', 'reward', or 'time_to_end'.")
        if rl_mode != "train":
            raise ValueError("rl_mode must be 'train' when using optuna in config.yaml.")
        if c_settings["optuna"]["use_pruning"] and c_settings["optuna"]["objective"] == "time_to_end":
            raise ValueError("Pruning cannot be used when objective is 'time_to_end' in config.yaml.")

    # --- Replay buffer constraints ---
    if c_settings["save_replay_buffer"]:
        if used_algo == "IQL":
            raise ValueError("save_replay_buffer not supported for IQL.")
        if rl_mode == "train":
            raise ValueError("save_replay_buffer cannot be True when rl_mode is 'train'.")
        if c_settings["save_size"] > c_algo["memory_size"]:
            raise ValueError(f"save_size cannot be larger than memory_size in algorithm config for {used_algo}.")

    # --- Algorithm-specific constraints ---
    if used_algo == "PPO" and isinstance(c_algo, dict):
        if c_algo["separate_optimizers"] and config["algorithm"]["share_encoder"]:
            raise ValueError("When using separate optimizers, their encoders can't be shared.")
    if used_algo == "TD3":
        if hierarchy_type == "planner_agent":
            raise ValueError("TD3 not supported for planner_agent.")
        if config["algorithm"]["use_tanh_dist"]:
            raise ValueError("TD3 does not support tanh action distribution. It doesn't use a distribution "
                             "at all and the Outputs of the Network need a tanh activation instead. "
                             "Disable use_tanh in config.yaml.")
    if used_algo == "IQL":
        if hierarchy_type == "planner_agent":
            raise ValueError("IQL not supported for planner_agent.")
def sim(config : dict, overrides: dict = None, trial=None):

    config = inject_overrides(config, overrides, trial)
    assert_config(config)

    # Always safe the initial config to a dummy file in the root directory
    # This is not beautiful, but it works for now, the main cause for this is
    # that the config object can be modified at runtime (e.g. optuna overrides)
    dummy_config_path = os.path.join(get_project_paths("root_path"), "used_config.yaml")
    with open(dummy_config_path, 'w') as f:
        yaml.dump(config, f)

    if config["settings"].get("pytorch_detect_anomaly"):
        os.environ["PYTORCH_DETECT_ANOMALY"] = "true"

    from hierarchy_manager import HierarchyManager

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

    # RL_Status Dictionary, sending back and forth to C++
    sim_bridge = SimulationBridge(config)

    llm = None
    if config["settings"]["llm_support"]:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")
        #llm = LLMPredictorAPI("Qwen/Qwen2-1.5B")

    # Initialize the EngineCore and send the RL_Status
    engine = firesim.EngineCore()
    engine.Init(config["settings"]["mode"], dummy_config_path)
    engine.SendRLStatusToModel(sim_bridge.get_status())
    engine.InitializeMap()

    # Wait for the initial mode selection from the user
    while not engine.InitialModeSelectionDone():
        engine.HandleEvents()
        engine.Update() # Pretty useless here
        engine.Render()

    sim_bridge.set_status(engine.GetRLStatusFromModel())

    # If the user changed settings in the GUI, we need to update the config OBJECT
    # Update the config with the new settings from the GUI
    if config["settings"]["skip_gui_init"] is False:
        general_settings = config["settings"]
        general_settings["hierarchy_type"] = sim_bridge.get("hierarchy_type")
        general_settings["rl_mode"] = sim_bridge.get("rl_mode")

        agent_type = config["settings"]["hierarchy_type"]
        agent_settings = config["environment"]["agent"][agent_type]
        if sim_bridge.get("rl_mode") == "eval":
            # Shouldn't really be an issue when the user selects "train" mode, but just in case
            agent_settings["default_model_folder"] = sim_bridge.get("model_path")
            agent_settings["default_model_name"] = sim_bridge.get("model_name")
            # Only change resume parameter if the user really loaded a model
            general_settings["resume"] = sim_bridge.get("resume")
            if sim_bridge.get("resume"):
                general_settings["rl_mode"] = "train"
                sim_bridge.set("rl_mode", "train")

    # Check if the model_folder is empty, if not, ask the user if they want to proceed (and delete the contents in the folder)
    # The content only needs to be deleted if the resume parameter is set to False
    engine.HandleEvents()
    if os.path.exists(sim_bridge.get("model_path")) and engine.IsRunning():
        if os.listdir(sim_bridge.get("model_path")) and not (sim_bridge.get("resume") or (sim_bridge.get("rl_mode") == "eval")):
            # Ask the user here if they want to delete the contents of the model path
            # If the user does not want to delete the contents, we will exit the program
            if config["settings"]["mode"] != 0:  # GUI Mode only ask in non-GUI mode
                if not looks_like_a_notebook():
                    user_input = input(f"Do you want to delete the contents of {sim_bridge.get('model_path')}? (y/n): ")
                    if user_input.lower() != 'y':
                        print("Exiting program.")
                        engine.Clean()
                        return
                else:
                    print(f"\033[31mNotebook environment detected. For safety reasons, exiting program to avoid accidental data loss."
                          f"\nIf you want to overwrite your training session, please delete the contents of {sim_bridge.get('model_path')} manually and restart\033[0m")
                    engine.Clean()
                    return
            # Go through all files and directories in the model path and delete them
            print(f"Deleting contents of {sim_bridge.get('model_path')}...")
            for filename in os.listdir(sim_bridge.get("model_path")):
                file_path = os.path.join(sim_bridge.get("model_path"), filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")

    # Initialize the HierarchyManager
    hierarchy_manager = None
    if engine.IsRunning():
        hierarchy_manager = HierarchyManager(config, sim_bridge)
        engine.SendRLStatusToModel(sim_bridge.get_status())

    # for optuna pruning
    loop_step = 0
    REPORT_INTERVAL = 1000
    # Batch status IPC: in NoGUI mode, skip expensive C++/Python boundary
    # crossings most steps since no user interaction occurs.
    _STATUS_INTERVAL = 100
    _no_gui = config["settings"]["mode"] == 2
    _ipc_counter = 0

    try:
        while engine.IsRunning() and sim_bridge.get("agent_online"):
            engine.Update()
            engine.Render()
            engine.HandleEvents()

            _ipc_counter += 1
            _needs_ipc = not _no_gui or (_ipc_counter % _STATUS_INTERVAL == 0)

            # C++ Python Interface & Controls
            if _needs_ipc:
                sim_bridge.set_status(engine.GetRLStatusFromModel())
                hierarchy_manager.update_status()

            if engine.AgentIsRunning() and sim_bridge.get("agent_online"):

                if sim_bridge.get("rl_mode") == "train":
                    hierarchy_manager.train(engine)
                else:
                    hierarchy_manager.eval(engine)

                if trial is not None and config["settings"]["optuna"]["use_pruning"]:
                    if loop_step % REPORT_INTERVAL == 0:
                        report_idx = loop_step // REPORT_INTERVAL
                        intermediate = sim_bridge.get("objective")
                        trial.report(intermediate, step=report_idx)
                        if trial.should_prune():
                            raise optuna.TrialPruned()
                    loop_step += 1

                if _needs_ipc:
                    hierarchy_manager.update_status()
                    engine.SendRLStatusToModel(sim_bridge.get_status())

            if config["settings"]["llm_support"]:
                user_input = engine.GetUserInput()
                if user_input != "":
                    # TODO: DEPRECATED (nice idea, but I have no clue on how to use this meaningfully in this context rn)
                    llm(engine, user_input, 0, 0)
        engine.Clean()
        return hierarchy_manager.get_final_metric(config["settings"]["optuna"]["objective"])
    except optuna.TrialPruned:
        engine.Clean()
        print(f"Trial pruned at step {trial.number}")
        raise
    except Exception as e:
        engine.Clean()
        print(f"Error: {e}")
        return -float("inf")

def inject_overrides(config: dict, overrides: dict, trial: optuna.Trial) -> dict:
    """
    Injects the overrides into the config dictionary.
    :param config: The config dictionary to inject the overrides into.
    :param overrides: The overrides dictionary.
    :param trial: The current Optuna trial.
    :return: The updated config dictionary.
    """
    if overrides is None:
        return config

    agent_dict = config["environment"]["agent"][config["settings"]["hierarchy_type"]]
    used_algorithm = agent_dict["algorithm"]
    algo_dict = config["algorithm"][used_algorithm]
    reward_dict = agent_dict["rewards"]
    if "hparams" in overrides:
        for key, value in overrides["hparams"].items():
            if key in algo_dict:
                algo_dict[key] = value

    if "rewards" in overrides:
        for key, value in overrides["rewards"].items():
            if key in reward_dict:
                if key == "BoundaryTerminal" and "Collision" in reward_dict:
                    reward_dict["Collision"] = value
                reward_dict[key] = value

    config["paths"]["model_directory"] = os.path.join(config["settings"]["optuna"]["study_root"], config["settings"]["optuna"]["study_name"], f"trial_{trial.number}")

    config["settings"]["rl_mode"] = "train" # Force train mode for Optuna runs
    config["settings"]["auto_train"]["use_auto_train"] = False # Disable auto_train for Optuna runs
    config["settings"]["auto_train"]["train_episodes"] = 1 # Force 1 full training for Optuna runs, it will prune if not good

    return config

def optuna_run(config: dict, direction: str = "maximize"):
    # You can set a sampler & pruner for speed
    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True, n_startup_trials=config_["settings"]["optuna"]["n_startup_trials"])
    pruner  = optuna.pruners.MedianPruner(n_warmup_steps=config_["settings"]["optuna"]["n_warmup_steps"])

    study = optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    study.optimize(objective_factory(config), n_trials=config_["settings"]["optuna"]["n_trials"])

    print("Best value:", study.best_value)
    print("Best params:", study.best_trial.params)
    return study

def objective_factory(config: dict):

    def objective(trial: optuna.Trial) -> float:
        # adapt ranges to your PPO implementation or other algos
        # do this per hand as I won't write this into the config
        # you should now what you want to test here any ways!
        # ---- sample your hyperparams here ----
        batch_size = [2 ** x for x in [11, 12, 13, 14, 15]]
        horizon = [2 ** x for x in [15, 16, 17, 18, 19]]
        hparams = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-3),
            # "batch_size": trial.suggest_categorical("batch_size", batch_size),
            # "horizon": trial.suggest_categorical("horizon", horizon),
            # "k_epochs": trial.suggest_int("k_epochs", 1, 20),
            # "entropy_coeff": trial.suggest_float("entropy_coeff", 1e-6, 1e-2, log=True),
            # "separate_optimizers": trial.suggest_categorical("separate_optimizers", [True, False]),
            # "gamma": trial.suggest_float("gamma", 0.97, 0.999),
            # "_lambda": trial.suggest_float("_lambda", 0.9, 0.99),
            # "eps_clip": trial.suggest_float("eps_clip", 0.1, 0.3),
            # "share_encoder": trial.suggest_categorical("share_encoder", [True, False]),
        }

        # FlyAgent rewards
        # rewards = {
        #     "GoalReached": trial.suggest_float("GoalReached", 0.2, 5, log=True),
        #     "BoundaryTerminal": trial.suggest_float("BoundaryTerminal", -5, -0.2),
        #     "Extinguish": trial.suggest_float("Extinguish", 1e-3, 0.2, log=True),
        #     "TimeOut": trial.suggest_float("TimeOut", -5, -0.2),
        #     "DistanceImprovement": trial.suggest_float("DistanceImprovement", 1e-3, 0.5, log=True),
        #     "ProximityPenalty": trial.suggest_float("ProximityPenalty", -0.5, -1e-3),
        # }

        # PlannerAgent rewards
        # rewards = {
        #     "GoalReached": trial.suggest_float("GoalReached", 0.1, 5),
        #     "MapBurnedTooMuch": trial.suggest_float("MapBurnedTooMuch", -5, -0.1),
        #     "FlyingTowardsGroundStation": trial.suggest_float("FlyingTowardsGroundStation", -1, -1e-3),
        #     "SameGoalPenalty": trial.suggest_float("SameGoalPenalty", -2, -1e-4),
        #     "TimeOut": trial.suggest_float("TimeOut", -5, -0.2),
        #     "ExtinguishFires": trial.suggest_float("ExtinguishFires", 1e-3, 0.8),
        #     "FastExtinguish": trial.suggest_float("FastExtinguish", 1e-3, 0.8),
        # }

        overrides = {
            "hparams": hparams,
            #"rewards": rewards
        }

        # run the sim; pass trial to enable pruning reports
        score = sim(config, overrides=overrides, trial=trial)
        # helpful metadata
        trial.set_user_attr("models_dir", config["paths"]["model_directory"])
        return score
    return objective

if __name__ == '__main__':
    config_path = sys.argv[1] if len(sys.argv) > 1 else ""
    root_path = get_project_paths("root_path")
    if not config_path or not os.path.exists(config_path):
        config_path = os.path.join(root_path, 'config.yaml')
    print(f"Using config file: {config_path}")
    with open(config_path, 'r') as f:
        config_ = yaml.safe_load(f)
    if not config_["settings"]["optuna"]["use_optuna"]:
        metric = sim(config_)
        print(f"Final Objective: {metric}")
    else:
        direction = "maximize"
        if config_["settings"]["optuna"]["objective"] == "time_to_end":
            direction = "minimize"
        optuna_run(config_, direction)
