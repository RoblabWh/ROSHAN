import yaml
import optuna
import os, shutil
import sys
from utils import SimulationBridge, get_project_paths
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
    Asserts that the config dictionary doesn't have some wild configurations.
    """
    assert config["settings"]["mode"] in [0, 2], "Invalid mode in config.yaml. Must be 0 (GUI_RL), 2 (NoGUI_RL)"
    assert config["settings"]["hierarchy_type"] in ["fly_agent", "explore_agent", "planner_agent"], \
        "Invalid hierarchy_type in config.yaml. Must be 'fly_agent', 'explore_agent', or 'planner_agent'."
    assert config["settings"]["rl_mode"] in ["train", "eval"], "Invalid rl_mode in config.yaml. Must be 'train' or 'eval'."
    hierarchy_type = config["settings"]["hierarchy_type"]
    if hierarchy_type == "planner_agent":
        assert config["environment"]["agent"]["fly_agent"]["default_model_folder"] != "", \
            "default_model_folder for fly_agent cannot be empty in config.yaml when planner_agent is selected."
        assert config["environment"]["agent"]["fly_agent"]["default_model_name"] != "", \
            "default_model_name cannot be empty in config.yaml when planner_agent is selected."
    # Assert that resume is off when auto_train is on
    if config["settings"]["auto_train"]["use_auto_train"]:
        assert not config["settings"]["resume"], "resume cannot be True when auto_train is enabled in config.yaml."
        assert not config["settings"]["rl_mode"] == "eval", ("rl_mode cannot be 'eval' when auto_train is enabled in config.yaml.\n"
                                                             "While this configuration WOULD work, it is not intended to be used in this way, since it would automatically "
                                                             "resume training from the last checkpoint, which is not what you want when evaluating a model.")

    if config["settings"]["optuna"]["use_optuna"]:
        assert config["settings"]["optuna"]["objective"] in ["objective", "reward"], "Invalid objective in config.yaml for optuna. Must be 'objective' or 'reward'."
def sim(config : dict, overrides: dict = None, trial=None):

    config = inject_overrides(config, overrides, trial)

    assert_config(config)

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
    engine.Init(config["settings"]["mode"])
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
                user_input = input(f"Do you want to delete the contents of {sim_bridge.get('model_path')}? (y/n): ")
                if user_input.lower() != 'y':
                    print("Exiting program.")
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

    try:
        while engine.IsRunning() and sim_bridge.get("agent_online"):
            engine.Update()
            engine.Render()
            engine.HandleEvents()

            # C++ Python Interface & Controls
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

    used_algorithm = config["environment"]["agent"][config["settings"]["hierarchy_type"]]["algorithm"]
    algo_dict = config["algorithm"][used_algorithm]
    if "hparams" in overrides:
        for key, value in overrides["hparams"].items():
            if key in algo_dict:
                algo_dict[key] = value

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
        hparams = {
            # "lr": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048]),
            "horizon": trial.suggest_categorical("horizon", [2048, 4096, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576]),
            "k_epochs": trial.suggest_int("k_epochs", 1, 20),
            # "entropy_coeff": trial.suggest_float("entropy_coeff", 1e-6, 1e-2, log=True),
            # "separate_optimizers": trial.suggest_categorical("separate_optimizers", [True, False]),
            # "gamma": trial.suggest_float("gamma", 0.97, 0.999),
            # "_lambda": trial.suggest_float("_lambda", 0.9, 0.99),
            # "eps_clip": trial.suggest_float("eps_clip", 0.1, 0.3),
            # "share_encoder": trial.suggest_categorical("share_encoder", [True, False]),
        }

        overrides = {
            "hparams": hparams,
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
        optuna_run(config_)
