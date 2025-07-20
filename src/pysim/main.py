import json
import yaml
import os
import sys
from pathlib import Path
from utils import find_project_root

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = find_project_root(Path(script_directory))

with open(Path(root_directory / 'config.json'), 'r') as f:
    std_paths = json.load(f)

with open(Path(root_directory / 'parameter_config.yaml'), 'r') as f:
    config = yaml.safe_load(f)

sys.path.insert(0, std_paths['module_directory'])

# Change the current working directory, so that the python script has the same folder as the c++ executable
os.chdir(std_paths['module_directory'])

# Check Python Version
pythonversion = str(sys.version_info[0]) + str(sys.version_info[1])

# Find Firesim Module in the current directory
firesimverison = [f for f in os.listdir(std_paths['module_directory']) if f[:7] == 'firesim'][0].split('-')[1]

if pythonversion != firesimverison:
    raise ValueError(f"Python Version {pythonversion} does not match Firesim Version {firesimverison}."
                     f"Activate your conda environment before compiling the library.")

import firesim
from agent_handler import AgentHandler
from hierarchy_manager import HierarchyManager

def build_status(console):

    hierarchy = config["settings"]["hierarchy_type"]
    environment = config["environment"]
    agent = environment["agent"]["fly_agent"] if hierarchy == "fly_agent" else environment["agent"]["explore_agent"] if hierarchy == "explore_agent" else environment["agent"]["planner_agent"]
    status = {# Non-Settable Parameters (either flags or communication with C++)
              "console": console,
              "obs_collected": 0, # Used by GUI to show the number of collected observations
              "min_update": 0,  # How many obs before updating the policy? Decided by the RL Algorithm
              "agent_online": True,
              "train_episode": 0,  # Current training episode
              "train_step": 0,  # How often did you train?
              "current_episode": 0,
              "policy_updates": 0,  # How often did you update the policy?
              "objective": 0,  # Tracking the Percentage of the Objective
              "best_objective": 0,  # Best Objective so far
              # Settable Parameters (can be changed by the user, best through Config)
              "rl_mode": config["settings"]["rl_mode"],  # "train" or "eval"
              "model_path": std_paths['models_directory'],
              "model_name": "my_model.pt",
              "num_agents": agent["num_agents"],  # Number of agents in the environment
              "flyAgentTimesteps": environment["agent"]["fly_agent"]["time_steps"],  # Number of timesteps for the FlyAgent
              "exploreAgentTimesteps": environment["agent"]["explore_agent"]["time_steps"],
              "rl_algorithm": "PPO",  # RL Algorithm to use, either PPO, IQL, TD3
              "auto_train": False,  # If True, the agent will train several episodes and then evaluate
              "train_episodes": 1,  # Number of total trainings containing each max_train steps
              "max_eval": 3,  # Number of Environments to run before stopping evaluation
              "max_train": 1,  # Number of train_steps before stopping training
              "hierarchy_type": hierarchy,  # Either fly_agent, ExplorationAgent
              "resume": False,  # If True, the agent will resume training from the last checkpoint
              }

    return status

def main():
    # Lists alls the functions in the EngineCore class
    # print(dir(EngineCore))

    # Steps in the simulation, used for training and evaluation
    steps = 0

    # Folder the models are stored in
    console = ""

    # RL_Status Dictionary, sending back and forth to C++
    status = build_status(console)

    llm = None
    if config["settings"]["llm_support"]:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")
        #llm = LLMPredictorAPI("Qwen/Qwen2-1.5B")

    # Initialize the EngineCore and send the RL_Status
    engine = firesim.EngineCore()
    engine.Init(config["settings"]["mode"])
    engine.SendRLStatusToModel(status)
    if config["settings"]["mode"] == 0 or config["settings"]["use_default"] == 1:
        engine.InitializeMap(Path(root_directory).joinpath("maps").joinpath(config["paths"]["init_map"]).__str__()
                             if config["paths"]["init_map"] is not None else "")

    # Wait for the initial mode selection from the user
    while not engine.InitialModeSelectionDone():
        engine.HandleEvents()
        engine.Update()
        engine.Render()
    status = engine.GetRLStatusFromModel()

    # If the user changed settings in the GUI, we need to update the config OBJECT
    if config["settings"]["use_default"] is False:
        # Update the config with the new settings from the GUI
        config["settings"]["hierarchy_type"] = status["hierarchy_type"]
        agent_type = config["settings"]["hierarchy_type"]
        agent_settings = config["environment"]["agent"][agent_type]
        agent_settings["default_model_folder"] = status["model_path"]
        agent_settings["default_model_name"] = status["model_name"]
        # config["settings"]["rl_algorithm"] = status["rl_algorithm"]
        # config["settings"]["rl_mode"] = status["rl_mode"]
        # config["settings"]["auto_train"] = status["auto_train"]
        # config["settings"]["train_episodes"] = status["train_episodes"]
        # config["settings"]["max_eval"] = status["max_eval"]
        # config["settings"]["max_train"] = status["max_train"]

    # Now get the view range and time steps from the engine, these parameters are currently set in the model_params
    # TODO Keep this in the model_params as soft hidden param since it meddles with the model structure?

    # Create the Agent Object, this is used by the hierarchy manager which might spawn other low_level agents
    agent = AgentHandler(status=status, config=config, logdir=config["paths"]["log_directory"])
    status["console"] += agent.load_model(status=status)

    hierarchy_manager = HierarchyManager(status, config, agent)

    engine.SendRLStatusToModel(status)

    while engine.IsRunning() and status["agent_online"]:
        engine.HandleEvents()
        engine.Update()
        engine.Render()

        # C++ Python Interface & Controls
        status = engine.GetRLStatusFromModel()
        hierarchy_manager.update_status(status)

        if engine.AgentIsRunning() and status["agent_online"]:
            # Initial Observation
            hierarchy_manager.restruct_current_obs(engine.GetObservations())

            if status["rl_mode"] == "train":
                hierarchy_manager.train(status, engine)
            else:
                hierarchy_manager.eval(status, engine)
            engine.SendRLStatusToModel(status)

            steps += 1

        user_input = engine.GetUserInput()
        if config["settings"]["llm_support"] and user_input != "":
            # TODO DEPRECATED
            llm(engine, user_input, 0, 0)

    engine.Clean()

# def optimize(trial):
#
#     # --- Sample hyperparameters
#     # learning_rate = trial.suggest_loguniform("lr", 1e-5, 1e-3)
#     # clip_range = trial.suggest_uniform("clip_range", 0.1, 0.3)
#     # ent_coef = trial.suggest_loguniform("ent_coef", 1e-6, 1e-2)
#     # gamma = trial.suggest_uniform("gamma", 0.95, 0.999)
#     k_epochs = trial.suggest_int("k_epochs", 3, 60)
#     batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512, 1024, 2048, 4096])
#     # horizon = trial.suggest_int("horizon", 4096, 25600)
#
#     config = {
#         "models_directory": f"../models/optuna_models/trial_{trial.number}/",
#         "log_directory": f"../logs/optuna_logs/trial_{trial.number}/"
#     }
#
#     os.makedirs(config["models_directory"], exist_ok=True)
#     os.makedirs(config["log_directory"], exist_ok=True)
#
#
#     # RL_Status Dictionary, sending back and forth to C++
#     status = {"rl_mode": "train", # "train" or "eval"
#               "model_path": config['models_directory'],
#               "model_name": "model.pt",
#               "console": "",
#               "agent_online": True,
#               "obs_collected": 0,
#               "min_update": 0,  # How many obs before updating the policy?
#               "num_agents": 1,
#               "flyAgentTimesteps": 3,
#               "exploreAgentTimesteps": 20,
#               "frame_skips": 1,
#               "rl_algorithm": "PPO", # RL Algorithm to use, either PPO, IQL, TD3
#               "auto_train": True, # If True, the agent will train several episodes and then evaluate
#               "objective": 0, # Tracking the Percentage of the Objective
#               "best_objective": 0, # Best Objective so far
#               "train_episodes": 1, # Number of total trainings containing each max_train steps
#               "train_episode": 0, # Current training episode
#               "train_step": 0, # How often did you train?
#               "policy_updates": 0, # How often did you update the policy?
#               "max_eval": 100, # Number of Environments to run before stopping evaluation
#               "max_train": 30, # Number of train_steps before stopping training
#               "current_episode": 0,
#               "hierarchy_type": "ExploreAgent", # Either fly_agent, ExplorationAgent
#               "resume": False, # If True, the agent will resume training from the last checkpoint
#               }
#
#     # Globals
#
#     # 0: GUI_RL, 2: NoGUI_RL
#     mode = 2
#     # This map is used in NoGUI setup, if left empty("") the default map will be used. Has no impact on GUI Setup
#     #init_map = "/home/nex/Dokumente/Code/ROSHAN/maps/21x21_Field.tif"
#     init_map = ""
#
#     # Initialize the EngineCore and send the RL_Status
#     engine = firesim.EngineCore()
#     engine.Init(mode)
#     engine.InitializeMap(init_map)
#     engine.SendRLStatusToModel(status)
#
#     # Wait for the initial mode selection from the user
#     while not engine.InitialModeSelectionDone():
#         engine.HandleEvents()
#         engine.Update()
#         engine.Render()
#     status = engine.GetRLStatusFromModel()
#
#     # Now get the view range and time steps from the engine, these parameters are currently set in the model_params
#     # TODO Keep this in the model_params as soft hidden param since it meddles with the model structure?
#     view_range = engine.GetViewRange(status["hierarchy_type"]) + 1
#     map_size = engine.GetMapSize()
#     time_steps = status["flyAgentTimesteps"] if status["hierarchy_type"] == "fly_agent" else status["exploreAgentTimesteps"]
#
#     # Create the Agent Object, this is used by the hierachy manager which might spawn other low_level agents
#     agent = AgentHandler(status=status, algorithm=status['rl_algorithm'], vision_range=view_range, map_size=map_size, time_steps=time_steps, logdir=config['log_directory'])
#     status["console"] += agent.load_model(status=status)
#
#     # Inject hyperparameters
#     ppo = agent.algorithm
#     # ppo.lr = learning_rate
#     # ppo.gamma = gamma
#     # ppo.clip_range = clip_range
#     # ppo.entropy_coeff = ent_coef
#     ppo.k_epochs = k_epochs
#     ppo.batch_size = batch_size
#     # ppo.horizon = horizon
#
#     hierarchy_manager = HierarchyManager(status, agent)
#
#     engine.SendRLStatusToModel(status)
#     try:
#         while engine.IsRunning() and status["agent_online"]:
#             engine.HandleEvents()
#             engine.Update()
#             engine.Render()
#
#             # C++ Python Interface & Controls
#             status = engine.GetRLStatusFromModel()
#             hierarchy_manager.update_status(status)
#
#             if engine.AgentIsRunning() and status["agent_online"]:
#                 # Initial Observation
#                 hierarchy_manager.restruct_current_obs(engine.GetObservations())
#
#                 if status["rl_mode"] == "train":
#                     hierarchy_manager.train(status, engine)
#                 else:
#                     hierarchy_manager.eval(status, engine)
#                 engine.SendRLStatusToModel(status)
#
#         engine.Clean()
#
#         # Return the objective value for Optuna
#         return sum(agent.stats["reward"]) / len(agent.stats["reward"])
#
#     except Exception as e:
#         engine.Clean()
#         print(f"[Trial {trial.number}] Error: {e}")
#         return -float("inf")

# def optuna():
#     import optuna
#     study = optuna.create_study(direction="maximize")
#     study.optimize(optimize, n_trials=100)
#     print("Best trial:", study.best_trial.params)
#     print("Best value:", study.best_trial.value)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
    #optuna()

