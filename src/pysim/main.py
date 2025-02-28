import json
import os
import sys
from pathlib import Path
from utils import find_project_root

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = find_project_root(Path(script_directory))

with open(Path(root_directory / 'config.json'), 'r') as f:
    config = json.load(f)

sys.path.insert(0, config['module_directory'])

# Change the current working directory, so that the python script has the same folder as the c++ executable
os.chdir(config['module_directory'])

# Check Python Version
pythonversion = str(sys.version_info[0]) + str(sys.version_info[1])

# Find Firesim Module in the current directory
firesimverison = [f for f in os.listdir(config['module_directory']) if f[:7] == 'firesim'][0].split('-')[1]

if pythonversion != firesimverison:
    raise ValueError(f"Python Version {pythonversion} does not match Firesim Version {firesimverison}."
                     f"Activate your conda environment before compiling the library.")

import firesim
from agent_handler import AgentHandler
from hierarchy_manager import HierarchyManager
from memory import SwarmMemory

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Lists alls the functions in the EngineCore class
    # print(dir(EngineCore))

    # Globals
    steps = 0
    llm_support = False

    # 0: GUI_RL, 2: NoGUI_RL
    mode = 0

    # This map is used in NoGUI setup, if left empty("") the default map will be used. Has no impact on GUI Setup
    init_map = "/home/nex/Dokumente/Code/ROSHAN/maps/21x21_Field.tif"
    #init_map = ""


    # Folder the models are stored in
    console = ""
    if os.path.isfile(os.path.abspath(config['models_directory'])):
        raise ValueError("Model path must be a directory, not a file")
    if not os.path.exists(os.path.abspath(config['models_directory'])):
        console += f"Creating model directory under {os.path.abspath(config['models_directory'])}"
        os.makedirs(os.path.abspath(config['models_directory']))

    # RL_Status Dictionary, sending back and forth to C++
    status = {"rl_mode": "", # "train" or "eval"
              "model_path": config['models_directory'],
              "model_name": "my_model.pt",
              "console": console,
              "agent_online": True,
              "obs_collected": 0,
              "num_agents": 1,
              "n_steps": 1024, #128 #TODO: Deprecated
              "horizon": 1024,#12800,
              "batch_size": 512,
              "auto_train": False, # If True, the agent will train several episodes and then evaluate
              "objective": 0, # Tracking the Percentage of the Objective
              "best_objective": 0, # Best Objective so far
              "train_episodes": 10, # Number of total trainings containing each max_train steps
              "train_episode": 0, # Current training episode
              "train_step": 0,
              "max_eval": 1000, # Number of Environments to run before stopping evaluation
              "max_train": 1000, # Number of Updates to perform before stopping training
              "K_epochs": 18,
              "current_episode": 0,
              "agent_type": "FlyAgent", # Either FlyAgent, ExplorationAgent
              "resume": False, # If True, the agent will resume training from the last checkpoint
              }

    if llm_support:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")
        #llm = LLMPredictorAPI("Qwen/Qwen2-1.5B")
    else:
        llm = None

    # Initialize the EngineCore and send the RL_Status
    engine = firesim.EngineCore()
    engine.Init(mode, init_map)
    engine.SendRLStatusToModel(status)

    # Wait for the initial mode selection from the user
    while not engine.InitialModeSelectionDone():
        engine.HandleEvents()
        engine.Update()
        engine.Render()
    status = engine.GetRLStatusFromModel()

    # Now get the view range and time steps from the engine, these parameters are currently set in the model_params
    # TODO Keep this in the model_params as soft hidden param since it meddles with the model structure?
    view_range = engine.GetViewRange() + 1
    map_size = engine.GetMapSize()
    time_steps = engine.GetTimeSteps()

    # Create the Agent Object, this is used by the hierachy manager which might spawn other low_level agents
    agent = AgentHandler(status=status, algorithm='ppo', vision_range=view_range, map_size=map_size, time_steps=time_steps, logdir=config['log_directory'])
    status["console"] += agent.load_model(status=status)

    hierarchy_manager = HierarchyManager(status, agent)

    engine.SendRLStatusToModel(status)

    while engine.IsRunning():
        engine.HandleEvents()
        engine.Update()
        engine.Render()

        # C++ Python Interface & Controls
        status = engine.GetRLStatusFromModel()
        hierarchy_manager.update_status(status)

        if engine.AgentIsRunning() and status["agent_online"]:
            # Initial Observation
            hierarchy_manager.initial_observation(engine.GetObservations())

            if status["rl_mode"] == "train":
                hierarchy_manager.train(status, engine, steps)
            else:
                hierarchy_manager.eval(status, engine)
            steps += 1
            engine.SendRLStatusToModel(status)
        else:
            if not status["agent_online"]:
                status["console"] += "Autotrain done! Yippie"

        user_input = engine.GetUserInput()
        if llm_support and user_input != "":
            # TODO DEPRECATED
            llm(engine, user_input, 0, 0)

    engine.Clean()
