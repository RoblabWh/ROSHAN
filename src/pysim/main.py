import json
import os
import sys
import warnings
from pathlib import Path
import numpy as np
from utils import find_project_root

script_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = find_project_root(Path(script_directory))

with open(Path(root_directory / 'config.json'), 'r') as f:
    config = json.load(f)

sys.path.insert(0, config['module_directory'])

# Change the current working directory, so that the python script has the same folder as the c++ executable
os.chdir(config['module_directory'])

import firesim
from agent import Agent
from memory import SwarmMemory
from utils import Logger


def restructure_data(observations_):
    all_terrains, all_fire_statuses, all_velocities, all_maps, all_positions = [], [], [], [], []

    for deque in observations_:
        drone_states = np.array([state for state in deque if isinstance(state, firesim.DroneState)])
        if len(drone_states) == 0:
            continue

        terrains = np.array([state.GetTerrainNorm() for state in drone_states])
        fire_statuses = np.array([state.GetFireStatus() for state in drone_states])
        velocities = np.array([state.GetVelocityNorm() for state in drone_states])
        #maps = np.array([state.GetMap() for state in drone_states])
        positions = np.array([state.GetPositionNorm() for state in drone_states])

        all_terrains.append(terrains)
        all_fire_statuses.append(fire_statuses)
        all_velocities.append(velocities)
        #all_maps.append(maps)
        all_positions.append(positions)

    return np.array(all_terrains), np.array(all_fire_statuses), np.array(all_velocities), np.array(all_positions)

    # return np.array(all_terrains), np.array(all_fire_statuses), np.array(all_velocities), np.array(all_maps), np.array(
    #     all_positions)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Lists alls the functions in the EngineCore class
    # print(dir(EngineCore))

    # Ugly globals
    t = -1
    train_step = 0
    eval_step = 0
    llm_support = True

    # 0: GUI_RL, 2: NoGUI_RL
    mode = 0

    # This map is used in NoGUI setup, if left empty("") the default map will be used. Has no impact on GUI Setup
    map = "/home/nex/Dokumente/Code/ROSHAN/maps/Small2.tif"
    #map = ""

    console = ""
    # Folder the models are stored in
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
              "horizon": 5000,
              "batch_size": 50,
              "auto_train": False, # If True, the agent will train several episodes and then evaluate
              "train_episodes": 10, # Number of total trainings containing each max_train steps
              "train_episode": 0, # Current training episode
              "train_step": 0,
              "max_eval": 1000, # Number of Environments to run before stopping evaluation
              "max_train": 100 # Number of Updates to perform before stopping training
              }

    if llm_support:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")
        #llm = LLMPredictorAPI("Qwen/Qwen2-1.5B")

    # Initialize the EngineCore and send the RL_Status
    engine = firesim.EngineCore()
    engine.Init(mode, map)
    engine.SendRLStatusToModel(status)

    # Wait for the initial mode selection from the user
    while not engine.InitialModeSelectionDone():
        engine.HandleEvents()
        engine.Update()
        engine.Render()
    status = engine.GetRLStatusFromModel()

    # Memory and Logger Objects, setting logging will create a tensorboardX writer
    memory = SwarmMemory(num_agents=status["num_agents"], max_size=status["horizon"])
    logger = Logger(log_dir='./logs', horizon=status["horizon"])
    logger.set_logging(True)

    # Now get the view range and time steps from the engine, these parameters are currently set in the model_params
    # TODO Keep this in the model_params as soft hidden param since it meddles with the model structure?
    view_range = engine.GetViewRange() + 1
    time_steps = engine.GetTimeSteps()

    # Create the Agent Object, which might actually hold multiple agents...
    agent = Agent(status=status, algorithm='ppo', logger=logger, vision_range=view_range, time_steps=time_steps)

    # agent.set_paths(status["model_path"], status["model_name"])
    status["console"] = agent.load_model(status=status, resume=False, train_=status["rl_mode"])

    engine.SendRLStatusToModel(status)

    while engine.IsRunning():
        engine.HandleEvents()
        engine.Update()
        engine.Render()

        # C++ Python Interface
        status = engine.GetRLStatusFromModel()
        memory = agent.update_status(status, memory)

        if engine.AgentIsRunning() and status["agent_online"]:
            t += 1
            if t == 0:
                next_obs = restructure_data(engine.GetObservations())
                agent_cnt = next_obs[0].shape[0]
                next_terminals = [False] * agent_cnt

            obs = next_obs
            terminals = next_terminals

            if status["rl_mode"] == "train":
                actions, action_logprobs = agent.act(obs)
                drone_actions = agent.get_action(actions)
                next_observations, rewards, next_terminals, _, percent_burned = engine.Step(drone_actions)
                next_obs = restructure_data(next_observations)
                memory.add(obs, actions, action_logprobs, rewards, terminals)

                # Logging and sending data
                logger.episode_log(next_obs, next_terminals[0], percent_burned)
                status["obs_collected"] = len(memory) + 1

                if agent.should_train(memory):
                    agent.update(status, memory, mini_batch_size=status["batch_size"], next_obs=next_obs, next_terminals=next_terminals)
                    logger.log(status)
            else:
                actions = agent.act_certain(obs)
                drone_actions = agent.get_action(actions)
                next_observations, rewards, terminals, dones, percent_burned = engine.Step(drone_actions)
                next_obs = restructure_data(next_observations)
                agent.log_episode(status, rewards, terminals, dones, percent_burned)
            engine.SendRLStatusToModel(status)
        else:
            obs, rewards = None, None

        user_input = engine.GetUserInput()
        if llm_support and user_input != "":
            llm(engine, user_input, obs, rewards)

    engine.Clean()
