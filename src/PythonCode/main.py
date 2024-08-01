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
from memory import Memory
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


def log_outcome(rewards, terminals, dones, percent_burned, stats):
    # Log stats
    stats['reward'][-1] += rewards[0]
    stats['time'][-1] += 1

    if terminals[0]:
        print("Episode: {} finished with terminal state".format(stats['episode'][-1] + 1))
        if dones[0]:  # Drone died
            print("Drone died.\n")
            stats['died'][-1] += 1
        else:  # Drone reached goal
            print("Drone extinguished all fires.\n")
            stats['reached'][-1] += 1

        stats['perc_burn'][-1] = percent_burned
        # Print stats
        print("-----------------------------------")
        print("Episode: {}".format(stats['episode'][-1] + 1))
        print("Reward: {}".format(stats['reward'][-1]))
        print("Time: {}".format(stats['time'][-1]))
        print("Percentage burned: {}".format(stats['perc_burn'][-1]))
        print("Died: {}".format(stats['died'][-1]))
        print("Reached: {}".format(stats['reached'][-1]))
        print("Total average time: {}".format(np.mean(stats['time'])))
        print("Total average percentage burned: {}".format(np.mean(stats['perc_burn'])))
        print("Total average reward: {}".format(np.mean(stats['reward'])))
        print("Total died: {}".format(sum(stats['died'])))
        print("Total reached: {}".format(sum(stats['reached'])))
        print("-----------------------------------\n\n")

        # Reset stats
        stats['episode'].append(stats['episode'][-1] + 1)
        stats['died'].append(0)
        stats['perc_burn'].append(0)
        stats['reached'].append(0)
        stats['time'].append(0)
        stats['reward'].append(0)

        return True
    return False


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Lists alls the functions in the EngineCore class
    # print(dir(EngineCore))

    horizon = 250
    mini_batch_size = 50

    # Ugly globals
    t = -1
    train_step = 0
    eval_step = 0

    # 0: GUI_RL, 2: NoGUI_RL
    mode = 0

    # If the LLM support is enabled or not
    llm_support = True
    model_directory = config['models_directory']
    model_name = "best.pth"
    console = ""
    # Folder the models are stored in
    if os.path.isfile(os.path.abspath(model_directory)):
        raise ValueError("Model path must be a directory, not a file")
    if not os.path.exists(os.path.abspath(model_directory)):
        console += f"Creating model directory under {os.path.abspath(model_directory)}"
        os.makedirs(os.path.abspath(model_directory))

    # If the agent should be trained or not, if not the agent will act with the best policy if it can be loaded
    train = True
    max_train = 200 # Number of Updates to perform before stopping training
    max_eval = 1000 # Number of Environments to run before stopping evaluation

    # This map is used in NoGUI setup, if left empty("") the default map will be used. Has no impact on GUI Setup
    map = "/home/nex/Dokumente/Code/ROSHAN/maps/Small2.tif"
    map = ""

    # Stats to log
    stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}

    # RL_Status, sending to C++
    status = {"train": train, "model_path": model_directory, "model_name": model_name, "console": console}

    if llm_support:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")
        #llm = LLMPredictorAPI("Qwen/Qwen2-1.5B")
    engine = firesim.EngineCore()
    memory = Memory(max_size=horizon+1)
    logger = Logger(log_dir='./logs', horizon=horizon)

    logger.set_logging(True)

    engine.Init(mode, map)

    # Wait for the model to be initialized
    while not engine.ModelInitialized():
        engine.HandleEvents()
        engine.Update()
        engine.Render()

    # Now get the view range and time steps
    view_range = engine.GetViewRange() + 1
    time_steps = engine.GetTimeSteps()

    agent = Agent('ppo', logger, vision_range=view_range, time_steps=time_steps, model_path=model_directory, model_name=model_name)
    if not train:
        weights = os.path.join(config['module_directory'], 'best.pth')
        train = not agent.algorithm.load_model(weights)

    engine.SendRLStatusToModel(status)

    while engine.IsRunning():
        engine.HandleEvents()
        engine.Update()
        engine.Render()
        if engine.AgentIsRunning():
            t += 1
            if t == 0:
                next_obs = restructure_data(engine.GetObservations())
                agent_cnt = next_obs[0].shape[0]
                next_terminals = [False] * agent_cnt

            obs = next_obs
            terminals = next_terminals

            if train:
                actions, action_logprobs = agent.act(obs)
                drone_actions = agent.get_action(actions)
                next_observations, rewards, next_terminals, _, percent_burned = engine.Step(drone_actions)
                next_obs = restructure_data(next_observations)
                memory.add(obs, actions, action_logprobs, rewards, terminals)

                # Logging and sending data
                logger.episode_log(next_obs, next_terminals[0], percent_burned)
                obs_collected = f'{len(memory)}/{horizon}'
                status["ObsCollected"] = obs_collected
                status["Train Step"] = train_step

                if agent.should_train(memory, horizon, t):
                    console += agent.update(memory, horizon, mini_batch_size, next_obs, next_terminals)
                    logger.log()
                    train_step += 1
                    if train_step >= max_train:
                        print("Training finished, after {} training steps".format(train_step))
                        break
            else:
                actions = agent.act_certain(obs)
                drone_actions = agent.get_action(actions)
                next_observations, rewards, terminals, dones, percent_burned = engine.Step(drone_actions)
                next_obs = restructure_data(next_observations)
                if log_outcome(rewards, terminals, dones, percent_burned, stats):
                    eval_step += 1
                    if eval_step >= max_eval:
                        break
            status["console"] = console
            engine.SendRLStatusToModel(status)
        else:
            obs, rewards = None, None

        user_input = engine.GetUserInput()
        if llm_support and user_input != "":
            llm(engine, user_input, obs, rewards)

    engine.Clean()
