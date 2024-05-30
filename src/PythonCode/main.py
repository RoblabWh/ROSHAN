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
        maps = np.array([state.GetMap() for state in drone_states])
        positions = np.array([state.GetPositionNorm() for state in drone_states])

        all_terrains.append(terrains)
        all_fire_statuses.append(fire_statuses)
        all_velocities.append(velocities)
        all_maps.append(maps)
        all_positions.append(positions)

    return np.array(all_terrains), np.array(all_fire_statuses), np.array(all_velocities), np.array(all_maps), np.array(
        all_positions)


def log_outcome(rewards, terminals, dones, stats):
    # Log stats
    stats['reward'][-1] += rewards[0]
    stats['time'][-1] += 1

    if terminals[0]:
        print("Episode: {} finished with terminal state".format(stats['episode'][-1]))
        if dones[0]:  # Drone died
            print("Drone died.\n")
            stats['died'][-1] += 1
        else:  # Drone reached goal
            print("Drone extinguished all fires.\n")
            stats['reached'][-1] += 1

        # Print stats
        print("-----------------------------------")
        print("Episode: {}".format(stats['episode'][-1]))
        print("Reward: {}".format(stats['reward'][-1]))
        print("Time: {}".format(stats['time'][-1]))
        print("Died: {}".format(stats['died'][-1]))
        print("Reached: {}".format(stats['reached'][-1]))
        print("Total average time: {}".format(np.mean(stats['time'])))
        print("Total average reward: {}".format(np.mean(stats['reward'])))
        print("Total died: {}".format(sum(stats['died'])))
        print("Total reached: {}".format(sum(stats['reached'])))
        print("-----------------------------------\n\n")

        # Reset stats
        stats['episode'].append(stats['episode'][-1] + 1)
        stats['died'].append(0)
        stats['reached'].append(0)
        stats['time'].append(0)
        stats['reward'].append(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # Lists alls the functions in the EngineCore class
    # print(dir(EngineCore))
    batch_size = 2048
    mini_batch_size = 64
    t = -1

    # Stats to log
    stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0]}

    llm_support = True

    if llm_support:
        from llmsupport import LLMPredictorAPI
        llm = LLMPredictorAPI("mistralai/Mistral-7B-Instruct-v0.3")

    engine = firesim.EngineCore()
    memory = Memory()
    logger = Logger(log_dir='./logs', log_interval=1)
    agent = Agent('ppo', logger)
    train = False
    if not train:
        train = not agent.algorithm.load_model('best.pth')
    logger.set_logging(True)
    if memory.max_size <= batch_size:
        warnings.warn("Memory size is smaller than horizon. Setting horizon to memory size.")
        horizon = memory.max_size - 1
    engine.Init(0)

    while engine.IsRunning():
        engine.HandleEvents()
        engine.Update()
        engine.Render()
        if engine.AgentIsRunning():
            t += 1  # TODO use simulation time instead of timesteps
            observations = engine.GetObservations()
            obs = restructure_data(observations)

            if train:
                actions, action_logprobs = agent.act(obs, t)
                drone_actions = []
                for activation in actions:
                    drone_actions.append(
                        firesim.DroneAction(activation[0], activation[1], int(np.round(activation[2]))))
                next_observations, rewards, terminals, _ = engine.Step(drone_actions)
                next_obs = restructure_data(next_observations)
                memory.add(obs, actions, action_logprobs, rewards, next_obs, terminals)
                if agent.should_train(memory, batch_size, t):
                    agent.update(memory, batch_size, mini_batch_size)
                    logger.log()
            else:
                actions = agent.act_certain(obs)
                drone_actions = []
                for activation in actions:
                    drone_actions.append(
                        firesim.DroneAction(activation[0], activation[1], int(np.round(activation[2]))))
                next_observations, rewards, terminals, dones = engine.Step(drone_actions)
                if not train:
                    log_outcome(rewards, terminals, dones, stats)
        else:
            obs, rewards = None, None

        user_input = engine.GetUserInput()
        if llm_support and user_input != "":
            llm(engine, user_input, obs, rewards)

    engine.Clean()
