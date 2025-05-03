from algorithms.ppo import PPO
from algorithms.iql import IQL
from algorithms.td3 import TD3
from algorithms.rl_config import RLConfig, PPOConfig, IQLConfig, TD3Config
import firesim
from utils import Logger
import numpy as np
import os
from memory import SwarmMemory
from explore_agent import ExploreAgent
from flying_agent import FlyAgent


class AgentHandler:
    def __init__(self, status, algorithm: str = 'PPO', vision_range=21, map_size=50, time_steps=None, logdir='./logs'):

        self.agent_actions = None
        if time_steps is None:
            print("Time steps not set, what are you doing!?")
            time_steps = [3, 32]

        supported_algos = ['PPO', 'IQL', 'TD3']
        assert algorithm in supported_algos, f"Algorithm {algorithm} not supported, only {supported_algos} are supported"

        self.algorithm_name = algorithm
        self.env_reset = False
        self.mode = status["rl_mode"]
        self.max_eval = status["max_eval"]

        # Used for Evaluation
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        self.logger = Logger(log_dir=logdir)
        self.initialized = False #TODO Maybe Unused
        self.eval_steps = 0

        # Agent Type is either FlyAgent or ExploreAgent
        self.agent_type = self.get_agent_from_type(status["hierarchy_type"])

        # On which Level of the Hierarchy is the agent
        self.hierarchy_level = self.agent_type.get_hierachy_level()
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False

        # Use Intrinsic Reward according to: https://arxiv.org/abs/1810.12894
        self.use_intrinsic_reward = self.agent_type.use_intrinsic_reward

        # Other variables
        self.env_step = 0

        if self.use_intrinsic_reward:
            self.agent_type.initialize_rnd_model(vision_range, drone_count=status["num_agents"],
                                                 map_size=map_size, time_steps=time_steps)
        self.current_obs = None
        if algorithm == 'PPO':
            config = PPOConfig(algorithm=algorithm,
                               model_path=status["model_path"],
                               model_name=status["model_name"],
                               vision_range=vision_range,
                               drone_count=status["num_agents"],
                               map_size=map_size,
                               time_steps=time_steps,
                               use_next_obs=False
                               )
            self.algorithm = PPO(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=config)
            self.initialized = True
            status["console"] += "PPO agent initialized\n"
        elif algorithm == 'IQL':
            config = IQLConfig(algorithm=algorithm,
                              model_path=status["model_path"],
                              model_name=status["model_name"],
                              vision_range=vision_range,
                              drone_count=status["num_agents"],
                              map_size=map_size,
                              time_steps=time_steps,
                              action_dim=self.agent_type.action_dim,
                              clear_memory=False
                              )
            self.algorithm = IQL(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=config)
            self.initialized = True
            status["console"] += "IQL agent initialized\n"
        elif algorithm == 'TD3':
            config = TD3Config(algorithm=algorithm,
                                 model_path=status["model_path"],
                                 model_name=status["model_name"],
                                 vision_range=vision_range,
                                 drone_count=status["num_agents"],
                                 map_size=map_size,
                                 time_steps=time_steps,
                                 action_dim=self.agent_type.action_dim,
                                 clear_memory=False
                                 )
            self.algorithm = TD3(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=config)
        self.use_next_obs = self.algorithm.use_next_obs
        self.memory = SwarmMemory(max_size=self.algorithm.memory_size,
                                  num_agents=status["num_agents"],
                                  action_dim=self.algorithm.action_dim,
                                  use_intrinsic_reward=self.use_intrinsic_reward,
                                  use_next_obs=self.use_next_obs)
        status["console"] += f"{status['num_agents']} agents of Type: {self.agent_type.name} initialized\n"
        status["console"] += f"Training with {self.algorithm_name} algorithm\n"

    @staticmethod
    def get_agent_from_type(agent_type):
        if agent_type not in ["FlyAgent", "ExploreAgent"]:
            raise ValueError("Invalid agent type, must be either 'FlyAgent' "
                             "or 'ExploreAgent', was: {}".format(agent_type))
        if agent_type == "FlyAgent":
            return FlyAgent()
        if agent_type == "ExploreAgent":
            return ExploreAgent()

    def reset(self, status):
        if status["train_episode"] >= status["train_episodes"]:
            status["agent_online"] = False
            status["console"] += "Training finished, after {} training episodes\n".format(status["train_episode"])
            print("Training finished, after {} training episodes".format(status["train_episode"]))
            status["console"] += "Agent is offline\n"
            print("Agent is offline")
        else:
            status["console"] += ("Resume with next training "
                                  "step {}/{}\n").format(status["train_episode"] + 1, status["train_episodes"])
            status["train_step"] = 0
            self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
            self.hierarchy_steps = 0
            self.env_reset = True
            self.algorithm.reset()
            self.logger.reset_bests()
            self.initialized = True

    def should_train(self):
        if self.algorithm_name == 'PPO':
            return len(self.memory) >= self.algorithm.horizon
        elif self.algorithm_name == 'IQL':
            return len(self.memory) >= self.algorithm.min_memory_size and self.env_step % self.algorithm.policy_freq == 0
        elif self.algorithm_name == 'TD3':
            return len(self.memory) >= self.algorithm.min_memory_size
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(self.algorithm_name))

    def load_model(self, status):
        train = True if status["rl_mode"] == "train" else False
        # Load model, return True if successful and set model to evaluation mode
        if not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                status["rl_mode"] = "eval"
                return f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to evaluation mode\n"
            else:
                self.algorithm.set_train()
                status["rl_mode"] = "train"
                return "No checkpoint found to evaluate model, start training from scratch\n"
        elif status["resume"]:
            if self.algorithm.load():
                self.algorithm.set_train()
                status["rl_mode"] = "train"
                return f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to training mode\n"
            else:
                self.algorithm.set_train()
                status["rl_mode"] = "train"
                return "No checkpoint found to resume training, start training from scratch\n"
        else:
            self.algorithm.set_train()
            status["rl_mode"] = "train"
            return "Training from scratch\n"

    def set_paths(self, model_path, model_name):
        self.algorithm.set_paths(model_path, model_name)

    def add_memory_entry(self, obs, actions, action_logprobs, rewards, terminals, next_obs=None, intrinsic_rewards=None):
        self.memory.add(obs, actions, action_logprobs, rewards, terminals, next_obs=next_obs, intrinsic_reward=intrinsic_rewards)

    def update_status(self, status):
        self.set_paths(status["model_path"], status["model_name"])
        status["obs_collected"] = len(self.memory) + 1
        if self.algorithm_name == 'PPO':
            status['min_update'] = self.algorithm.horizon
        else:
            status["min_update"] = self.algorithm.min_memory_size
        if status["rl_mode"] != self.mode:
            self.mode = status["rl_mode"]
            if self.mode == "train":
                self.algorithm.set_train()
            else:
                self.algorithm.set_eval()

    def restruct_current_obs(self, observations):
        self.current_obs = self.restructure_data(observations)

    def train_loop(self, status, engine):
        actions, action_logprobs = self.act(self.current_obs)
        next_obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(status, engine, actions)
        intrinsic_reward = None
        if self.use_intrinsic_reward:
            intrinsic_reward = self.agent_type.get_intrinsic_reward(self.current_obs)
            # The environment has not been reset so we can send the intrinsic
            # reward to the model (only for displaying purposes)
            if not any(all_terminals):
                status["intrinsic_reward"] = intrinsic_reward.detach().cpu().numpy().tolist()
                engine.SendRLStatusToModel(status)
                engine.UpdateReward()
        n_obs = None
        if self.use_next_obs:
            n_obs = next_obs
        # Memory Adding
        self.add_memory_entry(self.current_obs, actions, action_logprobs,
                              rewards, all_terminals, next_obs=n_obs, intrinsic_rewards=intrinsic_reward)
        # Logging
        status["objective"], status["best_objective"] = self.update_logging(terminal_result)
        # Training
        if self.should_train():
            self.update(status, mini_batch_size=self.algorithm.batch_size, next_obs=next_obs)
            if self.use_intrinsic_reward and self.algorithm == 'PPO':
                self.agent_type.update_rnd_model(self.memory, self.algorithm.horizon, self.algorithm.batch_size)
            # Clear memory
            if self.algorithm.clear_memory:
                self.memory.clear_memory()
        self.current_obs = next_obs
        self.env_reset = terminal_result["EnvReset"]
        self.handle_env_reset(status)

    def eval_loop(self, status, engine, evaluate=False):
        actions = self.act_certain(self.current_obs)
        obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(status, engine, actions)
        if evaluate: self.evaluate(status, rewards, all_terminals, terminal_result, percent_burned)
        self.current_obs = obs
        return terminal_result["AllAgentsSucceeded"], terminal_result["EnvReset"] # True if all agents reached their goal can do "OneAgentSucceeded"

    def step_agent(self, status, engine, actions):
        self.agent_actions = self.get_action(actions)
        observations, rewards, all_terminals, terminal_result, percent_burned = engine.Step(self.agent_type.name, self.agent_actions)
        obs = self.restructure_data(observations)
        self.env_step += 1
        return obs, rewards, all_terminals, terminal_result, percent_burned

    def sim_step(self, engine):
        engine.SimStep(self.agent_actions)

    def handle_env_reset(self, status):
        if self.env_reset:
            status["current_episode"] += 1

    def update(self, status, mini_batch_size, next_obs):
        status["console"] += self.algorithm.update(self.memory, mini_batch_size, next_obs, self.logger)
        status["train_step"] += 1
        if self.algorithm_name == 'PPO':
            status["policy_updates"] += self.algorithm.horizon // self.algorithm.batch_size * self.algorithm.k_epochs
        elif self.algorithm_name == 'IQL':
            status["policy_updates"] += self.algorithm.k_epochs
        elif self.algorithm_name == 'TD3':
            status["policy_updates"] += self.algorithm.k_epochs
        if status["train_step"] >= status["max_train"] and status["auto_train"]:
            status["train_episode"] += 1
            status["console"] += "Training finished, after {} training steps, now starting Evaluation\n".format(status["train_step"])
            print("Starting evaluation")
            status["rl_mode"] = "eval"
        self.logger.summarize_metrics(status)

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs

    def get_action(self, actions):
        return self.agent_type.get_action(actions)

    def update_logging(self, terminal_result):
        self.logger.log_episode(terminal_result)
        self.logger.calc_objective_percentage()
        return self.logger.current_objective, self.logger.get_best_objective()[1]

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

    def evaluate(self, status, rewards, terminals, terminal_result, percent_burned):
        episode_over, log = self.get_evaluation_string(status, rewards, terminals, terminal_result, percent_burned)
        status["console"] += log
        if episode_over and status["auto_train"]:
            self.eval_steps += 1
            if self.eval_steps >= status["max_eval"]:
                status["console"] += "Evaluation finished, after {} evaluation steps with average reward: {}\n".format(self.eval_steps, np.mean(self.stats["reward"]))
                status["rl_mode"] = "train"
                self.reset(status)

    def get_evaluation_string(self, status, rewards, terminals, terminal_result, percent_burned):
        # Log stats
        self.stats['reward'][-1] += rewards[0]
        self.stats['time'][-1] += 1
        console = ""

        if terminal_result['EnvReset']:
            console += "Episode: {} finished with terminal state\n".format(self.stats['episode'][-1] + 1)
            if terminal_result['OneAgentDied']:  # Drone died
                console += "One of the Drones died.\n\n"
                self.stats['died'][-1] += 1
            else:  # Drone reached goal
                console += "Drones extinguished all fires.\n\n"
                self.stats['reached'][-1] += 1

            self.stats['perc_burn'][-1] = percent_burned
            # Print stats
            console += "-----------------------------------\n"
            console += "Episode: {}\n".format(self.stats['episode'][-1] + 1)
            console += "Reward: {}\n".format(self.stats['reward'][-1])
            console += "Time: {}\n".format(self.stats['time'][-1])
            console += "Percentage burned: {}\n".format(self.stats['perc_burn'][-1])
            console += "Died: {}\n".format(self.stats['died'][-1])
            console += "Reached: {}\n".format(self.stats['reached'][-1])
            console += "Total average time: {}\n".format(np.mean(self.stats['time']))
            console += "Total average percentage burned: {}\n".format(np.mean(self.stats['perc_burn']))
            console += "Total average reward: {}\n".format(np.mean(self.stats['reward']))
            console += "Total died: {}\n".format(sum(self.stats['died']))
            console += "Total reached: {}\n".format(sum(self.stats['reached']))
            console += "-----------------------------------\n\n\n"

            # Reset stats if this is not the last evaluation
            if self.stats['episode'][-1] != status["max_eval"] - 1:
                self.stats['episode'].append(self.stats['episode'][-1] + 1)
                self.stats['died'].append(0)
                self.stats['perc_burn'].append(0)
                self.stats['reached'].append(0)
                self.stats['time'].append(0)
                self.stats['reward'].append(0)

            return True, console
        return False, console

    def restructure_data(self, observations_):
        return self.agent_type.restructure_data(observations_)




