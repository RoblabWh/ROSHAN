from algorithms.ppo import PPO
from algorithms.iql import IQL
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.td3 import TD3
from algorithms.rl_config import RLConfig, PPOConfig, IQLConfig, TD3Config, NoAlgorithmConfig
from utils import Logger, get_project_paths
import numpy as np
import os
from memory import SwarmMemory
from explore_agent import ExploreAgent
from flying_agent import FlyAgent
from planner_agent import PlannerAgent


class AgentHandler:
    def __init__(self, config: dict, agent_type: str = None, subtype: str = None, mode: str = None, status: dict = None, logdir=None):

        # Past Agent Actions
        self.agent_actions = None

        # If agent_type is None it is the FIRST selected agent type(this determines the hierarchy)
        agent_dict = config["environment"]["agent"][agent_type]

        vision_range = agent_dict["view_range"]
        time_steps = agent_dict["time_steps"]
        algorithm = agent_dict["algorithm"]
        map_size = config["fire_model"]["simulation"]["grid"]["exploration_map_size"]

        # Probably can be discarded, but kept for compatibility now
        self.original_algo = algorithm

        creation_string = ""

        if agent_type == "explore_agent":
            algorithm = 'no_algo'
            creation_string += "Using explore_agent, no algorithm needed\n"
            creation_string += "Training of explore_agents must be implemented first\n"
        supported_algos = ['no_algo', 'PPO', 'IQL', 'TD3']
        assert algorithm in supported_algos, f"Algorithm {algorithm} not supported, only {supported_algos} are supported"

        self.algorithm_name = algorithm
        self.env_reset = False
        self.rl_mode = mode
        self.resume = config["settings"]["resume"]

        # TODO : Check this

        # Used for Evaluation
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        auto_train_dict = config["settings"]["auto_train"]
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.max_eval = auto_train_dict["max_eval"]
        if logdir is not None:
            self.logger = Logger(log_dir=logdir)
        self.eval_steps = 0

        # Agent Type is either fly_agent, explore_agent or planner_agent
        drone_count = agent_dict["num_agents"]
        self.agent_type = self.get_agent_from_type(agent_type=agent_type if subtype is None else subtype, num_drones=drone_count)
        self.num_agents = self.agent_type.get_num_agents(agent_dict["num_agents"])

        # On which Level of the Hierarchy is the agent
        self.hierarchy_level = self.agent_type.get_hierarchy_level()
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False

        # Use Intrinsic Reward according to: https://arxiv.org/abs/1810.12894
        self.use_intrinsic_reward = self.agent_type.use_intrinsic_reward

        # Other variables
        self.env_step = 0

        if self.use_intrinsic_reward:
            self.agent_type.initialize_rnd_model(vision_range, drone_count=self.num_agents,
                                                 map_size=map_size, time_steps=time_steps)
        self.current_obs = None

        root_path = get_project_paths("root_path")
        loading_path = agent_dict["default_model_folder"]
        loading_name = agent_dict["default_model_name"]
        model_path = os.path.join(root_path, config["paths"]["model_directory"]) if not config["settings"]["resume"] else loading_path
        model_name = self.get_model_name_from_config(config) if not config["settings"]["resume"] else loading_name

        base_config = RLConfig(algorithm=algorithm,
                                 use_auto_train=self.use_auto_train,
                                 model_path=model_path,
                                 model_name=model_name,
                                 loading_path=loading_path,
                                 loading_name=loading_name,
                                 vision_range=vision_range,
                                 drone_count=drone_count,
                                 map_size=map_size,
                                 time_steps=time_steps)

        if algorithm == 'PPO':
            # PPOConfig is used for PPO algorithm
            rl_config = PPOConfig(**vars(base_config),
                                  use_categorical=agent_type == "planner_agent",
                                  use_variable_state_masks=agent_type == "planner_agent")
            rl_config.use_next_obs = False

            self.algorithm = PPO(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'IQL':
            # IQLConfig is used for IQL algorithm
            rl_config = IQLConfig(**vars(base_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False  # IQL does not clear memory

            self.algorithm = IQL(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'TD3':
            # TD3Config is used for TD3 algorithm
            rl_config = TD3Config(**vars(base_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False

            self.algorithm = TD3(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'no_algo':
            # NoAlgorithmConfig is used for No Algorithm scenario
            rl_config = NoAlgorithmConfig(**vars(base_config))
            rl_config.action_dim = self.agent_type.action_dim

            self.algorithm = RLAlgorithm(rl_config)

        self.use_next_obs = self.algorithm.use_next_obs

        self.memory = SwarmMemory(max_size=self.algorithm.memory_size,
                                  num_agents=self.num_agents,
                                  action_dim=self.agent_type.action_dim,
                                  use_intrinsic_reward=self.use_intrinsic_reward,
                                  use_next_obs=self.use_next_obs)

        # Update status dict now
        if status is not None:
            creation_string += f"Agents of Type: {self.agent_type.name} initialized\n"
            if self.num_agents != drone_count:
                creation_string += f"{self.num_agents} Agent controls {drone_count} Drones\n"
            else:
                creation_string += f"{self.num_agents} Agents in total who control {self.num_agents} Drones\n"
            creation_string += "Algorithm: {}\n".format(self.algorithm_name)
            status["console"] += creation_string
            status["model_path"] = self.algorithm.model_path
            status["model_name"] = self.algorithm.model_name

    @staticmethod
    def get_model_name_from_config(config):
        """
        Get the model name from the configuration.
        :param config: The configuration dictionary.
        :return: The model name.
        """
        if "model_name" in config["settings"]:
            return config["settings"]["model_name"]
        else:
            return "model.pt"

    @staticmethod
    def get_agent_from_type(agent_type, num_drones):
        allowed_types = ["fly_agent", "explore_agent", "ExploreFlyAgent", "PlannerFlyAgent", "planner_agent"]
        if agent_type not in allowed_types:
            raise ValueError("Invalid agent type, must be either {}, was: {}".format(allowed_types, agent_type))
        if agent_type == "fly_agent":
            return FlyAgent("fly_agent")
        if agent_type == "ExploreFlyAgent":
            return FlyAgent("ExploreFlyAgent")
        if agent_type == "PlannerFlyAgent":
            return FlyAgent("PlannerFlyAgent")
        if agent_type == "explore_agent":
            return ExploreAgent()
        if agent_type == "planner_agent":
            return PlannerAgent(num_drones)

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

    def load_model(self):
        # Load model if possible, return new rl_mode and possible console string
        if self.algorithm_name == 'no_algo':
            return self.rl_mode, "No algorithm used, no model to load\n"
        train = True if self.rl_mode == "train" else False
        if not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                return self.rl_mode, f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to evaluation mode\n"
            else:
                self.algorithm.set_train()
                self.rl_mode = "train"
                return self.rl_mode, "No checkpoint found to evaluate model, start training from scratch\n"
        elif self.resume:
            if self.algorithm.load():
                self.algorithm.set_train()
                return self.rl_mode, f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to training mode\n"
            else:
                self.algorithm.set_train()
                return self.rl_mode, "No checkpoint found to resume training, start training from scratch\n"
        else:
            self.algorithm.set_train()
            return self.rl_mode, "Training from scratch\n"

    def add_memory_entry(self, obs, actions, action_logprobs, rewards, terminals, next_obs=None, intrinsic_rewards=None):
        self.memory.add(obs, actions, action_logprobs, rewards, terminals, next_obs=next_obs, intrinsic_reward=intrinsic_rewards)

    def update_status(self, status):
        self.algorithm.set_paths(status["model_path"], status["model_name"])
        status["obs_collected"] = len(self.memory) + 1
        if self.algorithm_name == 'PPO':
            status['min_update'] = self.algorithm.horizon
        elif self.algorithm_name == 'IQL' or self.algorithm_name == 'TD3':
            status["min_update"] = self.algorithm.min_memory_size
        elif self.algorithm_name == 'no_algo':
            status["min_update"] = 0
        if status["rl_mode"] != self.rl_mode:
            self.rl_mode = status["rl_mode"]
            if self.rl_mode == "train":
                self.algorithm.set_train()
            else:
                self.algorithm.set_eval()

    def restruct_current_obs(self, observations):
        self.current_obs = self.restructure_data(observations)

    def train_loop(self, status, engine):
        actions, action_logprobs = self.act(self.current_obs)
        next_obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(engine, actions)
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
        obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(engine, actions)
        if evaluate: self.evaluate(status, rewards, all_terminals, terminal_result, percent_burned)
        self.current_obs = obs
        return terminal_result["AllAgentsSucceeded"], terminal_result["EnvReset"] # True if all agents reached their goal can do "OneAgentSucceeded"

    def step_agent(self, engine, actions):
        self.agent_actions = self.get_action(actions)
        observations, rewards, all_terminals, terminal_result, percent_burned = engine.Step(self.agent_type.name, self.agent_actions)
        obs = self.restructure_data(observations)
        self.env_step += 1
        return obs, rewards, all_terminals, terminal_result, percent_burned

    def step_without_network(self, status, engine):
        agent_actions = self.get_action([[0,0] for _ in range(self.num_agents)])
        _, _, _, terminal_result, _ = engine.Step(self.agent_type.name, agent_actions)
        self.env_reset = terminal_result["EnvReset"]
        self.handle_env_reset(status)

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
        if status["train_step"] >= status["max_train"] and self.use_auto_train:
            status["train_episode"] += 1
            status["console"] += "Training finished, after {} training steps, now starting Evaluation\n".format(status["train_step"])
            status["rl_mode"] = "eval"
        self.logger.summarize_metrics(status)

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs.ravel()

    def get_action(self, actions):
        return self.agent_type.get_action(actions)

    def update_logging(self, terminal_result):
        self.logger.log_episode(terminal_result)
        self.logger.calc_objective_percentage()
        return self.logger.current_objective, self.logger.get_best_objective()[1]

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

    def evaluate(self, status, rewards, terminals, terminal_result, percent_burned):
        episode_over, log = self.get_evaluation_string(rewards, terminals, terminal_result, percent_burned)
        status["console"] += log
        if episode_over and self.use_auto_train:
            self.eval_steps += 1
            if self.eval_steps >= self.max_eval:
                status["console"] += "Evaluation finished, after {} evaluation steps with average reward: {}\n".format(self.eval_steps, np.mean(self.stats["reward"]))
                status["rl_mode"] = "train"
                self.reset(status)

    def get_evaluation_string(self, rewards, terminals, terminal_result, percent_burned):
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
            if self.stats['episode'][-1] != self.max_eval - 1:
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




