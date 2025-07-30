from algorithms.ppo import PPO
from algorithms.iql import IQL
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.td3 import TD3
from algorithms.rl_config import RLConfig, PPOConfig, IQLConfig, TD3Config, NoAlgorithmConfig
from utils import SimulationBridge, Logger, get_project_paths
import numpy as np
import os
import yaml
from memory import SwarmMemory
from explore_agent import ExploreAgent
from flying_agent import FlyAgent
from planner_agent import PlannerAgent


class AgentHandler:
    def __init__(self, config: dict, sim_bridge: SimulationBridge, agent_type: str = None, subtype: str = None, mode: str = None, is_sub_agent: bool = True):

        # Past Agent Actions
        self.agent_actions = None
        self.sim_bridge = sim_bridge
        # A sub agent is an agent that is part of a lower hierarchy level,
        # e.g. a PlannerFlyAgent is a sub agent of ExploreFlyAgent or PlannerFlyAgent
        self.is_sub_agent = is_sub_agent

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

        # Auto Training Parameters
        auto_train_dict = config["settings"]["auto_train"]
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.max_train = auto_train_dict["max_train"]
        self.max_eval = auto_train_dict["max_eval"]
        self.train_episodes = auto_train_dict["train_episodes"]

        # TODO : Check this
        # Used for Evaluation
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        auto_train_dict = config["settings"]["auto_train"]
        self.use_auto_train = auto_train_dict["use_auto_train"]
        self.max_eval = auto_train_dict["max_eval"]
        self.max_train = auto_train_dict["max_train"]

        # Agent Type is either fly_agent, explore_agent or planner_agent
        drone_count = agent_dict["num_agents"]
        self.agent_type = self.get_agent_from_type(agent_type=agent_type if subtype is None else subtype, num_drones=drone_count)
        self.num_agents = self.agent_type.get_num_agents(agent_dict["num_agents"])

        # On which Level of the Hierarchy is the agent
        self.hierarchy_level = self.agent_type.get_hierarchy_level()

        # Use Intrinsic Reward according to: https://arxiv.org/abs/1810.12894
        self.use_intrinsic_reward = self.agent_type.use_intrinsic_reward

        # Other variables (need resetting)
        self.current_obs = None
        self.eval_steps = 0
        self.env_step = 0
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False

        # TODO: Maybe put this into the Agent Type?
        if self.use_intrinsic_reward:
            self.agent_type.initialize_rnd_model(vision_range, drone_count=self.num_agents,
                                                 map_size=map_size, time_steps=time_steps)

        root_path = get_project_paths("root_path")
        loading_path = os.path.join(root_path, agent_dict["default_model_folder"])
        loading_name = agent_dict["default_model_name"]
        model_path = os.path.join(root_path, config["paths"]["model_directory"]) if not self.resume else loading_path
        model_name = self.get_model_name_from_config(config)# if not self.resume else loading_name

        # TODO: If I resume the training, I need to load several things:
        # - The model
        # - The optimizer state
        # - The memory state (possibly, for OffPolicy algorithms)
        # - The logger state

        rl_config = RLConfig(algorithm=algorithm,
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
            rl_config = PPOConfig(**vars(rl_config),
                                  use_categorical=agent_type == "planner_agent",
                                  use_variable_state_masks=agent_type == "planner_agent")
            rl_config.use_next_obs = False

            self.algorithm = PPO(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'IQL':
            # IQLConfig is used for IQL algorithm
            rl_config = IQLConfig(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False  # IQL does not clear memory

            self.algorithm = IQL(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'TD3':
            # TD3Config is used for TD3 algorithm
            rl_config = TD3Config(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False

            self.algorithm = TD3(network=self.agent_type.get_network(algorithm=algorithm),
                                 config=rl_config)
        elif algorithm == 'no_algo':
            # NoAlgorithmConfig is used for No Algorithm scenario
            rl_config = NoAlgorithmConfig(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim

            self.algorithm = RLAlgorithm(rl_config)

        self.use_next_obs = self.algorithm.use_next_obs

        self.memory = SwarmMemory(max_size=self.algorithm.memory_size,
                                  num_agents=self.num_agents,
                                  action_dim=self.agent_type.action_dim,
                                  use_intrinsic_reward=self.use_intrinsic_reward,
                                  use_next_obs=self.use_next_obs)

        # Update status dict only if not a sub_agent
        if not self.is_sub_agent:
            model_path = self.algorithm.get_model_path()
            logging_path = os.path.join(model_path, "logs")
            self.logger = Logger(log_dir=logging_path, resume=self.resume)


            if not self.resume:
                root_model_path = self.algorithm.model_path
                # Save own Config.Yaml when this is a fresh start
                config_path = os.path.join(root_model_path, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, indent=4)


            # TODO This should ideally be done at the end of Evaluation
            #self.logger.log_hparams(rl_config.__dict__)

            creation_string += f"Agents of Type: {self.agent_type.name} initialized\n"
            if self.num_agents != drone_count:
                creation_string += f"{self.num_agents} Agent controls {drone_count} Drones\n"
            else:
                creation_string += f"{self.num_agents} Agents in total who control {self.num_agents} Drones\n"
            creation_string += "Algorithm: {}\n".format(self.algorithm_name)
            sim_bridge.append_console(creation_string)
            sim_bridge.set("model_path", self.algorithm.model_path)
            sim_bridge.set("model_name", self.algorithm.model_name)

            min_update = 0
            if self.algorithm_name == 'PPO':
                min_update = self.algorithm.horizon
            elif self.algorithm_name == 'IQL' or self.algorithm_name == 'TD3':
                min_update = self.algorithm.min_memory_size

            self.sim_bridge.set("min_update", min_update)

    @staticmethod
    def get_model_name_from_config(config):
        """
        Get the model name from the configuration.
        :param config: The configuration dictionary.
        :return: The model name.
        """
        if config["paths"]["model_name"] != "" and config["paths"]["model_name"] is not None:
            return config["paths"]["model_name"]
        else:
            algorithm = config["algorithm"]["type"].lower()
            agent_type = config["settings"]["hierarchy_type"].lower()
            return algorithm + "_" + agent_type + ".pt"

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

    def reset(self):
        """
        Resets the agent and its algorithm to the initial state.
        """

        train_episode = self.sim_bridge.get("train_episode")

        if self.use_auto_train:
            # Checks if the auto_training is finished
            if train_episode == self.train_episodes:
                self.sim_bridge.set("agent_online", False)
                log = "Training finished, after {} training episodes\n".format(train_episode)
                self.sim_bridge.append_console(log)
                self.sim_bridge.append_console("Agent is offline\n")
                return
            else:
                log = "Resume with next training step {}/{}\n".format(train_episode + 1, self.train_episodes)
                self.sim_bridge.append_console(log)
                self.sim_bridge.set("train_step", 0)
        self.algorithm.reset()
        self.reset_agent_variables()

    def reset_agent_variables(self):
        """
        Reset the agent variables to their initial state.
        This method is called when the environment is reset.
        """
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        self.hierarchy_steps = 0
        self.env_step = 0
        self.current_obs = None
        self.agent_actions = None
        self.env_reset = True
        # self.sim_bridge.set("env_reset", True)
        self.eval_steps = 0

        # Reset the Logger
        if not self.is_sub_agent: # should not really be False like ever, but just in case
            self.logger.close()
            model_path = self.algorithm.get_model_path()
            logging_path = os.path.join(model_path, "logs")
            self.logger = Logger(log_dir=logging_path)

        # Reset memory
        self.memory.clear_memory()

    def should_train(self):
        if self.algorithm_name == 'PPO':
            return len(self.memory) >= self.algorithm.horizon
        elif self.algorithm_name == 'IQL':
            return len(self.memory) >= self.algorithm.min_memory_size and self.env_step % self.algorithm.policy_freq == 0
        elif self.algorithm_name == 'TD3':
            return len(self.memory) >= self.algorithm.min_memory_size
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(self.algorithm_name))

    def load_model(self, change_status=False):
        # Load model if possible, return new rl_mode and possible console string
        log = ""
        if self.algorithm_name == 'no_algo':
            log += "No algorithm used, no model to load\n"
        train = True if self.rl_mode == "train" else False
        if not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                log += f"Load model from checkpoint:\n {os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}\n" \
                       f"Model set to evaluation mode\n"
            else:
                self.algorithm.set_train()
                self.rl_mode = "train"
                log +=  "No checkpoint found to evaluate model, start training from scratch\n"
        elif self.resume:
            if self.algorithm.load():
                self.algorithm.set_train()
                log += f"Load model from checkpoint:\n{os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}\n" \
                       f"Resume Training from checkpoint\n"
            else:
                self.algorithm.set_train()
                log += "No checkpoint found to resume training, start training from scratch\n"
        else:
            self.algorithm.set_train()
            log += "Training from scratch\n"

        if change_status:
            self.sim_bridge.set("rl_mode", self.rl_mode)
            self.sim_bridge.append_console(log)

    def add_memory_entry(self, obs, actions, action_logprobs, rewards, terminals, next_obs=None, intrinsic_rewards=None):
        self.memory.add(obs, actions, action_logprobs, rewards, terminals, next_obs=next_obs, intrinsic_reward=intrinsic_rewards)

    def update_status(self):
        model_path = self.sim_bridge.get("model_path")
        model_name = self.sim_bridge.get("model_name")
        self.algorithm.set_paths(model_path, model_name)

        self.sim_bridge.set("obs_collected", len(self.memory) + 1)

        if self.sim_bridge.get("rl_mode") != self.rl_mode:
            self.rl_mode = self.sim_bridge.get("rl_mode")
            self.algorithm.set_train() if self.rl_mode == "train" else self.algorithm.set_eval()

    def restruct_current_obs(self, observations):
        self.current_obs = self.restructure_data(observations)

    def train_loop(self, engine):
        actions, action_logprobs = self.act(self.current_obs)
        next_obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(engine, actions)
        intrinsic_reward = None
        if self.use_intrinsic_reward:
            intrinsic_reward = self.agent_type.get_intrinsic_reward(self.current_obs)
            # The environment has not been reset so we can send the intrinsic
            # reward to the model (only for displaying purposes)
            if not any(all_terminals):
                intrinsic_reward = intrinsic_reward.detach().cpu().numpy().tolist()
                self.sim_bridge.set("intrinsic_reward", intrinsic_reward)
                engine.SendRLStatusToModel(self.sim_bridge.status)
                engine.UpdateReward()
        n_obs = None
        if self.use_next_obs:
            n_obs = next_obs
        # Memory Adding
        self.add_memory_entry(self.current_obs, actions, action_logprobs,
                              rewards, all_terminals, next_obs=n_obs, intrinsic_rewards=intrinsic_reward)

        # Update the Logger before checking if we should train, so that the logger has the latest information
        # to calculate the objective percentage and best reward
        self.update_logging(terminal_result)

        # Training
        if self.should_train():
            self.update(mini_batch_size=self.algorithm.batch_size, next_obs=next_obs)
            if self.use_intrinsic_reward and self.algorithm == 'PPO':
                self.agent_type.update_rnd_model(self.memory, self.algorithm.horizon, self.algorithm.batch_size)
            # Clear memory
            if self.algorithm.clear_memory:
                self.memory.clear_memory()
        self.current_obs = next_obs
        self.env_reset = terminal_result["EnvReset"]
        self.handle_env_reset()

    def eval_loop(self, engine, evaluate=False):
        actions = self.act_certain(self.current_obs)
        obs, rewards, all_terminals, terminal_result, percent_burned = self.step_agent(engine, actions)
        self.current_obs = obs
        if evaluate: self.evaluate(rewards, all_terminals, terminal_result, percent_burned)
        return terminal_result["AllAgentsSucceeded"], terminal_result["EnvReset"] # True if all agents reached their goal can do "OneAgentSucceeded"

    def step_agent(self, engine, actions):
        self.agent_actions = self.get_action(actions)
        observations, rewards, all_terminals, terminal_result, percent_burned = engine.Step(self.agent_type.name, self.agent_actions)
        obs = self.restructure_data(observations)
        self.env_step += 1
        return obs, rewards, all_terminals, terminal_result, percent_burned

    def step_without_network(self, engine):
        agent_actions = self.get_action([[0,0] for _ in range(self.num_agents)])
        _, _, _, terminal_result, _ = engine.Step(self.agent_type.name, agent_actions)
        self.env_reset = terminal_result["EnvReset"]
        self.handle_env_reset()

    # Possibly unused
    def sim_step(self, engine):
        engine.SimStep(self.agent_actions)

    def handle_env_reset(self):
        if self.env_reset:
            self.sim_bridge.set("current_episode", self.sim_bridge.get("current_episode") + 1)

    def update(self, mini_batch_size, next_obs):
        log = self.algorithm.update(self.memory, mini_batch_size, next_obs, self.logger)
        self.sim_bridge.add_value("train_step", 1)

        if self.algorithm_name == 'PPO':
            policy_updates = self.algorithm.horizon // self.algorithm.batch_size * self.algorithm.k_epochs
            self.sim_bridge.add_value("policy_updates", policy_updates)
        elif self.algorithm_name == 'IQL':
            self.sim_bridge.add_value("policy_updates", self.algorithm.k_epochs)
        elif self.algorithm_name == 'TD3':
            self.sim_bridge.add_value("policy_updates", self.algorithm.k_epochs)

        if self.sim_bridge.get("train_step") >= self.max_train and self.use_auto_train:
            self.sim_bridge.add_value("train_episode", 1)
            log += "Training finished, after {} training steps, now starting Evaluation\n".format(self.sim_bridge.get("train_step"))
            self.sim_bridge.set("rl_mode", "eval")

        self.sim_bridge.append_console(log)
        self.logger.summarize()

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs.ravel()

    def get_action(self, actions):
        return self.agent_type.get_action(actions)

    def update_logging(self, terminal_result):
        self.logger.log_step(terminal_result)
        self.logger.calc_current_objective()

        self.sim_bridge.set("objective", self.logger.best_metrics["current_objective"])
        self.sim_bridge.set("best_objective", self.logger.get_best_objective()[1])

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

    def evaluate(self, rewards, terminals, terminal_result, percent_burned):
        episode_over = self.get_evaluation_string(rewards, terminals, terminal_result, percent_burned)

        if episode_over and self.use_auto_train:
            self.eval_steps += 1
            if self.eval_steps >= self.max_eval:
                log = "Evaluation finished, after {} evaluation steps with average reward: {}\n".format(self.eval_steps, np.mean(self.stats["reward"]))
                self.sim_bridge.append_console(log)
                self.sim_bridge.set("rl_mode", "train")
                self.reset()

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

            self.sim_bridge.append_console(console)

            return True
        return False

    def restructure_data(self, observations_):
        return self.agent_type.restructure_data(observations_)




