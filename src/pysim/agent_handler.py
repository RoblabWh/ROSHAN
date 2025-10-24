from algorithms.ppo import PPO
from algorithms.iql import IQL
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.td3 import TD3
from algorithms.rl_config import RLConfig, PPOConfig, IQLConfig, TD3Config, NoAlgorithmConfig, override_from_dict
from utils import SimulationBridge, get_project_paths, remove_suffix
import os, logging, yaml, shutil, math, importlib.util
from memory import SwarmMemory
from evaluation import Evaluator, TensorboardLogger
from explore_agent import ExploreAgent
from flying_agent import FlyAgent
from planner_agent import PlannerAgent


class AgentHandler:
    def __init__(self, config: dict, sim_bridge: SimulationBridge, agent_type: str = None, subtype: str = None, mode: str = None, is_sub_agent: bool = True):

        self.sim_bridge = sim_bridge
        self.no_gui = config["settings"]["mode"] == 2
        # A sub agent is an agent that is part of a lower hierarchy level,
        # e.g. a PlannerFlyAgent is a sub agent of ExploreFlyAgent or PlannerFlyAgent
        self.is_sub_agent = is_sub_agent
        self.agent_type_str = agent_type

        # If agent_type is None it is the FIRST selected agent type(this determines the hierarchy)
        agent_dict = config["environment"]["agent"][agent_type]

        vision_range = agent_dict["view_range"]
        time_steps = agent_dict["time_steps"]
        algorithm = agent_dict["algorithm"]
        map_size = config["fire_model"]["simulation"]["grid"]["exploration_map_size"]

        # Frame skips for the agent, e.g. if frame_skips=2, the agent repeats the same action for 2 environment steps
        self.frame_skips = agent_dict["frame_skips"]
        self.ctrl_ctr = 0
        self.cached_actions = None
        self.cached_logprobs = None

        # Probably can be discarded, but kept for compatibility now
        self.original_algo = algorithm

        supported_algos = ['no_algo', 'PPO', 'IQL', 'TD3']
        assert algorithm in supported_algos, f"Algorithm {algorithm} not supported, only {supported_algos} are supported"

        self.algorithm_name = algorithm
        self.env_reset = False
        self.rl_mode = mode
        self.resume = config["settings"]["resume"]

        # Auto Training Parameters
        at_dict = config["settings"]["auto_train"]
        use_auto_train = at_dict["use_auto_train"]

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
        self.cached_logprobs = None
        self.cached_actions = None
        self.env_step = 0
        self.hierarchy_steps = 0
        self.hierarchy_early_stop = False

        # Save or Load Replay Buffer
        self.save_replay_buffer = config["settings"]["save_replay_buffer"]
        self.save_size = config["settings"]["save_size"]

        # TODO: Maybe put this into the Agent Type?
        if self.use_intrinsic_reward:
            self.agent_type.initialize_rnd_model(vision_range, drone_count=self.num_agents,
                                                 map_size=map_size, time_steps=time_steps)

        root_path = get_project_paths("root_path")
        loading_path, loading_name, model_path, model_name = None, None, None, None
        if self.is_sub_agent:
            # If this is a sub agent, we load the model from the default agent's model path
            loading_path = os.path.join(root_path, agent_dict["default_model_folder"])
            loading_name, later_logs = self.get_model_name(path=str(loading_path),
                                               model_string=agent_dict["default_model_name"],
                                               agent_type=agent_type,
                                               is_loading_name=True)
            model_path = loading_path
            # model_name should not be used for sub agents
        else:
            # If this is the main agent, we use the model path from the config
            is_loading = self.resume or self.rl_mode == "eval"
            model_path = os.path.join(root_path, config["paths"]["model_directory"])
            model_string, later_logs = self.get_model_name(path=str(model_path),
                                             model_string=config["paths"]["model_name"],
                                             agent_type=agent_type,
                                             is_loading_name=is_loading)
            model_name = model_string if not is_loading else remove_suffix(model_string)
            loading_path = model_path if is_loading else None
            loading_name = model_string if is_loading else None
            # If this is the main agent and a fresh training, we need to create the model path
            if not is_loading and not os.path.exists(model_path):
                os.makedirs(model_path)

        self.root_model_path = model_path

        # Only create a FileHandler if this is not a sub agent
        logging_file = ""
        if not self.is_sub_agent:
            # Remove any exiting FileHandlers
            logger = logging.getLogger()
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)
                    handler.close()
            # Check if auto_train is used, if so, append "training_1" to model path (dirty, but works for now)
            mp = model_path.__str__() if not use_auto_train else os.path.join(model_path.__str__(), "training_1")
            logging_dir = os.path.join(mp, "logs")
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir, exist_ok=True)
            logging_file = os.path.join(logging_dir, "logging.log")
            file_handler = logging.FileHandler(logging_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

        # If this is not a sub agent, we can now log everything
        self.logger = logging.getLogger(agent_type)
        if not self.is_sub_agent:
            self.logger.info(f"Logging File at {logging_file}")
        if later_logs["msg"] is not None:
            self.logger.log(later_logs["level"], later_logs["msg"])

        # # If I resume the training, I need to load several things:
        # - The model
        # - The optimizer state
        # - The memory state (possibly, for OffPolicy algorithms)
        # - The logger state

        rl_config = RLConfig(algorithm=algorithm,
                                 use_auto_train=use_auto_train,
                                 model_path=str(model_path),
                                 model_name=model_name,
                                 loading_path=str(loading_path),
                                 loading_name=loading_name,
                                 vision_range=vision_range,
                                 drone_count=drone_count,
                                 map_size=map_size,
                                 time_steps=time_steps,
                                 share_encoder=config["algorithm"]["share_encoder"],
                                 use_tanh_dist=config["algorithm"]["use_tanh_dist"])

        algo_overrides = config["algorithm"].get(self.algorithm_name, {})

        # Load the network architecture from the previous run if available
        if self.algorithm_name != 'no_algo':
            try:
                network_classes = self._load_network_arch()
            except FileNotFoundError as e:
                self.logger.warning(f"{e}. Falling back to default network classes")
                network_classes = self.agent_type.get_network(algorithm=self.algorithm_name)
                if not isinstance(network_classes, (list, tuple)):
                    network_classes = (network_classes,)
        else:
            network_classes = None

        if algorithm == 'PPO':
            # PPOConfig is used for PPO algorithm
            rl_config = PPOConfig(**vars(rl_config),
                                  use_categorical= agent_type == "planner_agent",
                                  use_variable_state_masks= agent_type == "planner_agent")
            rl_config.use_next_obs = False if not self.save_replay_buffer else True
            rl_config = override_from_dict(rl_config, algo_overrides)

            self.algorithm = PPO(network=network_classes,
                                 config=rl_config)
        elif algorithm == 'IQL':
            # IQLConfig is used for IQL algorithm
            rl_config = IQLConfig(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False  # IQL does not clear memory
            rl_config = override_from_dict(rl_config, algo_overrides)

            self.algorithm = IQL(network=network_classes,
                                 config=rl_config)
        elif algorithm == 'TD3':
            # TD3Config is used for TD3 algorithm
            rl_config = TD3Config(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim
            rl_config.clear_memory = False
            rl_config = override_from_dict(rl_config, algo_overrides)

            self.algorithm = TD3(network=network_classes,
                                 config=rl_config)
        elif algorithm == 'no_algo':
            # NoAlgorithmConfig is used for No Algorithm scenario
            rl_config = NoAlgorithmConfig(**vars(rl_config))
            rl_config.action_dim = self.agent_type.action_dim

            self.algorithm = RLAlgorithm(rl_config)

        if self.algorithm_name != 'no_algo': self._save_network_arch(network_classes)

        self.use_next_obs = self.algorithm.use_next_obs

        if algorithm == 'IQL' and self.rl_mode != 'eval':
            memory_path = os.path.join(get_project_paths("root_path"), self.algorithm.buffer_path)
            if not os.path.exists(memory_path):
                raise FileNotFoundError(f"Replay buffer path {memory_path} does not exist, cannot load replay buffer")
            self.logger.info("Loading replay buffer from {}".format(memory_path))
            self.memory = SwarmMemory.load(memory_path)
        else:
            self.memory = SwarmMemory(max_size=self.algorithm.memory_size,
                                      num_agents=self.num_agents,
                                      action_dim=self.agent_type.action_dim,
                                      use_intrinsic_reward=self.use_intrinsic_reward,
                                      use_next_obs=self.use_next_obs)

        # Update status dict only if not a sub_agent
        if not self.is_sub_agent:
            model_path = self.algorithm.get_model_path()
            logging_path = os.path.join(model_path, "logs")
            # If we resume training or starting in evaluation mode, we need to load the logger
            is_loading = self.resume or self.rl_mode == "eval"
            self.tensorboard = TensorboardLogger(log_dir=logging_path, resume=is_loading)

            if not is_loading:
                root_model_path = self.algorithm.model_path
                # Save own Config.Yaml when this is a fresh start
                config_path = os.path.join(root_model_path, "config.yaml")
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, sort_keys=False, indent=4)
            else:
                current_episode = self.tensorboard.episode
                current_objective = self.tensorboard.best_metrics["current_objective"]
                best_objective = self.tensorboard.best_metrics["best_objective"]
                train_step = self.tensorboard.train_step
                policy_updates = self.tensorboard.policy_updates
                self.sim_bridge.set("current_episode", current_episode)
                self.sim_bridge.set("current_objective", current_objective)
                self.sim_bridge.set("best_objective", best_objective)
                self.sim_bridge.set("train_step", train_step)
                self.sim_bridge.set("policy_updates", policy_updates)

            sim_bridge.set("model_path", self.algorithm.model_path)
            sim_bridge.set("model_name", self.algorithm.model_name)

            min_update = 0
            if self.algorithm_name == 'PPO':
                min_update = self.algorithm.horizon
            elif self.algorithm_name == 'IQL':
                min_update = len(self.memory)  # Always allow training for IQL because we always load a full buffer
            elif self.algorithm_name == 'TD3':
                min_update = self.algorithm.min_memory_size

            self.sim_bridge.set("min_update", min_update)

            if not self.resume: self.logger.info(f"Top Level Hierarchy: {self.agent_type.name}")

            max_train = at_dict["max_train"] if not algorithm == 'IQL' else config["algorithm"]["IQL"]["offline_updates"] + config["algorithm"]["IQL"]["online_updates"]
            self.evaluator = Evaluator(log_dir=logging_path,
                                       auto_train_dict=at_dict,
                                       max_train = max_train,
                                       no_gui=self.no_gui,
                                       start_eval=self.rl_mode == "eval",
                                       sim_bridge=self.sim_bridge,
                                       logger=self.tensorboard)

        if not self.resume:
            self.logger.info(f"{self.agent_type.name} initialized")
            if not self.is_sub_agent:
                self.logger.info(f"Algorithm: {self.algorithm_name}")
                self.logger.info(f"{self.num_agents} Agents control {drone_count} Drones")

    def _save_network_arch(self, network_classes):
        # Save own Network Structure for future reference and loading
        parent_network_dir = os.path.join(os.path.dirname(os.path.normpath(self.root_model_path)), "networks")
        is_autotrain_path = self.root_model_path[:-1].endswith("training_")
        use_parent_networkdir = is_autotrain_path and os.path.exists(parent_network_dir)
        network_dir = os.path.join(str(self.root_model_path), "networks") if not use_parent_networkdir else parent_network_dir

        network_name = "network_" + self.agent_type.short_name + ".py"
        file_path = os.path.join(network_dir, network_name)
        if not os.path.exists(network_dir):
            os.makedirs(network_dir, exist_ok=True)
        if not os.path.exists(file_path):
            src_dir = os.path.join(get_project_paths("root_path"), "src/pysim/networks")
            src_file = os.path.join(src_dir, network_name)
            shutil.copy(src_file, network_dir)
            self.logger.info(f"Network sources archived at {network_dir}")

    def _load_network_arch(self):
        # Get the modules that need to be loaded
        module_names = self.agent_type.get_module_names(self.algorithm_name)

        parent_network_dir = os.path.join(os.path.dirname(os.path.normpath(self.root_model_path)), "networks")
        is_autotrain_path = self.root_model_path.split(os.sep)[-1].split("_")[0] == "training"

        use_parent_networkdir = is_autotrain_path and os.path.exists(parent_network_dir)
        network_dir_load = os.path.join(str(self.root_model_path), "networks") if not use_parent_networkdir else parent_network_dir
        network_dir_std = os.path.join(get_project_paths("root_path"), "src/pysim/networks")
        network_name = "network_" + self.agent_type.short_name + ".py"

        file_path = os.path.join(network_dir_std, network_name)
        if os.path.exists(os.path.join(network_dir_load, network_name)):
            file_path = os.path.join(network_dir_load, network_name)

        if os.path.exists(file_path):
            modules = {}
            for m_name in module_names:
                spec = importlib.util.spec_from_file_location(m_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                modules[m_name] = module
            classes = tuple(getattr(module, m_name) for m_name in module_names if hasattr(module, m_name))
            self.logger.info(f"Network sources loaded from {file_path}")

            return classes
        else:
            self.logger.error(f"Network sources not found at {file_path}, this should not happen WHOOPS")
            raise FileNotFoundError(f"Network sources not found at {file_path}")

    def get_model_name(self, path: str, model_string: str, agent_type: str, is_loading_name: bool):
        """
        Get the model name based on the path and model name.
        First checks if the model name is a valid .pt file, then checks if the model exists at the given path.
        If it's just a string, it checks if it is one of the valid strings like "latest", "best_reward", or "best_objective".
        If not, it constructs a valid model name based on the algorithm and agent type.
        Finally, it will search the path for a valid model file.
        :param path: The path to the model.
        :param model_string: The model name from the config.
        :param agent_type: The type of the agent (e.g., "fly_agent", "explore_agent", "planner_agent").
        :param is_loading_name: If True, the model must exist at the given path.
        :return: A valid model name.
        """
        # Return back a log_dict, this later needs to be used for logging when we know the log_path
        log_dict = {"msg": None, "level": 0}
        if self.algorithm_name == 'no_algo':
            return "no_algo.pt", log_dict
        found_model = False
        if model_string and model_string.endswith(".pt"):
            # If the model name is a valid .pt file, check if it exists
            full_path = os.path.join(path, model_string)
            found_model = True
            if is_loading_name and not os.path.exists(full_path):
                log_dict["msg"] = f"Model {full_path} does not exist. Try searching for a valid model in {path}"
                log_dict["level"] = 30 # WARNING level
                found_model = False
            if found_model:
                return model_string, log_dict

        # Construct a default model name based on the algorithm and agent type
        # Only do this if we are not loading a model and just need a default name
        if not is_loading_name and not found_model:
            algorithm = self.algorithm_name.lower()
            return algorithm + "_" + agent_type + ".pt", log_dict

        # If the model name is just a string, check if it is one of the valid strings
        # This only applies if we are loading a model
        valid_strings = ["latest", "best_reward", "best_obj"]
        valid_models: list[str] = [file_name for file_name in os.listdir(path) if file_name.endswith(".pt")]

        if len(valid_models) == 0:
            log_dict["msg"] = f"No valid model files found in {path}. Please check the directory."
            log_dict["level"] = 40 # ERROR level
            if is_loading_name:
                raise FileNotFoundError(log_dict["msg"])
            return None, log_dict

        if model_string not in valid_strings:
            log_dict["msg"] = f"Provided model_string '{model_string}' is not a valid string. " \
                              f"Valid strings are: {valid_strings}. Using first valid model found in {path}." \
                              f"Valid models found: {valid_models}"
            log_dict["level"] = 30 # WARNING level
            return valid_models[0], log_dict  # Return the first valid model found in the directory

        # If the model string is one of the valid strings, construct the model name
        for model in valid_models:
            if model.split(".")[0].endswith(model_string):
                return model, log_dict

    def get_model_name_from_config(self, config):
        """
        Get the model name from the configuration.
        :param config: The configuration dictionary.
        :return: The model name.
        """
        if config["paths"]["model_name"] != "" and config["paths"]["model_name"] is not None\
                and config["paths"]["model_name"].endswith(".pt"):
            return config["paths"]["model_name"]
        else:
            algorithm = self.algorithm_name.lower()
            agent_type = config["settings"]["hierarchy_type"].lower()
            return algorithm + "_" + agent_type + ".pt"

    def get_model_name_from_string(self, model_string: str, agent_type: str):
        """
        Get the model name from a string.
        :param model_string: The model name string.
        :param agent_type: The type of the agent (e.g., "fly_agent", "explore_agent", "planner_agent").
        :return: The model name.
        """
        if not (model_string and model_string != ""):
            self.logger.warning("default_model_name cannot be None or empty")
            raise ValueError("default_model_name cannot be None or empty")

        valid_strings = ["latest", "best_reward", "best_objective"]

        if not (model_string in valid_strings or model_string.endswith(".pt")):
            self.logger.warning(
                "default_model_name must be either {} or end with .pt, was: {}".format(
                    valid_strings, model_string
                )
            )
            raise ValueError(
                "default_model_name must be either {} or end with .pt, was: {}".format(
                    valid_strings, model_string
                )
            )

        model_name = model_string
        if model_string in valid_strings:
            algorithm = self.algorithm_name
            model_name = algorithm.lower() + "_" + agent_type.lower() + "_" + model_string + ".pt"
        return model_name

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

    def should_train(self):
        if self.algorithm_name == 'PPO':
            return len(self.memory) >= self.algorithm.horizon
        elif self.algorithm_name == 'IQL':
            # Train always during offline phase, during online phase check policy frequency
            return (self.env_step < self.algorithm.offline_updates) or (self.env_step % self.algorithm.policy_freq == 0)
        elif self.algorithm_name == 'TD3':
            return len(self.memory) >= self.algorithm.min_memory_size
        else:
            raise NotImplementedError("Algorithm {} not implemented".format(self.algorithm_name))

    def load_model(self, change_status=False, new_rl_mode=None):
        # Load model if possible, return new rl_mode and possible console string
        log = ""
        if self.algorithm_name == 'no_algo':
            log += "No algorithm used, no model to load"

        probe_rl_mode = self.rl_mode if new_rl_mode is None else new_rl_mode
        train = True if probe_rl_mode == "train" else False
        if not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                log += (f"Load model from checkpoint: {os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}"
                        f" - Model set to evaluation mode")
            else:
                self.algorithm.set_train()
                self.rl_mode = "train"
                log +=  "No checkpoint found to evaluate model, start training from scratch"
        elif self.resume:
            if self.algorithm.load():
                self.algorithm.set_train()
                log += f"Load model from checkpoint: {os.path.join(self.algorithm.loading_path, self.algorithm.loading_name)}" \
                       f" - Resume Training from checkpoint"
            else:
                self.algorithm.set_train()
                log += "No checkpoint found to resume training, start training from scratch"
        else:
            self.algorithm.set_train()
            log += "Training from scratch"

        self.logger.info(log)

        if change_status:
            self.sim_bridge.set("rl_mode", self.rl_mode)

    def add_memory_entry(self, obs, actions, action_logprobs, rewards, terminals, next_obs=None, intrinsic_rewards=None):
        self.memory.add(obs, actions, action_logprobs, rewards, terminals, next_obs=next_obs, intrinsic_reward=intrinsic_rewards)

    def update_status(self):
        model_path = self.sim_bridge.get("model_path")
        model_name = self.sim_bridge.get("model_name")
        self.algorithm.set_paths(model_path, model_name)

        self.sim_bridge.set("obs_collected", len(self.memory))

        if self.sim_bridge.get("rl_mode") != self.rl_mode:
            self.logger.warning(f"RL Mode changed from {self.rl_mode} to {self.sim_bridge.get('rl_mode')}")
            self.ctrl_ctr = 0  # Reset control counter to avoid issues
            self.cached_actions = None
            self.cached_logprobs = None
            self.rl_mode = self.sim_bridge.get("rl_mode")
            self.algorithm.set_train() if self.rl_mode == "train" else self.algorithm.set_eval()

    def restruct_current_obs(self, observations):
        self.current_obs = self.restructure_data(observations)

    def intrinsic_reward(self):
        intrinsic_reward = None
        if self.use_intrinsic_reward:
            intrinsic_reward = self.agent_type.get_intrinsic_reward(self.current_obs)
            # The environment has not been reset so we can send the intrinsic
            # reward to the model (only for displaying purposes)
            if not any(terminals_vector):
                intrinsic_reward = intrinsic_reward.detach().cpu().numpy().tolist()
                self.sim_bridge.set("intrinsic_reward", intrinsic_reward)
                engine.SendRLStatusToModel(self.sim_bridge.status)
                engine.UpdateReward()
        return intrinsic_reward

    def train_loop(self, engine):
        # Check IQL conditions
        skip_step = False if not self.algorithm_name == "IQL" else (self.env_step < self.algorithm.offline_updates)
        next_obs = None
        if not skip_step:
            # Only sample new action at the start of the control
            start_of_control = self.ctrl_ctr % self.frame_skips == 0
            if start_of_control:
                if self.current_obs is None:
                    self.current_obs = self.restructure_data(engine.GetObservations())
                actions, action_logprobs = self.act(self.current_obs)
                self.cached_actions = actions
                self.cached_logprobs = action_logprobs

            next_obs_, rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.cached_actions)

            self.ctrl_ctr += 1
            end_of_control = terminal_result.env_reset or ((self.ctrl_ctr % self.frame_skips) == 0)

            if not end_of_control:
                # No memory adding or training until the end of the control
                return

            next_obs = self.restructure_data(next_obs_)

            # Intrinsic Reward Calculation (optinoal)
            intrinsic_reward = self.intrinsic_reward()

            # Memory Adding
            self.add_memory_entry(self.current_obs,
                                  self.cached_actions,
                                  self.cached_logprobs,
                                  rewards,
                                  terminals_vector,
                                  next_obs=next_obs if self.use_next_obs else None,
                                  intrinsic_rewards=intrinsic_reward)

            # Update the Logger before checking if we should train, so that the logger has the latest information
            # to calculate the objective percentage and best reward
            self.update_logging(terminal_result)

            # Only change the env_reset after actually taking a step
            self.env_reset = terminal_result.env_reset

        # Training
        if self.should_train():
            self.algorithm.apply_manual_decay(self.sim_bridge.get("train_step"))
            self.update(mini_batch_size=self.algorithm.batch_size, next_obs=next_obs)
            if self.use_intrinsic_reward and self.algorithm == 'PPO':
                self.agent_type.update_rnd_model(self.memory, self.algorithm.horizon, self.algorithm.batch_size)
            if self.algorithm.clear_memory:
                self.memory.clear_memory()

        # Advance to the next step
        self.env_step += 1
        self.current_obs = next_obs
        self.handle_env_reset()

    def eval_loop(self, engine, evaluate=False):

        start_of_control = self.ctrl_ctr % self.frame_skips == 0
        if start_of_control:
            if self.current_obs is None:
                self.current_obs = self.restructure_data(engine.GetObservations())
            self.cached_actions = self.act_certain(self.current_obs)

        obs_, rewards, terminals_vector, terminal_result, percent_burned = self.step_agent(engine, self.cached_actions)

        # Only do these extra steps when you SHOULD populate memory
        if self.save_replay_buffer:
            # Intrinsic Reward Calculation (optional)
            intrinsic_reward = self.intrinsic_reward()
            if self.env_step % 5000 == 0:
                self.logger.info(f"Replay Buffer size: {len(self.memory)}/{int(self.save_size)}")
            # Memory Adding
            self.add_memory_entry(self.current_obs,
                                  self.cached_actions,
                                  None,
                                  rewards,
                                  terminals_vector,
                                  next_obs=self.restructure_data(obs_) if self.use_next_obs else None,
                                  intrinsic_rewards=intrinsic_reward)
            if len(self.memory) >= self.save_size:
                mem_name = os.path.join(self.root_model_path, 'memory.pkl')
                self.memory.save(mem_name)
                self.logger.info(f'Replay Buffer saved at {mem_name}')
                self.sim_bridge.set("agent_online", False)

        self.ctrl_ctr += 1
        end_of_control = terminal_result.env_reset or ((self.ctrl_ctr % self.frame_skips) == 0)

        if not end_of_control:
            # No Evaluation until the end of the control
            return False, False

        self.current_obs = self.restructure_data(obs_)
        if evaluate and not self.save_replay_buffer:
            flags = self.evaluator.evaluate(rewards, terminal_result, percent_burned)
            self.check_reset(flags)

        self.env_step += 1

        return terminal_result.any_succeeded, terminal_result.env_reset

    def check_reset(self, flags):
        """
        Check if the environment should be reset based on evaluation flags.
        :param flags: Dictionary containing evaluation flags.
        """
        if flags.get("reset", False):
            if flags.get("auto_train", False) and flags.get("auto_train_not_finished", True):
                self.sim_bridge.set("rl_mode", "train")
                self.sim_bridge.set("agent_is_running", True)
                self.algorithm.reset()

                # Reset the Logger (Only in Auto Training Case, otherwise some values might need a reset)
                if not self.is_sub_agent:  # should not really be False like ever, but just in case
                    self.tensorboard.close()
                    model_path = self.algorithm.get_model_path()
                    logging_path = os.path.join(model_path, "logs")
                    # Remove and add another FileHandler to avoid issues
                    logger = logging.getLogger()
                    for handler in logger.handlers[:]:
                        if isinstance(handler, logging.FileHandler):
                            logger.removeHandler(handler)
                            handler.close()
                    if not os.path.exists(logging_path):
                        os.makedirs(logging_path, exist_ok=True)
                    logging_file = os.path.join(logging_path, "logging.log")
                    file_handler = logging.FileHandler(logging_file, encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG)
                    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
                    file_handler.setFormatter(formatter)
                    logger.addHandler(file_handler)
                    self.tensorboard = TensorboardLogger(log_dir=logging_path)
                    self.evaluator.tb_logger = self.tensorboard

                # Reset memory
                self.memory.clear_memory()

            self.hierarchy_steps = 0
            self.ctrl_ctr = 0  # Reset control counter to avoid issues
            self.cached_actions = None
            self.cached_logprobs = None
            self.env_step = 0
            self.current_obs = None
            self.env_reset = True
            # self.sim_bridge.set("env_reset", True)

    def get_final_metric(self, metric_name: str):
        if not self.is_sub_agent:
            return self.evaluator.final_metrics()[metric_name]
        else:
            self.logger.warning("Sub agents do not have a Tensorboard Logger, returning None")
            return None

    def step_agent(self, engine, actions):
        env_step = engine.Step(self.agent_type.name, self.get_action(actions))

        rewards = env_step.rewards
        observations = env_step.observations
        percent_burned = env_step.percent_burned
        terminals = env_step.terminals
        terminal_result = env_step.summary
        all_terminals = [t.is_terminal for t in terminals if t is not None]

        return observations, rewards, all_terminals, terminal_result, percent_burned

    def step_without_network(self, engine):
        agent_actions = self.get_action([[0,0] for _ in range(self.num_agents)])
        env_step = engine.Step(self.agent_type.name, agent_actions)
        self.env_reset = env_step.summary.env_reset
        self.handle_env_reset()

    def handle_env_reset(self):
        if self.env_reset:
            self.sim_bridge.set("current_episode", self.sim_bridge.get("current_episode") + 1)
            self.ctrl_ctr = 0  # Reset control counter to avoid issues
            self.current_obs = None
            self.cached_actions = None
            self.cached_logprobs = None

    def update(self, mini_batch_size, next_obs):
        try:
            self.algorithm.update(self.memory, mini_batch_size, next_obs, self.tensorboard)
        except Exception as e:
            self.logger.error(f"Error during algorithm update: {e}")
            raise e
        self.sim_bridge.add_value("train_step", 1)
        self.tensorboard.train_step += 1

        if self.algorithm_name == 'PPO':
            policy_updates = self.algorithm.horizon // self.algorithm.batch_size * self.algorithm.k_epochs
            self.sim_bridge.add_value("policy_updates", policy_updates)
            self.tensorboard.policy_updates += policy_updates
        elif self.algorithm_name == 'IQL':
            batches_per_epoch, epochs = self.algorithm.get_batches_and_epochs(len(self.memory), mini_batch_size)
            self.sim_bridge.add_value("policy_updates", batches_per_epoch * epochs)
            self.tensorboard.policy_updates += self.algorithm.k_epochs
            if self.sim_bridge.get("train_step") == self.algorithm.offline_updates:
                self.logger.info("Finished Offline Updates.")
                self.algorithm.offline_end = True
                if not self.algorithm.online_updates > 0:
                    self.logger.info("No Online Updates specified, stopping training.")
                else:
                    self.logger.info(f"Starting Online Updates.")
        elif self.algorithm_name == 'TD3':
            self.sim_bridge.add_value("policy_updates", self.algorithm.k_epochs)
            self.tensorboard.policy_updates += self.algorithm.k_epochs

        if self.evaluator.on_update():
            # Need to inject and load BEST model here
            model_name, _ = self.get_model_name(path=str(self.algorithm.get_model_path()), model_string="best_obj",
                                                agent_type=self.agent_type_str, is_loading_name=True)
            self.algorithm.loading_path = self.root_model_path
            self.algorithm.loading_name = model_name
            self.load_model(new_rl_mode="eval")

        self.tensorboard.summarize()

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs.ravel() if action_logprobs is not None else None

    def get_action(self, actions):
        return self.agent_type.get_action(actions)

    def update_logging(self, terminal_result):
        self.tensorboard.log_step(terminal_result)
        self.tensorboard.calc_current_objective()

        self.sim_bridge.set("objective", self.tensorboard.best_metrics["current_objective"])
        self.sim_bridge.set("best_objective", self.tensorboard.get_best_objective()[1])

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

    def restructure_data(self, observations_):
        return self.agent_type.restructure_data(observations_)




