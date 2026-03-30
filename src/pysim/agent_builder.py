import os
import logging
import yaml
import shutil
import importlib.util

from algorithms.ppo import PPO
from algorithms.iql import IQL
from algorithms.rl_algorithm import RLAlgorithm
from algorithms.td3 import TD3
from algorithms.rl_config import RLConfig, PPOConfig, IQLConfig, TD3Config, NoAlgorithmConfig, override_from_dict
from utils import SimulationBridge, get_project_paths, remove_suffix
from memory import SwarmMemory
from training_monitor import TrainingMonitor
from explore_agent import ExploreAgent
from flying_agent import FlyAgent
from planner_agent import PlannerAgent
from agent_handler import AgentHandler, FrameSkipController

def _query_schema_dims(agent_name):
    """Query C++ FeatureSchema for agent/neighbor dims. Returns (agent_dim, neighbor_dim)."""
    try:
        import firesim
        info = firesim.GetFeatureSchemaInfo(agent_name)
        agent_dim = info["agent"]["dims"] if "agent" in info else 9
        neighbor_dim = info["neighbors"]["dims"] if "neighbors" in info else 4
        return agent_dim, neighbor_dim
    except Exception:
        return 9, 4  # fallback defaults


def resolve_model_name(path: str, model_string: str, agent_type: str,
                       algorithm_name: str, is_loading_name: bool):
    """Resolve a model filename from config/path.

    Returns (model_name, log_dict) where log_dict has 'msg' and 'level' keys
    for deferred logging (logger may not be set up yet at call time).
    """
    log_dict = {"msg": None, "level": 0}
    if algorithm_name == 'no_algo':
        return "no_algo.pt", log_dict

    found_model = False
    if model_string and model_string.endswith(".pt"):
        full_path = os.path.join(path, model_string)
        found_model = True
        if is_loading_name and not os.path.exists(full_path):
            log_dict["msg"] = f"Model {full_path} does not exist. Try searching for a valid model in {path}"
            log_dict["level"] = 30
            found_model = False
        if found_model:
            return model_string, log_dict

    if not is_loading_name and not found_model:
        algorithm = algorithm_name.lower()
        return algorithm + "_" + agent_type + ".pt", log_dict

    valid_strings = ["latest", "best_reward", "best_obj"]
    valid_models = [f for f in os.listdir(path) if f.endswith(".pt")]

    if len(valid_models) == 0:
        log_dict["msg"] = f"No valid model files found in {path}. Please check the directory."
        log_dict["level"] = 40
        if is_loading_name:
            raise FileNotFoundError(log_dict["msg"])
        return None, log_dict

    if model_string not in valid_strings:
        log_dict["msg"] = (f"Provided model_string '{model_string}' is not a valid string. "
                           f"Valid strings are: {valid_strings}. Using first valid model found in {path}."
                           f"Valid models found: {valid_models}")
        log_dict["level"] = 30
        return valid_models[0], log_dict

    for model in valid_models:
        if model.split(".")[0].endswith(model_string):
            return model, log_dict


class AgentBuilder:
    """Constructs AgentHandler instances from config."""

    def __init__(self, config: dict, sim_bridge: SimulationBridge):
        self.config = config
        self.sim_bridge = sim_bridge

    def build(self, agent_type: str, mode: str, subtype: str = None,
              is_sub_agent: bool = True) -> AgentHandler:
        config = self.config
        sim_bridge = self.sim_bridge
        no_gui = config["settings"]["mode"] == 2

        agent_dict = config["environment"]["agent"][agent_type]
        vision_range = agent_dict["view_range"]
        time_steps = agent_dict["time_steps"]
        algorithm = agent_dict["algorithm"]
        map_size = config["fire_model"]["simulation"]["grid"]["exploration_map_size"]

        fsc = FrameSkipController(agent_dict["frame_skips"])

        supported_algos = ['no_algo', 'PPO', 'IQL', 'TD3']
        assert algorithm in supported_algos, f"Algorithm {algorithm} not supported, only {supported_algos} are supported"

        algorithm_name = algorithm
        resume = config["settings"]["resume"]
        log_eval = config["settings"]["log_eval"]

        at_dict = config["settings"]["auto_train"]
        use_auto_train = at_dict["use_auto_train"]

        drone_count = agent_dict["num_agents"]
        agent_type_obj = self._get_agent_from_type(
            agent_type=agent_type if subtype is None else subtype,
            num_drones=drone_count
        )
        num_agents = agent_type_obj.get_num_agents(agent_dict["num_agents"])
        hierarchy_level = agent_type_obj.get_hierarchy_level()
        use_intrinsic_reward = agent_type_obj.use_intrinsic_reward

        save_replay_buffer = config["settings"]["save_replay_buffer"]
        save_size = config["settings"]["save_size"]

        if use_intrinsic_reward:
            agent_type_obj.initialize_rnd_model(vision_range, drone_count=num_agents,
                                                map_size=map_size, time_steps=time_steps)

        # --- Path resolution ---
        root_path = get_project_paths("root_path")
        default_model_folder = None
        model_name = None
        loading_path, loading_name, model_path = None, None, None

        if is_sub_agent:
            if agent_type != "explore_agent":
                assert subtype in ["ExploreFlyAgent", "PlannerFlyAgent"], f"{subtype} is not a valid sub agent type"
                agent_type_d = "explore_agent" if subtype == "ExploreFlyAgent" else "planner_agent"
                default_model_folder = config["environment"]["agent"][agent_type_d]["default_model_folder"]
                default_model_name = config["environment"]["agent"][agent_type_d]["default_model_name"]
                loading_path = os.path.join(root_path, default_model_folder)
            else:
                loading_path = ""
                default_model_name = None
            loading_name, later_logs = resolve_model_name(
                path=str(loading_path), model_string=default_model_name,
                agent_type=agent_type, algorithm_name=algorithm_name, is_loading_name=True
            )
            model_path = loading_path
        else:
            is_loading = resume or mode == "eval"
            model_path = os.path.join(root_path, config["paths"]["model_directory"])
            model_string, later_logs = resolve_model_name(
                path=str(model_path), model_string=config["paths"]["model_name"],
                agent_type=agent_type, algorithm_name=algorithm_name, is_loading_name=is_loading
            )
            model_name = model_string if not is_loading else remove_suffix(model_string)
            loading_path = model_path if is_loading else None
            loading_name = model_string if is_loading else None
            if not is_loading and not os.path.exists(model_path):
                os.makedirs(model_path)

        root_model_path = model_path

        # --- Logging setup ---
        logging_file = ""
        if not is_sub_agent:
            logger_root = logging.getLogger()
            for handler in logger_root.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger_root.removeHandler(handler)
                    handler.close()
            mp = str(model_path) if not use_auto_train else os.path.join(str(model_path), "training_1")
            logging_dir = os.path.join(mp, "logs")
            if not os.path.exists(logging_dir):
                os.makedirs(logging_dir, exist_ok=True)
            logging_file = os.path.join(logging_dir, "logging.log")
            file_handler = logging.FileHandler(logging_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
            file_handler.setFormatter(formatter)
            logging.getLogger().addHandler(file_handler)

        loggername = subtype if subtype else agent_type
        logger = logging.getLogger(loggername)
        if not is_sub_agent:
            logger.info(f"Logging File at {logging_file}")
        if later_logs["msg"] is not None:
            logger.log(later_logs["level"], later_logs["msg"])

        # --- Config injection for sub-agents ---
        share_encoder = config["algorithm"]["share_encoder"]
        use_tanh_dist = config["algorithm"]["use_tanh_dist"]
        collision = config["environment"]["agent"]["fly_agent"]["collision"]
        if default_model_folder is not None:
            config_path = os.path.join(get_project_paths("root_path"), default_model_folder, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    injecting_config = yaml.safe_load(f)
                share_encoder = injecting_config["algorithm"]["share_encoder"]
                use_tanh_dist = injecting_config["algorithm"]["use_tanh_dist"]
                collision = injecting_config["environment"]["agent"]["fly_agent"]["collision"]
                time_steps = injecting_config["environment"]["agent"]["fly_agent"]["time_steps"]
                logger.info("Injected network parameters from sub agent config at {}".format(config_path))
            else:
                logger.warning("No config file found at {}, cannot inject network parameters".format(config_path))

        # --- Query schema dims from C++ FeatureSchema ---
        agent_dim, neighbor_dim = _query_schema_dims(agent_type_obj.name)

        # --- Build RL config ---
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
                             share_encoder=share_encoder,
                             use_tanh_dist=use_tanh_dist,
                             collision=collision,
                             agent_dim=agent_dim,
                             neighbor_dim=neighbor_dim)

        algo_overrides = config["algorithm"].get(algorithm_name, {})

        # --- Load network architecture ---
        if algorithm_name != 'no_algo':
            try:
                network_classes = self._load_network_arch(root_model_path, agent_type_obj, algorithm_name)
            except FileNotFoundError as e:
                logger.warning(f"{e}. Falling back to default network classes")
                network_classes = agent_type_obj.get_network(algorithm=algorithm_name)
                if not isinstance(network_classes, (list, tuple)):
                    network_classes = (network_classes,)
        else:
            network_classes = None

        # --- Instantiate algorithm ---
        if algorithm == 'PPO':
            rl_config = PPOConfig(**vars(rl_config),
                                  use_categorical=agent_type == "planner_agent",
                                  use_variable_state_masks=agent_type == "planner_agent")
            rl_config.use_next_obs = False if not save_replay_buffer else True
            rl_config = override_from_dict(rl_config, algo_overrides)
            algo_instance = PPO(network=network_classes, config=rl_config)
        elif algorithm == 'IQL':
            rl_config = IQLConfig(**vars(rl_config))
            rl_config.action_dim = agent_type_obj.action_dim
            rl_config.clear_memory = False
            rl_config = override_from_dict(rl_config, algo_overrides)
            algo_instance = IQL(network=network_classes, config=rl_config)
        elif algorithm == 'TD3':
            rl_config = TD3Config(**vars(rl_config))
            rl_config.action_dim = agent_type_obj.action_dim
            rl_config.clear_memory = False
            rl_config = override_from_dict(rl_config, algo_overrides)
            algo_instance = TD3(network=network_classes, config=rl_config)
        elif algorithm == 'no_algo':
            rl_config = NoAlgorithmConfig(**vars(rl_config))
            rl_config.action_dim = agent_type_obj.action_dim
            algo_instance = RLAlgorithm(rl_config)

        if algorithm_name != 'no_algo':
            self._save_network_arch(root_model_path, agent_type_obj, network_classes)

        use_next_obs = algo_instance.use_next_obs

        # --- Build memory ---
        if algorithm == 'IQL' and mode != 'eval':
            memory_path = os.path.join(get_project_paths("root_path"), algo_instance.buffer_path)
            if not os.path.exists(memory_path):
                raise FileNotFoundError(f"Replay buffer path {memory_path} does not exist")
            logger.info("Loading replay buffer from {}".format(memory_path))
            memory = SwarmMemory.load(memory_path)
        else:
            memory = SwarmMemory(max_size=algo_instance.memory_size,
                                 num_agents=num_agents,
                                 action_dim=agent_type_obj.action_dim,
                                 use_intrinsic_reward=use_intrinsic_reward,
                                 use_next_obs=use_next_obs)

        # --- Build TrainingMonitor (main agent only) ---
        monitor = None
        if not is_sub_agent:
            algo_model_path = algo_instance.get_model_path()
            logging_path_m = os.path.join(algo_model_path, "logs")
            is_loading = resume or mode == "eval"
            monitor = TrainingMonitor(
                logging_path=logging_path_m, config=config, sim_bridge=sim_bridge,
                algorithm=algo_instance, algorithm_name=algorithm_name,
                is_loading=is_loading, resume=resume, no_gui=no_gui,
                rl_mode=mode, log_eval=log_eval
            )
            if algorithm_name == 'IQL':
                monitor.set_iql_min_update(len(memory))
            if not resume:
                logger.info(f"Top Level Hierarchy: {agent_type_obj.name}")

        # --- Construct AgentHandler ---
        handler = AgentHandler(
            agent_type=agent_type_obj,
            agent_type_str=agent_type,
            algorithm=algo_instance,
            algorithm_name=algorithm_name,
            memory=memory,
            monitor=monitor,
            sim_bridge=sim_bridge,
            fsc=fsc,
            hierarchy_level=hierarchy_level,
            use_intrinsic_reward=use_intrinsic_reward,
            is_sub_agent=is_sub_agent,
            use_next_obs=use_next_obs,
            save_replay_buffer=save_replay_buffer,
            save_size=save_size,
            root_model_path=root_model_path,
            rl_mode=mode,
            resume=resume,
            no_gui=no_gui,
            num_agents=num_agents,
            logger=logger,
        )

        if not resume:
            logger.info(f"{agent_type_obj.name} initialized")
            if not is_sub_agent:
                logger.info(f"Algorithm: {algorithm_name}")
                logger.info(f"{num_agents} Agents control {drone_count} Drones")

        return handler

    @staticmethod
    def _get_agent_from_type(agent_type, num_drones):
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
        return PlannerAgent(num_drones)

    @staticmethod
    def _load_network_arch(root_model_path, agent_type_obj, algorithm_name):
        module_names = agent_type_obj.get_module_names(algorithm_name)

        parent_network_dir = os.path.join(os.path.dirname(os.path.normpath(root_model_path)), "networks")
        is_autotrain_path = root_model_path.split(os.sep)[-1].split("_")[0] == "training"
        use_parent_networkdir = is_autotrain_path and os.path.exists(parent_network_dir)
        network_dir_load = os.path.join(str(root_model_path), "networks") if not use_parent_networkdir else parent_network_dir
        network_dir_std = os.path.join(get_project_paths("root_path"), "src/pysim/networks")
        network_name = "network_" + agent_type_obj.short_name + ".py"

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
            return classes
        else:
            raise FileNotFoundError(f"Network sources not found at {file_path}")

    @staticmethod
    def _save_network_arch(root_model_path, agent_type_obj, network_classes):
        parent_network_dir = os.path.join(os.path.dirname(os.path.normpath(root_model_path)), "networks")
        is_autotrain_path = root_model_path[:-1].endswith("training_")
        use_parent_networkdir = is_autotrain_path and os.path.exists(parent_network_dir)
        network_dir = os.path.join(str(root_model_path), "networks") if not use_parent_networkdir else parent_network_dir

        network_name = "network_" + agent_type_obj.short_name + ".py"
        file_path = os.path.join(network_dir, network_name)
        if not os.path.exists(network_dir):
            os.makedirs(network_dir, exist_ok=True)
        if not os.path.exists(file_path):
            src_dir = os.path.join(get_project_paths("root_path"), "src/pysim/networks")
            src_file = os.path.join(src_dir, network_name)
            shutil.copy(src_file, network_dir)
