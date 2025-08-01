from typing import Union, Any

import numpy as np
import torch
import json
import os
from pathlib import Path
from collections import deque, defaultdict
from torch.utils.tensorboard import SummaryWriter

class SimulationBridge:
    def __init__(self, config: dict):
        """
        A bridge to pass status information between different components.
        :param status: A dictionary containing status information.
        """
        self.status = self.build_status(config)

    @staticmethod
    def build_status(config):

        hierarchy = config["settings"]["hierarchy_type"]
        environment = config["environment"]
        agent = environment["agent"][hierarchy]
        root_path = get_project_paths("root_path")
        auto_train_dict = config["settings"]["auto_train"]
        status = {
            # Non-Settable Parameters (either flags or communication with C++)
            "obs_collected": 0,  # Used by GUI to show the number of collected observations
            "min_update": 0,  # How many obs before updating the policy? Decided by the RL Algorithm
            "agent_online": True,
            "train_episode": 0,  # Current training episode
            "train_step": 0,  # How often did you train?
            "current_episode": 0,
            "policy_updates": 0,  # How often did you update the policy?
            "objective": 0,  # Tracking the Percentage of the Objective
            "best_objective": 0,  # Best Objective so far
            # Settable Parameters (can be changed by the user, BUT should really only be changed through the config file)
            "hierarchy_type": hierarchy,  # Either fly_agent, explore_agent or planner_agent
            "rl_mode": config["settings"]["rl_mode"],  # "train" or "eval"
            "resume": config["settings"]["resume"],  # If True, the agent will resume training from the last checkpoint
            "model_path": os.path.join(root_path, config["paths"]["model_directory"]),
            "model_name": config["paths"]["model_name"],  # Name of the model to load or save
            "num_agents": agent["num_agents"],  # Number of agents in the environment
            "rl_algorithm": "PPO",  # RL Algorithm to use, either PPO, IQL, TD3
            "auto_train": auto_train_dict["use_auto_train"], # If True, the agent will train several episodes and then evaluate
            "train_episodes": auto_train_dict["train_episodes"],  # Number of total trainings containing each max_train steps
            "max_eval": auto_train_dict["max_eval"],  # Number of Environments to run before stopping evaluation
            "max_train": auto_train_dict["max_train"],  # Number of train_steps before stopping training
        }

        return status

    def set_status(self, status):
        """
        Set the status dictionary.
        :param status: The status dictionary to set.
        """
        if not isinstance(status, dict):
            raise ValueError("Status must be a dictionary.")
        self.status = status

    def get_status(self):
        """
        Get the current status dictionary.
        :return: The status dictionary.
        """
        return self.status

    def add_value(self, key, value):
        """
        Adds the value to the current value of the key in the status dictionary.
        :param key: The key to add the value to.
        :param value: The value to add.
        """
        if key not in self.status:
            raise KeyError(f"Key '{key}' not found in status dictionary.")
        if isinstance(value, (int, float)):
            self.status[key] += value
        else:
            raise ValueError(f"Value for key '{key}' must be an int or float, got {type(value)} instead.")

    def set(self, key, value):
        """
        Set a value in the status dictionary.
        :param key: The key to set the value for.
        :param value: The value to set.
        """
        self.status[key] = value

    def get(self, key):
        """
        Get a value from the status dictionary.
        :param key: The key to retrieve the value for.
        :return: The value associated with the key.
        """
        return self.status[key]

def get_in_features_2d(h_in, w_in, layers_dict):
    for layer in layers_dict:
        padding = layer['padding']
        dilation = layer['dilation']
        kernel_size = layer['kernel_size']
        stride = layer['stride']

        h_in = ((h_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        w_in = ((w_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1

    return h_in * w_in


def get_in_features_3d(h_in, w_in, d_in, layers_dict):
    for layer in layers_dict:
        padding = layer['padding']
        dilation = layer['dilation']
        kernel_size = layer['kernel_size']
        stride = layer['stride']

        d_in = ((d_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) // stride[0]) + 1
        h_in = ((h_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) // stride[1]) + 1
        w_in = ((w_in + 2 * padding[2] - dilation[2] * (kernel_size[2] - 1) - 1) // stride[2]) + 1

        if d_in <= 0 or h_in <= 0 or w_in <= 0:
            raise ValueError(f"Invalid dimensions after layer: {d_in}x{h_in}x{w_in}")

    return d_in * h_in * w_in


def find_project_root(current_path):
    # Search for a prominent marker of the project root
    for parent in current_path.parents:
        if (parent / '.git').is_dir():
            return parent
    raise Exception('Could not find project root directory')

def get_project_paths(name):
    script_directory = os.path.dirname(os.path.abspath(__file__))
    root_directory = find_project_root(Path(script_directory))

    with open(Path(root_directory / 'project_paths.json'), 'r') as f:
        std_paths = json.load(f)

    if name not in std_paths.keys():
        raise ValueError(f"Project path '{name}' not found in project_paths.json")

    return std_paths[name]


def initialize_output_weights(m, out_type):
    """
    Initialize the weights of the output layer of the actor and critic networks
    :param m: the layer to initialize
    :param out_type: the type of the output layer (actor or critic)
    """
    if isinstance(m, torch.nn.Linear):
        if out_type == 'actor':
            torch.nn.init.orthogonal_(m.weight, gain=0.1)
        elif out_type == 'critic':
            torch.nn.init.orthogonal_(m.weight, gain=1.0)
        else: # Hidden Layers
            torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain('relu'))
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


def standardize(tensor):
    """
    Standardize a tensor to mean zero and standard deviation one
    """
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class DataCollector(object):
    """
    This class is used to collect data from the environment, store it in histograms and log it to tensorboard.
    """


class RunningMeanStd(object):
    """
    This class is used to calculate the running mean and standard deviation of a data.
    """
    # from https://github.com/openai/baselines
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.epsilon = 1e-8
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

    def get_std(self):
        return np.sqrt(self.var + self.epsilon)


class Logger:
    """
    Logger class for logging training and evaluation metrics using TensorBoard.
    :param log_dir: Directory where the logs will be saved.
    :param resume: If True, resume logging from the last checkpoint.
    """
    def __init__(self, log_dir, resume=False):
        self.log_dir = log_dir
        self.writer = SummaryWriter(log_dir)
        self.metrics = defaultdict(list)
        self.histograms = defaultdict(list)
        self.episode_steps = 0 # Steps in the current episode
        self.objectives = deque(maxlen=100)
        self.logging_step = 0
        self.episode = 0 # Current episode number
        self.best_metrics = {"current_objective": 0.0, "best_objective": -np.inf,
                             "current_reward": 0.0, "best_reward": -np.inf}
        # Non-essential metrics that still need to be saved
        self.train_step = 0  # How often did you train?
        self.policy_updates = 0  # How often did you update the policy?
        self.episode_ended = False

        if resume:
            self._load_state()

    def summarize(self):
        # Log all scalar metrics (averaged over the episode)
        for tag, values in self.metrics.items():
            if values:
                self.log_scalar(tag, np.mean(values))
        # Log all histogram metrics
        for tag, values in self.histograms.items():
            if values:
                self.log_histogram(tag, values)
        # Log Episode Length, Objectives, etc.
        if self.episode_ended:
            self.log_scalar("Sanitylogs/Episode_Steps", self.episode_steps)
            self.episode_steps = 0
        if len(self.objectives) == self.objectives.maxlen:
            self.log_scalar("Objective", np.mean(self.objectives))

        # Clear for next summary
        self.metrics.clear()
        self.histograms.clear()
        self.episode_ended = False
        self.logging_step += 1
        self.save_state()

    def log_hparams(self, hparams: dict):
        """
        Logs hyperparameters to TensorBoard.
        :param hparams: Dictionary of hyperparameters.
        """
        if not isinstance(hparams, dict):
            raise ValueError("Hyperparameters must be a dictionary.")
        # Ensure all values are bool, str, float, int or None
        allowed_types = (bool, str, float, int, type(None))
        for key, value in hparams.items():
            if not isinstance(value, allowed_types):
                raise ValueError(f"Hyperparameter '{key}' must be of {allowed_types}, got {type(value)} instead.")

        self.writer.add_hparams(hparams, {"hparam/metric": 0.0})

    def log_step(self, terminal_result: dict):
        """
        Marks the end of an episode and logs the episode length and objective.
        :param terminal_result: Dictionary containing terminal results of the episode.
        """
        self.episode_steps += 1
        env_done = terminal_result["EnvReset"]

        if env_done:
            self.episode += 1
            objective_reached = not terminal_result["OneAgentDied"]
            self.objectives.append(1 if objective_reached else 0)
            self.episode_ended = True

    def add_metric(self, tag, value, hist=False):
        self.histograms[tag].append(value)
        self.metrics[tag].append(value)

    def log_scalar(self, tag, value, step: int = None):
        t = self.logging_step if step is None else step
        self.writer.add_scalar(tag, value, t, new_style=True)

    def log_histogram(self, tag, values, step: int = None):
        t = self.logging_step if step is None else step
        if isinstance(values, list):
            values = np.array(values)
        self.writer.add_histogram(tag + "_hist", values, t, bins='auto')

    def log_text(self, tag, text, step: int = None):
        t = self.logging_step if step is None else step
        self.writer.add_text(tag, text, t)

    def _load_state(self):
        state_path = os.path.join(self.log_dir, 'logger_state.npy')
        if os.path.exists(state_path):
            state = np.load(state_path, allow_pickle=True).item()
            self.episode = state.get("episode", 0)
            self.logging_step = state.get("logging_step", 0)
            self.objectives = deque(state.get("objectives", []), maxlen=100)
            self.best_metrics = state.get("best_metrics", {})
            self.train_step = state.get("train_step", 0)
            self.policy_updates = state.get("policy_updates", 0)

    def save_state(self):
        state = {
            "episode": self.episode,
            "logging_step": self.logging_step,
            "objectives": list(self.objectives),
            "best_metrics": self.best_metrics,
            "train_step": self.train_step,
            "policy_updates": self.policy_updates
        }
        state_path = os.path.join(self.log_dir, 'logger_state.npy')
        np.save(state_path, state)

    def calc_current_objective(self):
        if len(self.objectives) == self.objectives.maxlen:
            self.best_metrics["current_objective"] = float(np.mean(self.objectives))

    def get_best_objective(self):
        if self.best_metrics["current_objective"] > self.best_metrics["best_objective"]:
            return True, self.best_metrics["current_objective"]
        else:
            return False, self.best_metrics["best_objective"]

    def is_better_objective(self):
        """Checks if the current objective is better than the best recorded objective.
           Should only be called when checking if the Model should be saved.
           Otherwise, use `calc_current_objective` to check for the best objective.
        """
        is_better, self.best_metrics["best_objective"] = self.get_best_objective()
        return is_better

    def is_better_reward(self):
        """Checks if the current reward is better than the best recorded reward.
           Should only be called when checking if the Model should be saved.
        """
        current_reward = np.mean(self.metrics["reward"]) if "reward" in self.metrics else 0.0
        if current_reward >= self.best_metrics["best_reward"]:
            self.best_metrics["best_reward"] = current_reward
            return True
        else:
            return False

    def close(self):
        self.writer.close()