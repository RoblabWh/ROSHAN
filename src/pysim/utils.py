import numpy as np
import torch
from pathlib import Path
from collections import deque
from torch.utils.tensorboard import SummaryWriter


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
    :param horizon: The number of steps per episode or training horizon.
    """

    def __init__(self, log_dir, horizon):
        self.current_steps = None
        self.episode_finished = None
        self.writer = SummaryWriter(log_dir)
        self.log_dir = log_dir
        self.tag = 'run0'
        self.total_steps = 0  # Total steps across all epochs
        self.summarize_steps = 0
        self.horizon = horizon

        # Initialize storage for metrics
        self.reset_metrics()

        # Best reward tracking
        self.current_episode = 0
        self.best_reward = float('-inf')
        self.objectives = deque(maxlen=100)
        self.best_objective = float('-inf')
        self.current_objective = 0

        # Episodic metrics
        self.agent_steps = []

    def reset_metrics(self):
        """Resets all metrics at the end of an epoch or training cycle."""
        # Agent Observations
        self.velocities = []
        self.positions = []

        # Algorithm metrics
        self.entropies = []
        self.critic_losses = []
        self.actor_losses = []
        self.std_xs = []
        self.std_ys = []

        # Agent metrics
        self.rewards = []
        self.rewards_scaled = []
        self.returns = []
        self.values = []

        # Explained variance
        self.explained_variances = []

        # Network Metrics
        self.filters = []

        # Flags
        self.current_steps = 0
        self.episode_finished = False

    def log_episode(self, terminal_result):
        """Logs metrics at each step and handles episode completion."""
        self.current_steps += 1
        env_done = terminal_result["EnvReset"]
        
        if env_done:
            self.current_episode += 1
            objective_reached = not terminal_result["OneAgentDied"]
            self.objectives.append(objective_reached)
            self.agent_steps.append(self.current_steps)
            self.episode_finished = True
            self.current_steps = 0  # Reset steps for next episode

    def calc_objective_percentage(self):
        if len(self.objectives) == self.objectives.maxlen:
            self.current_objective = np.mean(self.objectives)

    def get_best_objective(self):
        if self.current_objective > self.best_objective:
            return True, self.current_objective
        else:
            return False, self.best_objective

    def add_filters(self, filters):
        """Adds the number of filters in each layer."""
        for idx in filters.shape[0]:
            self.filters.append(filters[idx])

    def add_std(self, std):
        """Adds standard deviation metrics."""
        self.std_xs.append(std[0])
        self.std_ys.append(std[1])

    def add_rewards(self, rewards):
        """Adds rewards."""
        self.rewards.extend(rewards)  # Assuming rewards is a list or array

    def add_rewards_scaled(self, rewards_scaled):
        """Adds scaled rewards."""
        self.rewards_scaled.extend(rewards_scaled)

    def add_returns(self, returns):
        """Adds returns."""
        self.returns.extend(returns)

    def add_values(self, values):
        """Adds value function predictions."""
        self.values.extend(values)  # Assuming values is a list or array

    def add_losses(self, critic_loss, actor_loss, entropy=0):
        """Adds loss metrics."""
        # self.entropies.extend(entropy)
        self.critic_losses.append(critic_loss)
        self.actor_losses.append(actor_loss)

    def add_explained_variance(self, explained_variance):
        """Adds explained variance metric."""
        self.explained_variances.append(explained_variance)

    def summarize_metrics(self, status):
        """Logs all metrics to TensorBoard."""
        t = self.summarize_steps
        if self.episode_finished:
            self.writer.add_histogram(f'{self.tag}/Agent Steps', np.array(self.agent_steps), t)
            self.agent_steps.clear()

        # Log scalar metrics
        self.writer.add_scalar(f'{self.tag}/Critic Loss', np.mean(self.critic_losses), t)
        self.writer.add_scalar(f'{self.tag}/Actor Loss', np.mean(self.actor_losses), t)
        # self.writer.add_scalar(f'{self.tag}/Entropy', np.mean(self.entropies), t)
        self.writer.add_scalar(f'{self.tag}/Reward (Avg)', np.mean(self.rewards), t)
        self.writer.add_scalar(f'{self.tag}/Reward Scaled (Avg)', np.mean(self.rewards_scaled), t)
        self.writer.add_scalar(f'{self.tag}/Return (Avg)', np.mean(self.returns), t)
        self.writer.add_scalar(f'{self.tag}/Value (Avg)', np.mean(self.values), t)
        self.writer.add_scalar(f'{self.tag}/Std X (Avg)', np.mean(self.std_xs), t)
        self.writer.add_scalar(f'{self.tag}/Std Y (Avg)', np.mean(self.std_ys), t)
        self.writer.add_scalar(f'{self.tag}/Explained Variance (AVG)', np.mean(self.explained_variances), t)

        # Log histogram metrics
        if self.rewards: self.writer.add_histogram(f'{self.tag}/Rewards', np.array(self.rewards), t)
        # self.writer.add_histogram(f'{self.tag}/Rewards Scaled', np.array(self.rewards_scaled), t)
        if self.returns: self.writer.add_histogram(f'{self.tag}/Returns', np.array(self.returns), t)
        if self.values: self.writer.add_histogram(f'{self.tag}/Values', np.array(self.values), t)
        if self.std_xs: self.writer.add_histogram(f'{self.tag}/Std X', np.array(self.std_xs), t)
        if self.std_ys: self.writer.add_histogram(f'{self.tag}/Std Y', np.array(self.std_ys), t)
        # self.writer.add_histogram(f'{self.tag}/Velocities', np.array(self.velocities), t)
        # self.writer.add_histogram(f'{self.tag}/Positions', np.array(self.positions), t)
        if self.explained_variances: self.writer.add_histogram(f'{self.tag}/Explained Variances', np.array(self.explained_variances), t)

        self.summarize_steps += 1
        self.tag = "run" + str(status["train_episode"])

        # Clear metrics after logging
        self.reset_metrics()

    def is_better_reward(self):
        """Checks if the current reward is better than the best recorded reward."""
        current_reward = np.mean(self.rewards)
        if current_reward >= self.best_reward:
            self.best_reward = current_reward
            return True
        else:
            return False

    def is_better_objective(self):
        """Checks if the current objective is better than the best recorded objective."""
        is_better, self.best_objective = self.get_best_objective()
        return is_better

    def close(self):
        """Closes the SummaryWriter."""
        self.writer.close()
