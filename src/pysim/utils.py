import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


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
    if out_type == 'actor':
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif out_type == 'critic':
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


def initialize_hidden_weights(m):
    """
    Initialize the weights of the hidden layers of the actor and critic networks
    :param m: the layer to initialize
    """
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0)


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


class Logger(object):
    """
    Logger class for logging training and evaluation metrics. It uses tensorboardX to log the metrics.

    :param log_dir: (string) directory where the logs will be saved
    """
    def __init__(self, log_dir, horizon):
        self.writer = None
        self.log_dir = log_dir
        self.logging = False
        self.total_steps = horizon
        self.horizon = horizon

        # Agent Observations
        self.velocities = []
        self.positions = []

        # Algorithm metrics
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []

        # Agent metrics
        self.reward = 0
        self.value = 0

        # Episodic metrics and calculation
        self.objective = []
        self.current_steps = 0
        self.agent_steps = []

        # Flags
        self.reward_best = -99999
        self.episode_finished = False

    def __del__(self):
        if self.logging:
            self.close()

    def set_logging(self, logging):
        if logging:
            self.writer = SummaryWriter(self.log_dir)
        self.logging = logging

    def episode_log(self, observations, done, burned_percentage):
        self.current_steps += 1
        self.velocities.append(observations[2].squeeze()[0])
        self.positions.append(observations[3].squeeze()[0])
        if done:
            self.add_objective(burned_percentage)
            self.agent_steps.append(self.current_steps)
            self.episode_finished = True

    def add_reward(self, rewards):
        # TODO implement reward dict
        self.reward = rewards

    def add_loss(self, loss, entropy, critic_loss, actor_loss):
        self.loss.append(loss)
        self.entropy.append(entropy)
        self.critic_loss.append(critic_loss)
        self.actor_loss.append(actor_loss)

    def add_value(self, values):
        self.value = values

    def add_objective(self, percent_burned):
        self.objective.append(percent_burned)

    def get_objective(self):
        return np.mean(self.objective) if len(self.objective) > 0 else 0

    def summary_loss(self):
        if self.logging:
            self.writer.add_scalars('loss', {'loss': np.mean(self.loss),
                                            'entropy': np.mean(self.entropy),
                                            'critic_loss': np.mean(self.critic_loss),
                                            'actor loss': np.mean(self.actor_loss)}, self.total_steps)

    def summary_objective(self):
        if self.logging:
            self.writer.add_scalar('Percentage Burned', np.mean(self.objective), self.total_steps, new_style=True)

    def summary_reward(self):
        if self.logging:
            self.writer.add_scalar('Reward (AVG)', np.mean(self.reward), self.total_steps, new_style=True)
            self.writer.add_histogram('Reward (HIST)', np.array(self.reward), self.total_steps)

    def summary_value(self):
        if self.logging:
            self.writer.add_scalar('Value (AVG)', np.mean(self.value), self.total_steps, new_style=True)
            self.writer.add_histogram('Value (HIST)', np.array(self.value), self.total_steps)

    def summary_steps_agents(self):
        if self.logging and self.episode_finished:
            self.writer.add_histogram('Steps (HIST)', np.array(self.agent_steps), self.total_steps)

    def summary_velocities(self):
        if self.logging:
            self.writer.add_histogram('Velocities (HIST)', np.array(self.velocities), self.total_steps)

    def summary_positions(self):
        if self.logging:
            self.writer.add_histogram('Positions (HIST)', np.array(self.positions), self.total_steps)

    def better_reward(self):
        if np.mean(self.reward) > self.reward_best:
            self.reward_best = np.mean(self.reward)
            return True
        else:
            return False

    def log(self):
        if self.episode_finished:
            self.summary_objective()
            self.summary_steps_agents()
        self.summary_velocities()
        self.summary_positions()
        self.summary_reward()
        self.summary_value()
        self.total_steps += self.horizon
        self.clear_summary()

    def reset_horizon(self):
        self.total_steps = 0


    def clear_summary(self):
        if self.episode_finished:
            self.objective = []
            self.current_steps = 0
            self.agent_steps = []
            self.episode_finished = False
        self.velocities = []
        self.positions = []
        self.loss = []
        self.critic_loss = []
        self.actor_loss = []
        self.reward = 0
        self.value = 0

    def close(self):
        self.writer.close()
