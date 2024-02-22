import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


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


def normalize(tensor):
    """
    Normalizes a tensor to mean zero and standard deviation one
    """
    return (tensor - tensor.mean()) / (tensor.std() + 1e-8)


def torchToNumpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


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
    :param log_interval: (int) interval for logging
    """
    def __init__(self, log_dir, log_interval):
        self.writer = None
        self.log_dir = log_dir
        self.log_interval = log_interval
        self.logging = False
        self.episode = 0
        self.last_logging_episode = 0
        # loss
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []

        self.actor_mean_linvel = []
        self.actor_mean_angvel = []
        self.actor_var_linvel = []
        self.actor_var_angvel = []

        self.reward = []
        self.value = []

        #objective
        self.reward_best = -99999
        self.objective_reached = 0
        self.number_of_agents = 0
        self.steps_agents = 0

    def __del__(self):
        if self.logging:
            self.close()

    def set_logging(self, logging):
        if logging:
            self.writer = SummaryWriter(self.log_dir)
        elif self.logging:
            self.close()
        self.logging = logging

    def build_graph(self, model, device):
        if self.logging:
            laser = torch.rand(4, 4, 1081).to(device)
            ori = torch.rand(4, 4, 2).to(device)
            dist = torch.rand(4, 4).to(device)
            vel = torch.rand(4, 4, 2).to(device)
            self.writer.add_graph(model, (laser, ori, dist, vel))

    def add_loss(self, loss, entropy, critic_loss, actor_loss):
        self.loss.append(loss)
        self.entropy.append(entropy)
        self.critic_loss.append(critic_loss)
        self.actor_loss.append(actor_loss)

    def summary_loss(self):
        if self.episode > self.last_logging_episode:
            if self.logging and not len(self.loss) == 0:
                self.writer.add_scalars('loss', {'loss': np.mean(self.loss),
                                                'entropy': np.mean(self.entropy),
                                                'critic_loss': np.mean(self.critic_loss),
                                                'actor loss': np.mean(self.actor_loss)}, self.episode)

    def add_step_agents(self, steps_agents):
        self.steps_agents += steps_agents

    def add_actor_output(self, actor_mean_linvel, actor_mean_angvel, actor_var_linvel, actor_var_angvel):
        self.actor_mean_linvel.append(actor_mean_linvel)
        self.actor_mean_angvel.append(actor_mean_angvel)
        self.actor_var_linvel.append(actor_var_linvel)
        self.actor_var_angvel.append(actor_var_angvel)

    def summary_actor_output(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalars('actor_output', {'Mean LinVel': np.mean(self.actor_mean_linvel),
                                                     'Mean AngVel': np.mean(self.actor_mean_angvel),
                                                     'Variance LinVel': np.mean(self.actor_var_linvel),
                                                     'Variance AngVel': np.mean(self.actor_var_angvel)}, self.episode)

    def summary_objective(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalar('objective reached', self.percentage_objective_reached(), self.episode)

    # def add_reward(self, rewards):
    #     for reward in rewards:
    #         for key in reward.keys():
    #             if key in self.reward.keys():
    #                 self.reward[key] += reward[key]
    #             else:
    #                 self.reward[key] = reward[key]
    def add_reward(self, rewards):
        for reward in rewards:
            self.reward.append(reward)

    def add_value(self, values):
        for value in values:
            self.value.append(value)

    def percentage_objective_reached(self):
        return self.objective_reached / (self.episode - self.last_logging_episode)

    def add_objective(self, reachedGoals):
        self.objective_reached += (np.count_nonzero(reachedGoals) / self.number_of_agents)

    def set_number_of_agents(self, number_of_agents):
        self.number_of_agents = number_of_agents

    # def summary_reward(self):
    #     if self.logging and self.episode > self.last_logging_episode:
    #         self.reward['total'] = 0
    #         for key in self.reward.keys():
    #             if key != 'total':
    #                 reward_per_step = self.reward[key] / self.steps_agents
    #                 self.reward[key] = reward_per_step
    #                 self.reward['total'] += reward_per_step
    #         self.writer.add_scalars('reward', self.reward, self.episode)

    def summary_reward(self):
        if self.logging:
            self.writer.add_scalar('reward', np.mean(self.reward), self.episode)

    def summary_value(self):
        if self.logging:
            self.writer.add_scalar('value', np.mean(self.value), self.episode)

    def summary_steps_agents(self):
        if self.logging and self.episode > self.last_logging_episode:
            self.writer.add_scalar('avg steps per agent', self.steps_agents / self.number_of_agents, self.episode)

    def better_reward(self):
        if np.mean(self.reward) > self.reward_best:
            self.reward_best = np.mean(self.reward)
            return True
        else:
            return False

    def log(self):

        # self.summary_reward()
        # objective_reached = self.percentage_objective_reached()
        # self.summary_objective()
        # self.summary_steps_agents()
        # self.summary_actor_output()
        # self.summary_loss()
        #
        # self.last_logging_episode = self.episode
        # self.clear_summary()
        # return sum([v for v in self.reward.values()]), objective_reached
        self.summary_reward()
        self.summary_value()
        self.episode += 1
        self.clear_summary()

    def clear_summary(self):
        self.actor_mean_linvel = []
        self.actor_mean_angvel = []
        self.actor_var_linvel = []
        self.actor_var_angvel = []
        self.loss = []
        self.entropy = []
        self.critic_loss = []
        self.actor_loss = []
        self.objective_reached = 0
        self.steps_agents = 0
        self.reward = []
        self.cnt_agents = 0

    def close(self):
        self.writer.close()
