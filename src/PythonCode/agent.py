from ppo import PPO
from td3 import TD3


class Agent:
    def __init__(self, algorithm: str = 'ppo', logger=None, action_dim=(1, 3), model_path="../models/"):
        self.algorithm_name = algorithm
        self.logger = logger
        if algorithm == 'ppo':
            self.algorithm = PPO(vision_range=21, lr=0.00003, betas=(0.9, 0.999), gamma=0.99, _lambda=0.9, K_epochs=4, eps_clip=0.2, model_path=model_path)
            print("PPO agent initialized")
        elif algorithm == 'td3':
            # Discontinoued for now do not use
            self.algorithm = TD3(action_dim=action_dim, lr=0.00003, vision_range=21, discount=0.99, tau=0.005, policy_noise=0.2, noise_clip=0.5, policy_freq=2)
            print("TD3 agent initialized")

    def should_train(self, memory, horizon, t):
        if self.algorithm_name == 'ppo':
            return memory.size > horizon
        elif self.algorithm_name == 'td3':
            if t >= 10000:
                return t % horizon == 0
            else:
                return False

    def load_model(self, path):
        # Load model, return True if successful and set model to evaluation mode
        if self.algorithm.load_model(path):
            self.algorithm.set_eval()

    def update(self, memory, batch_size, mini_batch_size):
        self.algorithm.update(memory, batch_size, mini_batch_size, self.logger)

    def act(self, observations, t):
        if self.algorithm_name == 'td3':
            if t < 10000:
                return self.algorithm.random_action()

        return self.algorithm.select_action(observations)

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)

