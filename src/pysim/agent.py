from ppo import PPO
import firesim
import numpy as np
import os


class Agent:
    def __init__(self, algorithm: str = 'ppo', logger=None, model_path="../models/", model_name="model_best.pth", vision_range=21, time_steps=4):
        self.algorithm_name = algorithm
        self.logger = logger
        if algorithm == 'ppo':
            self.algorithm = PPO(vision_range=vision_range, time_steps=time_steps, lr=0.00003, betas=(0.9, 0.999), gamma=0.99, _lambda=0.9, K_epochs=4, eps_clip=0.2, model_path=model_path, model_name=model_name)
            print("PPO agent initialized")

    def should_train(self, memory, horizon, t):
        if self.algorithm_name == 'ppo':
            return memory.size > horizon

    def load_model(self, resume=False, train=False):
        # Load model, return True if successful and set model to evaluation mode
        if not train:
            if self.algorithm.load():
                self.algorithm.set_eval()
                return f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to evaluation mode\n"
            else:
                self.algorithm.set_train()
                return "No checkpoint found to evaluate model, start training from scratch\n"
        elif resume:
            if self.algorithm.load():
                self.algorithm.set_train()
                return f"Load model from checkpoint {os.path.join(self.algorithm.model_path, self.algorithm.model_name)}\n" \
                       f"Model set to training mode\n"
            else:
                self.algorithm.set_train()
                return "No checkpoint found to resume training, start training from scratch\n"
        else:
            self.algorithm.set_train()
            return "Training from scratch\n"

    def set_paths(self, model_path, model_name):
        self.algorithm.set_paths(model_path, model_name)

    def update(self, memory, batch_size, mini_batch_size, next_obs, next_terminals):
        return self.algorithm.update(memory, batch_size, mini_batch_size, next_obs, next_terminals, self.logger)

    def act(self, observations):
        actions, action_logprobs = self.algorithm.select_action(observations)
        return actions, action_logprobs

    def get_action(self, actions):
        drone_actions = []
        for activation in actions:
            drone_actions.append(
                firesim.DroneAction(activation[0], activation[1], int(np.round(activation[2]))))
        return drone_actions

    def act_certain(self, observations):
        return self.algorithm.select_action_certain(observations)


