from ppo import PPO
import firesim
import numpy as np
import os


class Agent:
    def __init__(self, status, algorithm: str = 'ppo', logger=None, model_path="../models/", model_name="model_best.pth", vision_range=21, time_steps=4):
        self.algorithm_name = algorithm
        self.eval_steps = 0
        self.max_eval = status["max_eval"]
        self.horizon = status["horizon"]
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        self.logger = logger
        self.initialized = False
        if algorithm == 'ppo':
            self.algorithm = PPO(vision_range=vision_range, time_steps=time_steps, lr=0.00003, betas=(0.9, 0.999), gamma=0.99, _lambda=0.9, K_epochs=4, eps_clip=0.2, model_path=model_path, model_name=model_name)
            self.initialized = True
            print("PPO agent initialized")

    def reset(self):
        self.stats = {'died': [0], 'reached': [0], 'time': [0], 'reward': [0], 'episode': [0], 'perc_burn': [0]}
        self.initialized = True

    def should_train(self, memory):
        if self.algorithm_name == 'ppo':
            return len(memory) >= self.horizon

    def load_model(self, resume=False, train_="train"):
        train = True if train_ == "train" else False
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

    def check_horizon(self, new_horizon, memory):
        if new_horizon != self.horizon:
            self.horizon = new_horizon
            self.logger.episode_finished = True
            self.logger.horizon = self.horizon
            self.logger.clear_summary()
            memory.change_horizon(self.horizon)
        return memory

    def update_status(self, status, memory):
        self.set_paths(status["model_path"], status["model_name"])
        memory = self.check_horizon(status["horizon"], memory)
        return memory

    def update(self, status, memory, mini_batch_size, next_obs, next_terminals):
        status["console"] += self.algorithm.update(memory, self.horizon, mini_batch_size, next_obs, next_terminals, self.logger)
        status["train_step"] += 1
        if status["train_step"] >= status["max_train"]:
            status["console"] += "Training finished, after {} training steps".format(status["train_step"])
            status["agent_online"] = False
            status["console"] += "Agent is offline\n"

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

    def log_episode(self, status, rewards, terminals, dones, percent_burned):
        episode_over, log = self.episode_logging(rewards, terminals, dones, percent_burned)
        status["console"] += log
        if episode_over:
            self.eval_steps += 1
            if self.eval_steps >= status["max_eval"]:
                status["agent_online"] = False
                status["console"] += "Evaluation finished, agent is offline\n" \
                                     "If you wish to start anew, reset the agent\n"

    def episode_logging(self, rewards, terminals, dones, percent_burned):
        # Log stats
        self.stats['reward'][-1] += rewards[0]
        self.stats['time'][-1] += 1
        console = ""

        if terminals[0]:
            console += "Episode: {} finished with terminal state\n".format(self.stats['episode'][-1] + 1)
            if dones[0]:  # Drone died
                console += "Drone died.\n\n"
                self.stats['died'][-1] += 1
            else:  # Drone reached goal
                console += "Drone extinguished all fires.\n\n"
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

            # Reset stats
            self.stats['episode'].append(self.stats['episode'][-1] + 1)
            self.stats['died'].append(0)
            self.stats['perc_burn'].append(0)
            self.stats['reached'].append(0)
            self.stats['time'].append(0)
            self.stats['reward'].append(0)

            return True, console
        return False, console


