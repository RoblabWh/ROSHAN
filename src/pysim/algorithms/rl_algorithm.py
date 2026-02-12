import os
import torch
import warnings
from concurrent.futures import ThreadPoolExecutor
from algorithms.rl_config import RLConfig
from utils import RunningMeanStd
from evaluation import TensorboardLogger
import torch.nn as nn
import logging

# Shared background thread for non-blocking torch.save() calls
_save_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model_saver")

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """

    def __init__(self, config: RLConfig):

        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.version = 1
        self.logger = logging.getLogger(self.algorithm)
        self.reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        self.MSE_loss = nn.MSELoss()
        self.use_noised_action = False

    def set_paths(self, model_path, model_name):
        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name

    def get_model_path(self):
        """
        This method returns the absolute path to the CURRENT model_path.
        Will return self.model_path normally, but if auto_train is enabled,
        it will return the model_paths directory for the version.
        """
        if self.use_auto_train:
            return os.path.join(self.model_path, "training_" + str(self.version))
        else:
            return self.model_path

    def get_model_name_reward(self):
        reward_str = "_best_reward"
        return self.model_name.split(".")[0] + reward_str + "." + self.model_name.split(".")[1]

    def get_model_name_obj(self):
        obj_str = "_best_obj"
        return self.model_name.split(".")[0] + obj_str + "." + self.model_name.split(".")[1]

    def get_model_name_latest(self):
        return self.model_name.split(".")[0] + "_latest." + self.model_name.split(".")[1]

    def initialize_policy(self):
        """
        Initialize the policy network.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError

    def initialize_optimizers(self):
        """
        Initialize the optimizers for the policy network.
        This method should be overridden by subclasses.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the algorithm to its initial state.
        This method should be overridden by subclasses if needed.
        """
        self.version += 1
        self.initialize_policy()
        self.initialize_optimizers()
        self.MSE_loss = nn.MSELoss()
        self.reward_rms = RunningMeanStd()
        self.int_reward_rms = RunningMeanStd()
        self.set_train()

    def save_optimizers(self, path: str):
        raise NotImplementedError

    def load_optimizers(self, path: str):
        raise NotImplementedError

    def copy_networks(self):
        pass

    def save(self, logger: TensorboardLogger):
        # Snapshot state_dict on the main thread (fast, in-memory copy),
        # then write to disk on a background thread to avoid blocking training.
        def _bg_save(state_dict, path, opt_path):
            torch.save(state_dict, path)
            self.save_optimizers(opt_path)

        if logger.is_better_reward():
            self.logger.info(f"Saving at Episode {logger.episode}/Train Step {logger.train_step}, best Reward: {logger.best_metrics['best_reward']:.2f}")
            path = f'{os.path.join(self.get_model_path(), self.get_model_name_reward())}'
            sd = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
            _save_executor.submit(_bg_save, sd, path, path.split('.')[-2])
        if logger.is_better_objective():
            self.logger.info(f"Saving at Episode {logger.episode}/Train Step {logger.train_step}, best Objective {logger.best_metrics['best_objective']:.2f}")
            path = f'{os.path.join(self.get_model_path(), self.get_model_name_obj())}'
            sd = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
            _save_executor.submit(_bg_save, sd, path, path.split('.')[-2])
        path = f'{os.path.join(self.get_model_path(), self.get_model_name_latest())}'
        sd = {k: v.cpu().clone() for k, v in self.policy.state_dict().items()}
        _save_executor.submit(_bg_save, sd, path, path.split('.')[-2])

    def load(self):
        path: str = os.path.join(self.loading_path, self.loading_name).__str__()
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
                self.policy.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
            self.load_optimizers(path.split('.')[-2])
            self.policy.to(self.device)
            self.copy_networks()
            return True
        except FileNotFoundError:
            self.logger.warning(f"Could not load model from {path}. Falling back to train mode.")
            return False
        except TypeError:
            self.logger.warning(f"TypeError while loading model.")
            return False
        except RuntimeError:
            self.logger.warning(f"RuntimeError while loading model. Possibly due to architecture mismatch.")
            return False

    def select_action(self, observations):
        return self.policy.act(observations)

    def select_action_certain(self, observations):
        """
        Select action with certain policy.
        """
        return self.policy.act_certain(observations)

    def prepare_rewards(self, rewards: torch.FloatTensor, t_dict: dict, already_fit=False):
        # Check for intrinsic rewards
        intrinsic_rewards = None
        int_rewards_np = None
        if 'intrinsic_reward' in t_dict.keys():
            intrinsic_rewards = t_dict['intrinsic_reward']
            int_rewards_np = torch.cat(intrinsic_rewards).detach().cpu().numpy()
            if not already_fit:
                self.int_reward_rms.update(int_rewards_np)

        # Compute all raw rewards for logging
        ext_rewards_np = torch.cat(rewards).detach().cpu().numpy()
        log_rewards_raw = ext_rewards_np if int_rewards_np is None else ext_rewards_np + int_rewards_np

        # DON'T DO THIS: rewards = [np.clip(np.array(reward.detach().cpu()) / self.running_reward_std.get_std(), -10, 10) for reward in rewards]
        # !!!!DON'T SHIFT THE REWARDS BECAUSE YOU F UP YOUR OBJECTIVE FUNCTION!!!!; Clipping most likely is unnecessary
        # Normalize external rewards by reward running std
        if not already_fit:
            self.reward_rms.update(ext_rewards_np)
            ext_rewards = [reward / self.reward_rms.get_std() for reward in rewards]

            # Normalize for logging
            ext_rewards_norm = ext_rewards_np / self.reward_rms.get_std()
        else:
            ext_rewards = [reward / self.reward_rms.get_std() for reward in rewards]
            ext_rewards_norm = ext_rewards_np / self.reward_rms.get_std()

        # Combine normalized rewards
        if intrinsic_rewards is not None:
            intrinsic_rewards = [reward / self.int_reward_rms.get_std() for reward in intrinsic_rewards]
            intrinsic_rewards_norm = int_rewards_np / self.int_reward_rms.get_std()
            log_rewards_scaled = ext_rewards_norm + intrinsic_rewards_norm
        else:
            log_rewards_scaled = ext_rewards_norm

        rewards_total = ext_rewards if intrinsic_rewards is None \
            else [ext_reward + int_reward for ext_reward, int_reward in zip(ext_rewards, intrinsic_rewards)]

        return rewards_total, log_rewards_raw, log_rewards_scaled

    def set_eval(self):
        self.policy.eval()

    def set_train(self):
        self.policy.train()

    def apply_manual_decay(self, train_step: int):
        pass