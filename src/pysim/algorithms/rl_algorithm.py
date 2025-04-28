import os
from algorithms.rl_config import RLConfig

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """

    def __init__(self, config: RLConfig):
        self.model_path = os.path.abspath(config.model_path)
        self.model_name = config.model_name
        self.version = 1
        self.model_version_reward = self.model_name.split(".")[0] + "_reward_v" + str(self.version) + "." + self.model_name.split(".")[1]
        self.model_version_obj = self.model_name.split(".")[0] + "_obj_v" + str(self.version) + "." + self.model_name.split(".")[1]
        self.model_latest = self.model_name.split(".")[0] + "_latest." + self.model_name.split(".")[1]

    def set_paths(self, model_path, model_name):
        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name
        self.model_version_reward = model_name.split(".")[0] + "_reward_v" + str(self.version) + "." + model_name.split(".")[1]
        self.model_version_obj = model_name.split(".")[0] + "_obj_v" + str(self.version) + "." + model_name.split(".")[1]
        self.model_latest = model_name.split(".")[0] + "_latest." + model_name.split(".")[1]