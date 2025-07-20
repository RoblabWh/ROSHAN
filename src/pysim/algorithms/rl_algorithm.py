import os
from algorithms.rl_config import RLConfig

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """

    def __init__(self, config: RLConfig):
        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))
        self.model_path = os.path.abspath("./models") if config.model_path is None or config.model_path == "" else self.model_path
        self.model_name = "model.pt" if config.model_name is None or config.model_name == "" else config.model_name
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

    def set_train(self):
        pass

    def set_eval(self):
        pass