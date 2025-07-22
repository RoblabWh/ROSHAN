import os
from algorithms.rl_config import RLConfig

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """

    def __init__(self, config: RLConfig):

        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.version = 1

    def set_paths(self, model_path, model_name):
        self.model_path = os.path.abspath(model_path)
        self.model_name = model_name

    def get_model_name_reward(self, version=False):
        reward_str = "_best_reward_v" + str(self.version) if version else "_best_reward"
        return self.model_name.split(".")[0] + reward_str + "." + self.model_name.split(".")[1]

    def get_model_name_obj(self, version=False):
        obj_str = "_best_obj_v" + str(self.version) if version else "_best_obj"
        return self.model_name.split(".")[0] + obj_str + "." + self.model_name.split(".")[1]

    def get_model_name_latest(self):
        return self.model_name.split(".")[0] + "_latest." + self.model_name.split(".")[1]

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def load(self):
        pass