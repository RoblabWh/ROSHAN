import os
from algorithms.rl_config import RLConfig
import logging

class RLAlgorithm:
    """
    Base class for Reinforcement Learning algorithms.
    """

    def __init__(self, config: RLConfig):

        for field in config.__dataclass_fields__:
            setattr(self, field, getattr(config, field))

        self.version = 1
        self.logger = logging.getLogger(self.algorithm)

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

    def reset(self):
        """
        Reset the algorithm to its initial state.
        This method should be overridden by subclasses if needed.
        """
        self.version += 1
        self.reset_algorithm()

    def reset_algorithm(self):
        pass

    def set_train(self):
        pass

    def set_eval(self):
        pass

    def load(self):
        pass