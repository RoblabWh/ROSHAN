
class Agent:
    def __init__(self):
        self.hierarchy_level = "none"

    @staticmethod
    def get_num_agents(num_agents):
        return num_agents

    def get_hierarchy_level(self):
        return self.hierarchy_level

    @staticmethod
    def get_module_names(algorithm_name: str):
        if algorithm_name == "PPO":
            return "StochasticActor", "CriticPPO"
        elif algorithm_name == "IQL":
            return "StochasticActor", "OffPolicyCritic", "Value"
        elif algorithm_name == "TD3":
            return "DeterministicActor", "OffPolicyCritic"
        elif algorithm_name == "no_algo":
            return None, None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")