from networks.network_fly import StochasticActor, CriticPPO, OffPolicyCritic, DeterministicActor, Value
import firesim
from agent import Agent

class FlyAgent(Agent):
    def __init__(self, name):
        super().__init__()
        self.name = name
        self.short_name = "fly"
        self.hierarchy_level = "low"
        self.use_intrinsic_reward = False
        self.action_dim = 2

    @staticmethod
    def get_network(algorithm : str):
        if algorithm == "PPO":
            return StochasticActor, CriticPPO
        elif algorithm == "IQL":
            return StochasticActor, OffPolicyCritic, Value
        elif algorithm == "TD3":
            return DeterministicActor, OffPolicyCritic
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")


    @staticmethod
    def get_action(actions):
        return [firesim.DroneAction(a[0], a[1]) for a in actions]

