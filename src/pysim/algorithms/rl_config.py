from dataclasses import dataclass

@dataclass
class RLConfig:
    """
    Configuration class for Reinforcement Learning (RL) algorithms.
    """

    # Algorithm configuration
    algorithm: str

    # Environment configuration
    model_path: str
    model_name: str

    # For my implementation
    vision_range: int
    drone_count: int
    map_size: int
    time_steps: int

    # Memory configuration
    memory_size: int = 1e6
    action_dim: int  = 2
    clear_memory: bool = True

    # Learning Parameters
    lr: float = 3e-4

@dataclass
class PPOConfig(RLConfig):
    """
    Configuration class for Proximal Policy Optimization (PPO) algorithm.
    """
    # PPO specific parameters
    k_epochs: int = 4
    entropy_coeff: float = 0.01
    value_loss_coef: int = 0.5
    separate_optimizers: bool = False
    betas: tuple[int, int] = (0.9, 0.999)
    gamma: float = 0.99
    _lambda: float = 0.96
    eps_clip: float = 0.01

@dataclass
class IQLConfig(RLConfig):
    """
    Configuration class for IQL algorithm.
    """
    # IQL specific parameters
    discount: float = 0.99
    expectile: float = 0.7
    tau: float = 0.005
    temperature: float = 3.0
    action_dim = 2
    min_memory_size: int = 1000
    n_steps: int = 10
    k_epochs: int = 20
    betas: tuple[int, int] = (0.9, 0.999)


