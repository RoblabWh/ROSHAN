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
    use_next_obs: bool = True

    # Learning Parameters
    lr: float = 3e-4
    batch_size: int = 500

@dataclass
class PPOConfig(RLConfig):
    """
    Configuration class for Proximal Policy Optimization (PPO) algorithm.
    """
    # PPO specific parameters
    horizon: int = 12800 # 12800 for FlyNetwork
    k_epochs: int = 14 # 4
    entropy_coeff: float = 0.001
    value_loss_coef: int = 0.5
    separate_optimizers: bool = False
    betas: tuple[int, int] = (0.9, 0.999)
    gamma: float = 0.99
    _lambda: float = 0.96
    eps_clip: float = 0.15 #0.2

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
    min_memory_size: int = 1000
    policy_freq: int = 10
    k_epochs: int = 20
    betas: tuple[int, int] = (0.9, 0.999)

@dataclass
class TD3Config(RLConfig):
    """
    Configuration class for Twin Delayed Deep Deterministic Policy Gradient (TD3) algorithm.
    """
    # TD3 specific parameters
    min_memory_size: int = 25000
    k_epochs: int = 20 # 1 to 50 sometimes 100+
    discount: float = 0.99
    tau: float = 0.005
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    betas: tuple[int, int] = (0.9, 0.999)

