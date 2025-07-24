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
    loading_path: str
    loading_name: str
    use_auto_train: bool # Used for AutoTrain

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
    lr: float = 4e-4 #2.45e-05 #3e-4
    batch_size: int = 32 #1024 #500

@dataclass
class NoAlgorithmConfig(RLConfig):
    """
    Configuration class for No Algorithm (NoOp) scenario.
    This is used when no specific RL algorithm is applied.
    """
    # No specific parameters for No Algorithm
    pass

@dataclass
class PPOConfig(RLConfig):
    """
    Configuration class for Proximal Policy Optimization (PPO) algorithm.
    """
    # PPO specific parameters
    horizon: int = 64 # 12800 for FlyNetwork
    k_epochs: int = 14 #14 # 4
    entropy_coeff: float = 0.0006 #0.001
    value_loss_coef: int = 0.5
    separate_optimizers: bool = False
    beta1: float = 0.9
    beta2: float = 0.999
    gamma: float = 0.9635 #0.99
    _lambda: float = 0.96
    eps_clip: float = 0.2783 #0.15 #0.2
    use_categorical: bool = False
    use_variable_state_masks: bool = False
#Current Best: {'lr': 2.450621398427006e-05, 'gamma': 0.9635130394855251, 'clip_range': 0.2783066147446984, 'ent_coef': 0.0006194758615931694, 'k_epochs': 6, 'batch_size': 1024}
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
    beta1: float = 0.9
    beta2: float = 0.999

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
    beta1: float = 0.9
    beta2: float = 0.999

