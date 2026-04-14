import inspect
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import logging

logger = logging.getLogger("OptimizerFactory")

OPTIMIZER_REGISTRY = {
    "Adam": optim.Adam,
    "AdamW": optim.AdamW,
    "SGD": optim.SGD,
    "RMSprop": optim.RMSprop,
    "RAdam": optim.RAdam,
    "Muon": optim.Muon,  # torch >= 2.9; only supports 2D params (weight matrices)
}

SCHEDULER_REGISTRY = {
    "CosineAnnealingLR": lr_scheduler.CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": lr_scheduler.CosineAnnealingWarmRestarts,
    "StepLR": lr_scheduler.StepLR,
    "ExponentialLR": lr_scheduler.ExponentialLR,
    "ConstantLR": lr_scheduler.ConstantLR,
}

_MUON_FALLBACK_DEFAULTS = {
    "type": "AdamW", "betas": [0.9, 0.999], "eps": 1e-5, "weight_decay": 0.01,
}


class _CompositeOptimizer(optim.Optimizer):
    """Wraps two optimizers (Muon for 2D weights + fallback for the rest)
    behind a single ``torch.optim.Optimizer``-compatible interface.

    Inherits from Optimizer so that PyTorch schedulers accept it via
    ``isinstance(optimizer, Optimizer)``.
    """

    def __init__(self, muon_optimizer, fallback_optimizer):
        # Bypass Optimizer.__init__; set the attributes schedulers expect.
        self.muon = muon_optimizer
        self.fallback = fallback_optimizer
        self.defaults = {}
        self.state = {}

    @property
    def param_groups(self):
        return self.muon.param_groups + self.fallback.param_groups

    @param_groups.setter
    def param_groups(self, _value):
        pass  # schedulers may try to assign; silently ignore

    def step(self, closure=None):
        self.muon.step(closure)
        self.fallback.step(closure)

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.fallback.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "fallback": self.fallback.state_dict(),
        }

    def load_state_dict(self, state_dict):
        if "muon" in state_dict and "fallback" in state_dict:
            self.muon.load_state_dict(state_dict["muon"])
            self.fallback.load_state_dict(state_dict["fallback"])
        else:
            logger.warning(
                "Loading non-composite state_dict into CompositeOptimizer; "
                "applying to fallback optimizer only."
            )
            self.fallback.load_state_dict(state_dict)


def _filter_kwargs(cls, kwargs):
    """Filter kwargs to only include parameters accepted by cls.__init__."""
    sig = inspect.signature(cls)
    valid_keys = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def _build_single_optimizer(cls, params, lr, config_dict):
    """Build a single optimizer from a class, params, lr, and filtered kwargs."""
    kwargs = dict(config_dict)
    if "betas" in kwargs and isinstance(kwargs["betas"], list):
        kwargs["betas"] = tuple(kwargs["betas"])
    kwargs["lr"] = lr
    kwargs = _filter_kwargs(cls, kwargs)
    return cls(params, **kwargs)


def create_optimizer(config_dict: dict, params, lr: float) -> optim.Optimizer:
    """Create an optimizer from a config dict.

    The config dict should have a ``type`` key naming the optimizer class.
    All other keys are forwarded as constructor kwargs, filtered to what
    the chosen optimizer actually accepts (via ``inspect.signature``).

    For **Muon**, parameters are automatically split: 2D weight matrices
    go to Muon, everything else goes to a fallback optimizer (AdamW by
    default). Configure the fallback via the ``fallback`` key::

        {"type": "Muon", "momentum": 0.95, "fallback": {"type": "AdamW", ...}}
    """
    config_dict = dict(config_dict)  # shallow copy
    opt_type = config_dict.pop("type", "Adam")

    cls = OPTIMIZER_REGISTRY.get(opt_type)
    if cls is None:
        raise ValueError(
            f"Unknown optimizer '{opt_type}'. "
            f"Available: {list(OPTIMIZER_REGISTRY.keys())}"
        )

    # --- Muon: split params into 2D (Muon) and non-2D (fallback) ---
    if opt_type == "Muon":
        fallback_cfg = config_dict.pop("fallback", None) or dict(_MUON_FALLBACK_DEFAULTS)
        return _create_muon_composite(config_dict, fallback_cfg, params, lr)

    # --- Standard optimizer ---
    logger.info(f"Creating optimizer: {opt_type}")
    return _build_single_optimizer(cls, params, lr, config_dict)


def _create_muon_composite(muon_cfg, fallback_cfg, params, lr):
    """Split params by dimensionality and return a CompositeOptimizer or plain optimizer."""
    # Materialize params so we can iterate twice
    param_list = list(params)

    muon_params = [p for p in param_list if p.ndim == 2]
    fallback_params = [p for p in param_list if p.ndim != 2]

    n_muon = len(muon_params)
    n_fallback = len(fallback_params)

    if n_muon == 0:
        logger.warning(
            "Muon selected but no 2D parameters found. "
            "Using fallback optimizer for all parameters."
        )
        fallback_cfg = dict(fallback_cfg)
        fb_type = fallback_cfg.pop("type", "AdamW")
        fb_cls = OPTIMIZER_REGISTRY[fb_type]
        return _build_single_optimizer(fb_cls, param_list, lr, fallback_cfg)

    # Build Muon for 2D params
    muon_opt = _build_single_optimizer(optim.Muon, muon_params, lr, muon_cfg)

    if n_fallback == 0:
        logger.info(f"Muon: all {n_muon} params are 2D, no fallback needed")
        return muon_opt

    # Build fallback for non-2D params
    fallback_cfg = dict(fallback_cfg)
    fb_type = fallback_cfg.pop("type", "AdamW")
    fb_cls = OPTIMIZER_REGISTRY.get(fb_type)
    if fb_cls is None:
        raise ValueError(f"Unknown fallback optimizer '{fb_type}'")
    fb_opt = _build_single_optimizer(fb_cls, fallback_params, lr, fallback_cfg)

    logger.info(
        f"Muon composite: {n_muon} params -> Muon, "
        f"{n_fallback} params -> {fb_type}"
    )
    return _CompositeOptimizer(muon_opt, fb_opt)


def create_scheduler(config_dict: dict, optimizer):
    """Create a scheduler from a config dict, or return None.

    The config dict should have a ``type`` key naming the scheduler class
    (or ``"None"`` to disable scheduling). All other keys are forwarded
    as constructor kwargs, filtered to what the chosen scheduler accepts.

    Example config_dict::

        {"type": "CosineAnnealingLR", "T_max": 1000000}
    """
    if config_dict is None:
        return None

    config_dict = dict(config_dict)
    sched_type = config_dict.pop("type", "CosineAnnealingLR")

    if sched_type == "None" or sched_type is None:
        return None

    cls = SCHEDULER_REGISTRY.get(sched_type)
    if cls is None:
        raise ValueError(
            f"Unknown scheduler '{sched_type}'. "
            f"Available: {list(SCHEDULER_REGISTRY.keys()) + ['None']}"
        )

    kwargs = _filter_kwargs(cls, config_dict)

    # For CompositeOptimizer, scheduler operates on the combined param_groups
    logger.info(f"Creating scheduler: {sched_type}({kwargs})")
    return cls(optimizer, **kwargs)
