"""Layered configuration loading for ROSHAN (OmegaConf-based).

Replaces the old "one giant YAML per experiment" workflow. A single
``config/base.yaml`` holds all shared defaults; small overlay files under
``config/agent/`` and ``config/exp/`` contain only the deltas. Everything is
deep-merged in Python, then dumped *resolved* to ``used_config.yaml`` — which is
what the C++ engine parses (``model_parameters.h::init()`` via yaml-cpp). So all
layering is transparent to C++ as long as the dump stays plain YAML.

Invocation grammar (see ``split_args``)::

    python main.py config/agent/planner.yaml exp/water_on.yaml algorithm.PPO.lr=3e-4

Positional ``*.yaml`` args are overlays merged (in order) on top of ``base.yaml``;
``key=value`` args are dotted-path overrides applied last. Passing a legacy full
config (e.g. ``config_planner.yaml``) still works — merged on top of base, its keys
win and reproduce the old behavior.
"""
import os
from typing import List, Optional, Tuple

import yaml
from omegaconf import OmegaConf, DictConfig

from utils import get_project_paths

DEFAULT_BASE = "config/base.yaml"

_MISSING = object()


def _resolve(path: str) -> str:
    """Resolve a config path. main.py chdir's to the build dir, so relative paths
    are tried first as-given (cwd) then against the project root (the repo)."""
    if os.path.isabs(path):
        return path
    root_rel = os.path.join(get_project_paths("root_path"), path)
    if os.path.exists(path):
        return path
    if os.path.exists(root_rel):
        return root_rel
    raise FileNotFoundError(
        f"Config file not found: tried '{path}' (cwd) and '{root_rel}' (project root)."
    )


def split_args(argv: List[str]) -> Tuple[List[str], List[str]]:
    """Split CLI args into overlay file paths and dotted-path overrides.

    A token is an override if it contains '=' and does not look like a file path
    (``key=value`` vs ``some/file.yaml``). Everything else is an overlay path.
    """
    overlays, dotlist = [], []
    for a in argv:
        if "=" in a and not a.endswith((".yaml", ".yml")):
            dotlist.append(a)
        else:
            overlays.append(a)
    return overlays, dotlist


def _validate_dotlist_keys(cfg, dotlist: List[str]) -> None:
    """Reject CLI override keys that don't already exist in the merged config —
    this catches typos (the most common override mistake) with a clear error
    instead of silently creating a dead key that never reaches the engine."""
    unknown = [item.split("=", 1)[0] for item in dotlist
               if OmegaConf.select(cfg, item.split("=", 1)[0], default=_MISSING) is _MISSING]
    if unknown:
        raise KeyError(
            f"Unknown config override key(s): {unknown}. Overrides must target existing "
            f"dotted paths (e.g. 'algorithm.PPO.lr', 'environment.agent.use_water_limit')."
        )


def load_config(overlay_paths: Optional[List[str]] = None,
                dotlist: Optional[List[str]] = None,
                base_path: str = DEFAULT_BASE,
                struct: bool = False) -> DictConfig:
    """Load base.yaml, deep-merge overlays (in order) then dotted CLI overrides.

    Overlay files are trusted (full subtrees may be added). CLI ``key=value``
    overrides are validated against the merged config so typos fail loudly.
    ``struct=True`` additionally locks the returned config (opt-in; off by default
    because some runtime paths legitimately add keys, e.g. GUI-init eval).
    """
    overlay_paths = list(overlay_paths or [])
    dotlist = list(dotlist or [])

    cfg = OmegaConf.load(_resolve(base_path))
    for p in overlay_paths:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(_resolve(p)))
    if dotlist:
        _validate_dotlist_keys(cfg, dotlist)
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))

    if struct:
        OmegaConf.set_struct(cfg, True)
    return cfg


def dump_resolved(cfg, path: str) -> None:
    """Write a fully-resolved, plain-YAML snapshot (interpolations resolved, no
    OmegaConf/Python tags) — the format the C++ yaml-cpp parser expects. Used for
    both ``used_config.yaml`` and the per-model config snapshot."""
    if isinstance(cfg, (DictConfig,)) or OmegaConf.is_config(cfg):
        container = OmegaConf.to_container(cfg, resolve=True)
    else:
        container = cfg  # already a plain dict
    with open(path, "w") as f:
        yaml.safe_dump(container, f, sort_keys=False)


def cfg_get(cfg, dotted: str, default=None):
    """Safe dotted-path access, e.g. ``cfg_get(cfg, "environment.agent.planner_agent.hierarchy_timesteps", 60)``.
    Returns ``default`` if any segment is missing (works regardless of struct mode)."""
    return OmegaConf.select(cfg, dotted, default=default)


# C++-critical range checks NOT already covered by main.assert_config (which owns
# lr/gamma/eps_clip/horizon/view_range/num_agents). path -> (predicate, human description).
RANGE_CHECKS = {
    "environment.agent.water_capacity":                    (lambda v: v > 0, "must be > 0"),
    "environment.agent.recharge_time":                     (lambda v: v >= 0, "must be >= 0"),
    "environment.agent.water_refuel_threshold":            (lambda v: 0.0 <= v <= 1.0, "must be in [0, 1]"),
    "environment.agent.planner_agent.hierarchy_timesteps": (lambda v: v > 0, "must be > 0"),
    "fire_model.simulation.time.dt":                       (lambda v: v > 0, "must be > 0"),
    "fire_model.simulation.grid.exploration_map_size":     (lambda v: v > 0, "must be > 0"),
}


def _flatten(node, prefix=""):
    """Yield (dotted_path, leaf_value) for every scalar/list leaf in a nested dict.
    Lists are treated as opaque leaves (no element-level descent)."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _flatten(v, f"{prefix}.{k}" if prefix else k)
    else:
        yield prefix, node


def _select_plain(container, dotted):
    """Walk a plain nested dict by dotted path; return _MISSING if any segment is absent."""
    cur = container
    for part in dotted.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return _MISSING
    return cur


def _type_ok(expected, got):
    """Is `got` type-compatible with the reference leaf `expected`? bool is distinct from
    int; an int is accepted where a float is expected (not vice-versa); lists match lists."""
    if isinstance(expected, bool):
        return isinstance(got, bool)
    if isinstance(expected, int):           # int expected -> int only (not bool, not float)
        return isinstance(got, int) and not isinstance(got, bool)
    if isinstance(expected, float):         # float expected -> int or float ok
        return isinstance(got, (int, float)) and not isinstance(got, bool)
    if isinstance(expected, str):
        return isinstance(got, str)
    if isinstance(expected, list):
        return isinstance(got, list)
    return True  # None / unusual reference leaves: skip strict typing


def validate_config(cfg, reference_path: str = DEFAULT_BASE) -> None:
    """Fail-fast structural + type gate, run before the merged config is dumped for the C++
    engine. Asserts that every key present in the reference (``config/base.yaml`` — the full
    config the C++ side parses via ``model_parameters.h::init()``) exists in ``cfg`` with a
    compatible type, plus a few critical range checks. ALL problems are aggregated into one
    ValueError so they can be fixed in a single pass.

    Complements ``main.assert_config`` (semantic / cross-field rules); this owns the question
    "is every C++-required key present and correctly typed". ``base.yaml`` is the contract:
    add a key there and it is required automatically — no list to keep in sync with C++.
    """
    reference = yaml.safe_load(open(_resolve(reference_path)))
    merged = OmegaConf.to_container(cfg, resolve=True) if OmegaConf.is_config(cfg) else cfg
    problems = []

    for path, ref_leaf in _flatten(reference):
        got = _select_plain(merged, path)
        if got is _MISSING or got is None:
            problems.append(f"  {path}: missing (required by config/base.yaml / C++ engine)")
        elif not _type_ok(ref_leaf, got):
            problems.append(
                f"  {path}: type mismatch (expected {type(ref_leaf).__name__}, "
                f"got {type(got).__name__} = {got!r})"
            )

    for path, (ok, desc) in RANGE_CHECKS.items():
        v = _select_plain(merged, path)
        if v is _MISSING or v is None:
            continue  # absence already reported above
        try:
            valid = ok(v)
        except TypeError:
            valid = False  # non-numeric where a number was expected
        if not valid:
            problems.append(f"  {path}: {desc} (got {v!r})")

    if problems:
        raise ValueError(
            "Config validation failed — merged config is not safe to hand to the C++ engine:\n"
            + "\n".join(problems)
        )
