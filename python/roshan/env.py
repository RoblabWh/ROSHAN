"""Batched environments. Their buffers are torch tensors that Rust writes in place."""

import json

import torch

from roshan import _core


class NavigateEnv:
    """Drones fly to waypoints without colliding (crates/roshan-core/src/navigate.rs).

    Buffers live on the CPU, pinned when CUDA is available, with E environments of D drones:

        actions     [E, D, 2]     wanted velocity as a fraction of the maximum (read by step)
        agent       [E, D, 8]     own state                      see spec["agent"]
        neighbors   [E, D, D, 4]  slot 0 map edge, then drones   see spec["neighbor"]
        mask        [E, D, D]     1 where a neighbour slot is in view
        terms       [E, D, 8]     reward building blocks         see spec["terms"]
        terminated  [E]           episode ended for good (success, collision, left the map)
        truncated   [E]           episode cut off by the time limit
        info        [E, 4]        episode summary, valid where an episode ended; spec["info"]

    The step after an episode ended resets that environment and ignores its action, so the
    final observation of every episode is visible once. With `autoreset = False` finished
    environments stay frozen instead (evaluation: every seed runs exactly once).
    """

    def __init__(self, num_envs, *, seed=0, threads=0, nx=35, ny=36, cell_m=10.0, config=None, sim=None):
        self._env = _core.NavigateEnv(
            num_envs, nx, ny, cell_m, seed, threads, json.dumps(config or {}), json.dumps(sim or {})
        )
        spec = dict(self._env.spec)
        spec["config"], spec["sim"] = json.loads(spec["config"]), json.loads(spec["sim"])
        self.spec = spec

        e, d = num_envs, spec["drones"]
        pin = torch.cuda.is_available()

        def buffer(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        self.actions = buffer(e, d, 2)
        self.agent = buffer(e, d, len(spec["agent"]))
        self.neighbors = buffer(e, d, d, len(spec["neighbor"]))
        self.mask = buffer(e, d, d, dtype=torch.uint8)
        self.terms = buffer(e, d, len(spec["terms"]))
        self.terminated = buffer(e, dtype=torch.uint8)
        self.truncated = buffer(e, dtype=torch.uint8)
        self.info = buffer(e, len(spec["info"]))
        self._obs = [t.numpy() for t in (self.agent, self.neighbors, self.mask)]
        self._out = [t.numpy() for t in (self.terms, self.terminated, self.truncated, self.info)]
        self._actions = self.actions.numpy()
        self.reset()

    @property
    def autoreset(self):
        return self._env.autoreset

    @autoreset.setter
    def autoreset(self, on):
        self._env.autoreset = on

    def reset(self):
        """Start a fresh episode in every environment."""
        self._env.reset(*self._obs)

    def pilot(self, kind):
        """Actions [E, D, 2] of an untrained pilot for the current state: "straight" (at the
        waypoint, no avoidance), "scripted" (centralised, sees everything: an upper bound) or
        "decentral" (each drone alone from its observation row, like a fly policy)."""
        return torch.from_numpy(self._env.pilot(kind))

    def step(self, actions=None):
        """Advance one tick. `actions` (any device) is copied into `self.actions` first."""
        if actions is not None:
            self.actions.copy_(actions)
        self._env.step(self._actions, *self._obs, *self._out)


class AllocateEnv:
    """A planner assigns targets to drones on one scenario (crates/roshan-core/src/allocate.rs).

    Buffers (CPU, pinned when CUDA is available), E environments, D drones, S target slots
    (the scenario's ground stations first, then fire targets):

        actions      [E, D] int32   slot to fly to, or -1 to keep the current task (read by step)
        drones       [E, D, 6]      see spec["drone"]; positions are fractions of the map size
        targets      [E, S, 4]      see spec["target"]
        target_mask  [E, S]         1 where the slot holds a target
        action_mask  [E, D, S]      1 where that drone may choose that slot
        globals      [E, 5]         see spec["global"]
        decide       [E, D]         1 where the drone needs a decision now
        terms        [E, D, 5]      reward building blocks for the step, see spec["terms"]
        elapsed_s    [E]            simulated seconds the step took
        terminated   [E]            the fire went out
        truncated    [E]            the time limit was reached
        info         [E, 7]         episode summary where an episode ended, see spec["info"]
        seed, fingerprint [E]       the episode's seed and a digest of the fire it started with

    One step runs until the next decision event. `reset(seeds)` starts the episodes of the
    given seeds; an episode depends on nothing but its seed. The step after an episode ended
    resets that environment, unless `autoreset` is off, in which case it stays frozen.
    """

    def __init__(self, num_envs, scenario, *, seed=0, threads=0, config=None, sim=None):
        d = scenario.to_dict() if hasattr(scenario, "to_dict") else scenario
        self._env = _core.AllocateEnv(num_envs, d, seed, threads, json.dumps(config or {}), json.dumps(sim or {}))
        spec = dict(self._env.spec)
        spec["config"], spec["sim"] = json.loads(spec["config"]), json.loads(spec["sim"])
        self.spec = spec

        e, n, s = num_envs, spec["drones"], spec["slots"]
        pin = torch.cuda.is_available()

        def buffer(*shape, dtype=torch.float32):
            return torch.zeros(shape, dtype=dtype, pin_memory=pin)

        self.actions = buffer(e, n, dtype=torch.int32)
        self.drones = buffer(e, n, len(spec["drone"]))
        self.targets = buffer(e, s, len(spec["target"]))
        self.target_mask = buffer(e, s, dtype=torch.uint8)
        self.action_mask = buffer(e, n, s, dtype=torch.uint8)
        self.globals = buffer(e, len(spec["global"]))
        self.decide = buffer(e, n, dtype=torch.uint8)
        self.terms = buffer(e, n, len(spec["terms"]))
        self.elapsed_s = buffer(e)
        self.terminated = buffer(e, dtype=torch.uint8)
        self.truncated = buffer(e, dtype=torch.uint8)
        self.info = buffer(e, len(spec["info"]))
        self.seed = buffer(e, dtype=torch.uint64)
        self.fingerprint = buffer(e, dtype=torch.uint64)
        names = ("actions drones targets target_mask action_mask globals decide terms elapsed_s "
                 "terminated truncated info seed fingerprint").split()
        self._arrays = [getattr(self, name).numpy() for name in names]
        self.reset()

    @property
    def autoreset(self):
        return self._env.autoreset

    @autoreset.setter
    def autoreset(self, on):
        self._env.autoreset = on

    def reset(self, seeds=None):
        """Start fresh episodes: those of `seeds` (one per environment), or the next training ones."""
        self._env.advance(True, None if seeds is None else [int(s) for s in seeds], *self._arrays)

    def plan(self, method, horizon_s=120.0, candidates=4):
        """Actions [E, D] of a built-in planner for the current state: "greedy", or "rollout",
        which improves on greedy by simulating `horizon_s` ahead on copies of the world."""
        actions = torch.full(self.actions.shape, -1, dtype=torch.int32)
        self._env.plan(method, actions.numpy(), horizon_s, candidates)
        return actions

    def step(self, actions=None):
        """Run to the next decision event. `actions` (any device) is copied into `self.actions`."""
        if actions is not None:
            self.actions.copy_(actions)
        self._env.advance(False, None, *self._arrays)

    def fly_tick(self, velocity, actions=None, view_m=50.0):
        """One tick in which an outside pilot flies the drones instead of the scripted controller.

        `velocity` [E, D, 2] are fractions of the maximum speed; `actions` are applied where
        drones wait for a decision. Afterwards `self.pilot` = (agent, neighbors, mask) holds
        what a fly policy sees next (as in the navigate task, distances divided by `view_m`),
        and the planner's tables are current. Nothing keeps drones apart or on the map here;
        `info` reports the smallest distance between two drones when an episode ends.
        """
        e, d = self.actions.shape
        if not hasattr(self, "pilot"):
            pin = torch.cuda.is_available()
            self.velocity = torch.zeros((e, d, 2), pin_memory=pin)
            self.pilot = (
                torch.zeros((e, d, 8), pin_memory=pin),
                torch.zeros((e, d, d, 4), pin_memory=pin),
                torch.zeros((e, d, d), dtype=torch.uint8, pin_memory=pin),
            )
        self.velocity.copy_(velocity)
        if actions is None:
            self.actions.fill_(-1)
        else:
            self.actions.copy_(actions)
        flight = (self.velocity.numpy(), float(view_m), *(t.numpy() for t in self.pilot))
        self._env.advance(False, None, *self._arrays, flight)
