"""The navigate environment as seen from Python: buffers, validation, determinism."""

import threading
import time

import numpy as np
import pytest
import torch

from roshan import NavigateEnv, _core


def test_spec_names_every_column():
    env = NavigateEnv(3)
    spec = env.spec
    assert spec["drones"] == 4 and spec["num_envs"] == 3
    assert env.agent.shape == (3, 4, len(spec["agent"]))
    assert env.neighbors.shape == (3, 4, 4, len(spec["neighbor"]))
    assert env.terms.shape == (3, 4, len(spec["terms"]))
    assert {"goal", "success", "collision", "out_of_bounds", "timeout", "progress"} <= set(spec["terms"])
    assert spec["config"]["max_speed_ms"] == 10.0 and spec["sim"]["dt_s"] == pytest.approx(0.1)
    assert env.agent.abs().sum() > 0, "reset already wrote the first observation"


def test_config_overrides_and_rejects_unknown_keys():
    env = NavigateEnv(2, config={"drones": 6, "goals": 1}, nx=50, ny=20)
    assert env.spec["config"]["drones"] == 6 and env.agent.shape[1] == 6
    with pytest.raises(ValueError, match="unknown field"):
        NavigateEnv(2, config={"dronez": 6})
    with pytest.raises(ValueError, match="positive"):
        NavigateEnv(2, config={"drones": 0})
    with pytest.raises(ValueError):
        NavigateEnv(2, sim={"dt_s": -1.0})


def test_rust_writes_straight_into_the_torch_buffers():
    env = NavigateEnv(8)
    assert np.shares_memory(env._obs[0], env.agent.numpy())
    assert env.agent.is_pinned() == torch.cuda.is_available()
    before = env.agent.clone()
    env.step(torch.ones(8, 4, 2))
    assert not torch.equal(env.agent, before)
    assert torch.all(env.actions == 1.0), "actions were copied into the shared buffer"


def test_bad_buffers_are_rejected_not_copied():
    core = _core.NavigateEnv(4)
    agent = np.zeros((4, 4, 8), np.float32)
    neighbors = np.zeros((4, 4, 4, 4), np.float32)
    mask = np.zeros((4, 4, 4), np.uint8)
    core.reset(agent, neighbors, mask)
    with pytest.raises(ValueError, match="C-contiguous"):
        core.reset(np.zeros((4, 4, 16), np.float32)[:, :, ::2], neighbors, mask)
    with pytest.raises(ValueError, match="C-contiguous"):
        core.reset(np.asfortranarray(agent), neighbors, mask)
    with pytest.raises(ValueError, match="shape"):
        core.reset(np.zeros((5, 4, 8), np.float32), neighbors, mask)
    with pytest.raises(TypeError):
        core.reset(agent.astype(np.float64), neighbors, mask)
    with pytest.raises(TypeError):
        core.reset(agent, neighbors, mask.astype(np.float32))


def test_same_seed_same_episodes_and_the_reset_step_is_empty():
    def trace(seed):
        env = NavigateEnv(64, seed=seed, threads=3)
        torch.manual_seed(0)
        rows, after_end = [], torch.zeros(64, dtype=torch.bool)
        for _ in range(400):
            env.step(torch.rand(64, 4, 2) * 2 - 1)
            # The step after an episode ended only resets: no reward terms, no flags.
            assert env.terms[after_end].abs().sum() == 0
            assert (env.terminated[after_end] | env.truncated[after_end]).sum() == 0
            after_end = (env.terminated | env.truncated) > 0
            rows.append(torch.cat([env.agent.flatten(), env.terms.flatten(), env.terminated.float()]))
        return torch.stack(rows)

    a = trace(1)
    assert torch.equal(a, trace(1))
    assert not torch.equal(a, trace(2))


def test_interpreter_lock_is_released_during_step():
    env = NavigateEnv(400_000, threads=1)
    count, stop = [0], threading.Event()

    def spin():
        while not stop.is_set():
            count[0] += 1

    spinner = threading.Thread(target=spin)
    spinner.start()
    try:
        time.sleep(0.1)
        free_rate = count[0] / 0.1
        before, start = count[0], time.perf_counter()
        env.step()
        elapsed, during = time.perf_counter() - start, count[0] - before
    finally:
        stop.set()
        spinner.join()
    assert elapsed > 0.02, f"step too short to measure: {elapsed}"
    # With the lock held the spinner would not advance at all during the step.
    assert during > 0.2 * free_rate * elapsed, (during, free_rate, elapsed)
