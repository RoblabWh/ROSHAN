"""Throughput of the simulator alone: python -m roshan.bench

End-to-end training throughput is printed by `python -m roshan.rl.train`.
"""

import os
import time

import torch

from roshan import NavigateEnv


def rate(threads, num_envs=4096, seconds=2.0):
    """Drone-steps per second of the navigate task with random actions."""
    env = NavigateEnv(num_envs, threads=threads)
    env.actions.uniform_(-1, 1)
    env.step()  # warm-up: threads started, pages touched
    steps, start = 0, time.perf_counter()
    while time.perf_counter() - start < seconds:
        env.step()
        steps += 1
    return steps * env.actions.shape[0] * env.actions.shape[1] / (time.perf_counter() - start)


def main():
    cores = os.cpu_count()
    print(f"navigate, 4096 environments x 4 drones, random actions, {cores} cores")
    base = None
    for threads in sorted({1, 4, 8, 16, cores}):
        r = rate(threads)
        base = base or r
        print(f"  {threads:>2} threads: {r / 1e6:6.2f} M drone-steps/s   {r / base:5.1f}x")
    env = NavigateEnv(1, threads=1)
    start = time.perf_counter()
    for _ in range(20_000):
        env.step()
    print(f"  one environment: {(time.perf_counter() - start) / 20_000 * 1e6:.2f} µs per step call")
    if torch.cuda.is_available():
        print(f"  buffers pinned: {env.agent.is_pinned()}")


if __name__ == "__main__":
    main()
