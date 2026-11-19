"""Checks that the viewer works inside a Python process: a torch policy is called from the
simulation thread, Ctrl+C still works, and a failing policy does not take the window down.

    uv run python spikes/ui_spike.py --auto    # opens windows briefly
"""

import os
import signal
import subprocess
import sys
import threading
import time

import torch

from roshan import _core


def main():
    net = torch.nn.Linear(8, 2)
    policy_threads = set()

    def policy(obs):
        policy_threads.add(threading.get_ident())
        with torch.no_grad():
            return torch.tanh(net(torch.from_numpy(obs["agent"]))).flatten().tolist()

    def fly(policy, seconds):
        return _core.run_ui(policy=policy, auto_close_s=seconds, navigate="{}")

    if "--auto" not in sys.argv:
        print(fly(policy, None))
        return

    env = {k: v for k, v in os.environ.items() if k not in ("DISPLAY", "WAYLAND_DISPLAY")}
    subprocess.run([sys.executable, "-c", "import roshan"], env=env, check=True)
    print("ok  import works without a display")

    frames, ticks, calls = fly(policy, 3.0)
    assert calls > 30 and calls == ticks, f"policy called {calls} times in {ticks} ticks"
    assert threading.main_thread().ident not in policy_threads, "policy ran on the UI thread"
    print(f"ok  torch policy called {calls} times from the sim thread ({frames} frames, {ticks} ticks)")

    sent = []

    def interrupt():
        time.sleep(1.0)
        sent.append(time.perf_counter())
        os.kill(os.getpid(), signal.SIGINT)

    threading.Thread(target=interrupt, daemon=True).start()
    try:
        fly(policy, 10.0)
    except KeyboardInterrupt:
        delay = time.perf_counter() - sent[0]
        assert delay < 0.5, f"KeyboardInterrupt took {delay:.2f} s"
        print(f"ok  second window in the same process; Ctrl+C raised KeyboardInterrupt after {delay * 1000:.0f} ms")
    else:
        raise SystemExit("FAIL  SIGINT did not raise KeyboardInterrupt")

    def broken(obs):
        raise RuntimeError("boom")

    frames, ticks, calls = fly(broken, 1.5)
    assert ticks == 0 and frames > 0, (frames, ticks)
    print("ok  a failing policy pauses the simulation and the window stays up")

    # A scenario with drones: the allocate task with each built-in planner, then with a policy.
    for planner in ("off", "greedy", "rollout"):
        frames, ticks, calls = _core.run_ui(auto_close_s=2.0, planner=planner)
        assert ticks > 100 and calls == 0, (planner, ticks, calls)
    seen = {}

    def keep(obs):
        seen.update({k: v.shape for k, v in obs.items()})
        return [-1.0] * obs["drones"].shape[0]

    frames, ticks, calls = _core.run_ui(policy=keep, auto_close_s=2.0, planner="policy")
    assert calls >= 1 and seen["action_mask"] == (seen["drones"][0], seen["targets"][0]), (calls, seen)
    print(f"ok  allocate runs with off, greedy, rollout and a policy ({calls} decisions asked)")


if __name__ == "__main__":
    main()
