"""Checks for analysis/paired_eval.py on tiny synthetic evaluation CSVs."""
import csv

import numpy as np
import pytest

from analysis import paired_eval as pe  # conftest puts src/pysim on sys.path

FIELDS = ["Reward", "Time", "Percent_Burned", "Success", "Failure_Reason",
          "Episode_Index", "Episode_Seed", "Fire_Fingerprint"]


def _write(path, times, success, fps=None, extra=None):
    rows = []
    for i, (t, s) in enumerate(zip(times, success)):
        r = {"Reward": 1.0, "Time": t, "Percent_Burned": 0.0, "Success": float(s),
             "Failure_Reason": "None" if s else "Timeout",
             "Episode_Index": i + 1, "Episode_Seed": 7, "Fire_Fingerprint": (fps or list(range(100, 100 + len(times))))[i]}
        if extra:
            r.update({k: v[i] for k, v in extra.items()})
        rows.append(r)
    fields = FIELDS + [k for k in (extra or {}) if k not in FIELDS]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return str(path)


def test_fingerprint_mismatch_refuses(tmp_path):
    a = _write(tmp_path / "a.csv", [100] * 4, [1] * 4)
    b = _write(tmp_path / "b.csv", [100] * 4, [1] * 4, fps=[100, 101, 999, 103])
    with pytest.raises(ValueError, match="row 2"):
        pe.pair_arms([pe.load_csv(a)], [pe.load_csv(b)])


def test_constant_shift_is_detected(tmp_path):
    ta = [1000 + 37 * i for i in range(10)]
    a = _write(tmp_path / "a.csv", ta, [1] * 10)
    b = _write(tmp_path / "b.csv", [t - 100 for t in ta], [1] * 10)
    rng = np.random.default_rng(0)
    A, B = pe.pair_arms([pe.load_csv(a)], [pe.load_csv(b)])
    p = pe.paired_stats(A, B, 0.95, 500, rng)
    assert p["tte_n"] == 10 and p["diff_median"] == -100
    assert p["diff_median_ci"][1] < 0 and p["diff_mean_ci"][1] < 0
    assert p["wilcoxon_p"] < 0.05
    assert p["both"] == 10 and np.isnan(p["mcnemar_p"])  # no discordant pairs


def test_strict_success_and_timeout_share(tmp_path):
    a = _write(tmp_path / "a.csv", [100, 200, 3000, 400], [1, 1, 0, 1],
               extra={"Collision_Events": [0, 2, 0, 0]})
    s = pe.summarize_arm([pe.load_csv(a)], 0.95, 200, np.random.default_rng(0))
    assert s["success"] == 0.75 and s["strict_success"] == 0.5
    assert s["timeout"] == 0.25 and s["tte_ok_n"] == 3 and s["tte_all_median"] == 300
    assert s["burned_mean"] == 0.0 and s["burned_median"] == 0.0


def test_burned_area_paired_difference(tmp_path):
    a = _write(tmp_path / "a.csv", [100] * 6, [1] * 6, extra={"Percent_Burned": [0.10] * 6})
    b = _write(tmp_path / "b.csv", [100] * 6, [1] * 6, extra={"Percent_Burned": [0.04] * 6})
    A, B = pe.pair_arms([pe.load_csv(a)], [pe.load_csv(b)])
    p = pe.paired_stats(A, B, 0.95, 200, np.random.default_rng(0))
    assert abs(p["burned_diff_mean"] + 0.06) < 1e-9 and p["burned_diff_mean_ci"][1] < 0
    assert p["burned_wilcoxon_p"] < 0.05


def test_single_csv_arm_is_tiled_against_networks(tmp_path):
    a = _write(tmp_path / "a.csv", [100, 200], [1, 1])
    b1 = _write(tmp_path / "b1.csv", [90, 190], [1, 1])
    b2 = _write(tmp_path / "b2.csv", [80, 180], [1, 0])
    A, B = pe.pair_arms([pe.load_csv(a)], [pe.load_csv(b1), pe.load_csv(b2)])
    assert len(A["Time"]) == 4 and list(A["Time"]) == [100, 200, 100, 200]
    s = pe.summarize_arm([pe.load_csv(b1), pe.load_csv(b2)], 0.95, 200, np.random.default_rng(0))
    assert s["networks"] == 2 and s["net_success"][0] == 0.75


def test_old_csv_without_identifiers_pairs_by_row_with_warning(tmp_path):
    p = tmp_path / "old.csv"
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["Reward", "Time", "Percent_Burned", "Success", "Failure_Reason"])
        w.writeheader()
        w.writerows([{"Reward": 0, "Time": 10, "Percent_Burned": 0, "Success": 1, "Failure_Reason": "None"}] * 3)
    with pytest.warns(UserWarning, match="row order"):
        pe.pair_arms([pe.load_csv(str(p))], [pe.load_csv(str(p))])


def test_cli_runs_end_to_end(tmp_path, capsys):
    a = _write(tmp_path / "a.csv", [100, 200, 300], [1, 1, 1])
    b = _write(tmp_path / "b.csv", [90, 210, 250], [1, 1, 1])
    assert pe.main(["--arm", "greedy", a, "--arm", "hungarian", b, "--boot", "100"]) == 0
    out = capsys.readouterr().out
    assert "### greedy" in out and "### paired: hungarian - greedy" in out
