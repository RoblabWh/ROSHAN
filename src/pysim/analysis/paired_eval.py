#!/usr/bin/env python3
"""Paired comparison of ROSHAN evaluation runs (evaluation_stats.csv).

    paired_eval.py --arm greedy A/logs/evaluation_stats.csv [A2.csv ...] \
                   --arm learned B1.csv B2.csv ... [--pair greedy learned] [--ci 0.95] [--boot 2000]

Per arm, two aggregation levels are printed and labelled: pooled over episodes (Wilson CI on
success, bootstrap CIs on TTE) and, for arms with several CSVs, across networks (mean +- std of
per-network values). Censoring rule: timed-out episodes carry Time = step budget, so TTE-all is
a censored quantity; TTE-on-success is the uncensored one.

Pairing joins the two arms positionally after sorting each CSV by Episode_Index and REQUIRES
identical Fire_Fingerprint per row (same seed => same scenario). A single-CSV arm is tiled against
a multi-CSV arm (deterministic baseline vs k trained networks). CSVs without identifiers (pre
Sept 2026) pair by row order with a warning.
"""
import argparse
import csv
import sys
import warnings

import numpy as np
from scipy import stats as st

IDENT = ("Episode_Index", "Episode_Seed", "Fire_Fingerprint")


def load_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: no episodes")
    cols = {}
    for k in rows[0]:
        vals = [r[k] for r in rows]
        cols[k] = np.array(vals, dtype=object) if k == "Failure_Reason" else np.array([float(v) for v in vals])
    for req in ("Time", "Success", "Failure_Reason"):
        if req not in cols:
            raise ValueError(f"{path}: missing column {req}")
    if "Episode_Index" in cols:
        order = np.argsort(cols["Episode_Index"], kind="stable")
        cols = {k: v[order] for k, v in cols.items()}
    return cols


def concat(csvs):
    keys = set.intersection(*(set(c) for c in csvs))
    return {k: np.concatenate([c[k] for c in csvs]) for k in keys}


def wilson(k, n, ci):
    return tuple(st.binomtest(int(k), int(n)).proportion_ci(ci, method="wilson")) if n else (float("nan"),) * 2


def boot_ci(vals, fn, ci, n_boot, rng):
    vals = np.asarray(vals, dtype=float)
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(vals), size=(n_boot, len(vals)))
    s = fn(vals[idx], axis=1)
    a = (1 - ci) / 2
    return tuple(float(x) for x in np.quantile(s, [a, 1 - a]))


def summarize_arm(csvs, ci, n_boot, rng):
    pooled = concat(csvs)
    succ = pooled["Success"] > 0.5
    tte_ok = pooled["Time"][succ]
    out = {
        "n": len(succ), "networks": len(csvs),
        "success": succ.mean(), "success_ci": wilson(succ.sum(), len(succ), ci),
        "timeout": float(np.mean(pooled["Failure_Reason"] == "Timeout")),
        "tte_all_median": float(np.median(pooled["Time"])), "tte_all_mean": float(np.mean(pooled["Time"])),
        "tte_ok_n": len(tte_ok),
        "tte_ok_median": float(np.median(tte_ok)) if len(tte_ok) else float("nan"),
        "tte_ok_median_ci": boot_ci(tte_ok, np.median, ci, n_boot, rng),
        "tte_ok_mean": float(np.mean(tte_ok)) if len(tte_ok) else float("nan"),
        "tte_ok_mean_ci": boot_ci(tte_ok, np.mean, ci, n_boot, rng),
    }
    if "Percent_Burned" in pooled:
        b = pooled["Percent_Burned"]
        out["burned_mean"] = float(np.mean(b)); out["burned_mean_ci"] = boot_ci(b, np.mean, ci, n_boot, rng)
        out["burned_median"] = float(np.median(b))
    if "Collision_Events" in pooled:
        strict = succ & (pooled["Collision_Events"] == 0)
        out["strict_success"] = strict.mean()
        out["strict_success_ci"] = wilson(strict.sum(), len(strict), ci)
    if len(csvs) > 1:
        per_s = [float(np.mean(c["Success"] > 0.5)) for c in csvs]
        per_t = [float(np.median(c["Time"][c["Success"] > 0.5])) if np.any(c["Success"] > 0.5) else float("nan")
                 for c in csvs]
        out["net_success"] = (float(np.mean(per_s)), float(np.std(per_s, ddof=1)))
        out["net_tte_ok_median"] = (float(np.nanmean(per_t)), float(np.nanstd(per_t, ddof=1)))
    return out


def pair_arms(a_csvs, b_csvs):
    if len(a_csvs) == 1 and len(b_csvs) > 1:
        a_csvs = a_csvs * len(b_csvs)
    elif len(b_csvs) == 1 and len(a_csvs) > 1:
        b_csvs = b_csvs * len(a_csvs)
    elif len(a_csvs) != len(b_csvs):
        raise ValueError(f"cannot pair {len(a_csvs)} vs {len(b_csvs)} CSVs (tile only works from one)")
    a, b = concat(a_csvs), concat(b_csvs)
    if len(a["Time"]) != len(b["Time"]):
        raise ValueError(f"episode counts differ: {len(a['Time'])} vs {len(b['Time'])}")
    if "Fire_Fingerprint" in a and "Fire_Fingerprint" in b:
        bad = np.flatnonzero(a["Fire_Fingerprint"] != b["Fire_Fingerprint"])
        if len(bad):
            i = int(bad[0])
            raise ValueError(f"row {i}: Fire_Fingerprint {int(a['Fire_Fingerprint'][i])} != "
                             f"{int(b['Fire_Fingerprint'][i])} — runs were not seeded identically, refusing to pair")
    else:
        # ponytail: positional pairing; add per-network matching if arms ever mix seeds
        warnings.warn("no Fire_Fingerprint column on both arms: pairing by row order, unverified")
    return a, b


def paired_stats(a, b, ci, n_boot, rng):
    sa, sb = a["Success"] > 0.5, b["Success"] > 0.5
    both = sa & sb
    only_a, only_b = int(np.sum(sa & ~sb)), int(np.sum(~sa & sb))
    disc = only_a + only_b
    out = {"n": len(sa), "both": int(both.sum()), "only_a": only_a, "only_b": only_b,
           "neither": int(np.sum(~sa & ~sb)),
           "mcnemar_p": st.binomtest(only_b, disc, 0.5).pvalue if disc else float("nan")}
    d = b["Time"][both] - a["Time"][both]
    out["tte_n"] = len(d)
    out["diff_mean"] = float(np.mean(d)) if len(d) else float("nan")
    out["diff_mean_ci"] = boot_ci(d, np.mean, ci, n_boot, rng)
    out["diff_median"] = float(np.median(d)) if len(d) else float("nan")
    out["diff_median_ci"] = boot_ci(d, np.median, ci, n_boot, rng)
    if len(d) >= 2 and np.any(d != 0):
        out["wilcoxon_p"] = float(st.wilcoxon(d).pvalue)
    else:
        out["wilcoxon_p"] = float("nan")
    if "Percent_Burned" in a and "Percent_Burned" in b:
        db = b["Percent_Burned"] - a["Percent_Burned"]  # all pairs, not success-conditional
        out["burned_diff_mean"] = float(np.mean(db))
        out["burned_diff_mean_ci"] = boot_ci(db, np.mean, ci, n_boot, rng)
        out["burned_wilcoxon_p"] = float(st.wilcoxon(db).pvalue) if len(db) >= 2 and np.any(db != 0) else float("nan")
    return out


def _ci(c):
    return f"[{c[0]:.6g}, {c[1]:.6g}]"


def render_arm(name, s, ci):
    pct = int(ci * 100)
    lines = [f"### {name}  (pooled over {s['n']} episodes from {s['networks']} CSV(s); timeouts carry Time = budget)",
             "", "| metric | value |", "|---|---|",
             f"| success rate ({pct}% Wilson) | {s['success']:.3f} {_ci(s['success_ci'])} |",
             f"| timeout share | {s['timeout']:.3f} |",
             f"| TTE all episodes, median / mean (censored) | {s['tte_all_median']:.0f} / {s['tte_all_mean']:.0f} |",
             f"| TTE on success, median ({pct}% bootstrap) | {s['tte_ok_median']:.0f} {_ci(s['tte_ok_median_ci'])} (n={s['tte_ok_n']}) |",
             f"| TTE on success, mean ({pct}% bootstrap) | {s['tte_ok_mean']:.0f} {_ci(s['tte_ok_mean_ci'])} |"]
    if "burned_mean" in s:
        lines.append(f"| burned area, mean ({pct}% bootstrap) / median | {s['burned_mean']:.4f} {_ci(s['burned_mean_ci'])} / {s['burned_median']:.4f} |")
    if "strict_success" in s:
        lines.append(f"| strict success (no collision event) | {s['strict_success']:.3f} {_ci(s['strict_success_ci'])} |")
    if "net_success" in s:
        m, sd = s["net_success"]
        lines.append(f"| across networks: success mean +- std (n={s['networks']}) | {m:.3f} +- {sd:.3f} |")
        m, sd = s["net_tte_ok_median"]
        lines.append(f"| across networks: TTE-on-success median mean +- std | {m:.0f} +- {sd:.0f} |")
    return "\n".join(lines) + "\n"


def render_pair(na, nb, p, ci):
    pct = int(ci * 100)
    return "\n".join([
        f"### paired: {nb} - {na}  ({p['n']} episode pairs, fingerprints verified unless warned)", "",
        "| metric | value |", "|---|---|",
        f"| success: both / only {na} / only {nb} / neither | {p['both']} / {p['only_a']} / {p['only_b']} / {p['neither']} |",
        f"| McNemar exact p (success) | {p['mcnemar_p']:.3g} |",
        f"| TTE diff on jointly completed episodes, n | {p['tte_n']} |",
        f"| mean diff ({pct}% paired bootstrap) | {p['diff_mean']:.1f} {_ci(p['diff_mean_ci'])} |",
        f"| median diff ({pct}% paired bootstrap) | {p['diff_median']:.1f} {_ci(p['diff_median_ci'])} |",
        f"| Wilcoxon signed-rank p | {p['wilcoxon_p']:.3g} |"]
        + ([f"| burned-area diff, all pairs, mean ({pct}% paired bootstrap) | {p['burned_diff_mean']:.4f} {_ci(p['burned_diff_mean_ci'])} |",
            f"| burned-area Wilcoxon p | {p['burned_wilcoxon_p']:.3g} |"] if "burned_diff_mean" in p else [])) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", action="append", nargs="+", metavar=("NAME", "CSV"), required=True)
    ap.add_argument("--pair", nargs=2, metavar=("A", "B"), help="arms to pair (default: first two)")
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)
    arms = {a[0]: [load_csv(p) for p in a[1:]] for a in args.arm}
    for name, csvs in arms.items():
        print(render_arm(name, summarize_arm(csvs, args.ci, args.boot, rng), args.ci))
    names = args.pair or list(arms)[:2]
    if len(names) == 2:
        a, b = pair_arms(arms[names[0]], arms[names[1]])
        print(render_pair(names[0], names[1], paired_stats(a, b, args.ci, args.boot, rng), args.ci))
    return 0


if __name__ == "__main__":
    sys.exit(main())
