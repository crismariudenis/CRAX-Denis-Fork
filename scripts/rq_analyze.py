#!/usr/bin/env python3
"""Evaluate and aggregate the RQ experiments.

Step 1 `evaluate`: every saved checkpoint of every run is evaluated on the SAME
final-task environment (default: SafeBulletsHumanoid, level 3, 2 bullets), with
the SAME episodes/seeds, and the RAW environment reward (not the penalised reward
of the reward-engineering baseline). This makes curriculum runs (whose own
training-time evaluations are on easier stages) and direct runs comparable.

Step 2 `aggregate`: mean +- std over seeds per condition, bootstrap confidence
intervals of the difference to the reference condition, learning-curve plots and
a LaTeX table.

  python scripts/rq_analyze.py evaluate  --runs_root models/rq --out results/rq
  python scripts/rq_analyze.py aggregate --out results/rq --budget 0.1

Expected layout (written by scripts/rq_experiments.py):
  models/rq/<condition>/seed<k>/<run_name>/[stage_<i>/]<step>/
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIELDS = ["condition", "seed", "global_step", "ret", "cost", "hazard_cost",
          "fall_cost", "fall_rate", "episode_len"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def list_checkpoints(run_dir):
    """[(global_step, path)] for a run directory, with curriculum stage offsets."""
    stage_dirs = sorted((p for p in run_dir.glob("stage_*") if p.name[6:].isdigit()),
                        key=lambda p: int(p.name[6:]))
    groups = stage_dirs if stage_dirs else [run_dir]
    out, offset = [], 0
    for g in groups:
        steps = sorted((int(p.name), p) for p in g.iterdir()
                       if p.is_dir() and p.name.isdigit())
        for s, p in steps:
            out.append((offset + s, p))
        if steps:
            offset += steps[-1][0]
    return out


def find_runs(root):
    """[(condition, seed, run_dir)]"""
    runs = []
    for cond in sorted(p for p in Path(root).iterdir() if p.is_dir()):
        for sd in sorted(cond.glob("seed*")):
            for run in sorted(p for p in sd.iterdir() if p.is_dir()):
                runs.append((cond.name, int(sd.name[4:]), run))
    return runs


def make_rollout(env, policy, episode_len, n_episodes):
    """Vectorised rollout of `n_episodes` episodes of the unwrapped env.

    Per-episode sums stop accumulating at the first `done` (a fall: the episode
    terminates when the torso leaves the healthy height range or the platform).
    """
    import jax
    import jax.numpy as jnp

    v_reset = jax.jit(jax.vmap(env.reset))
    v_step = jax.jit(jax.vmap(env.step))
    v_pol = jax.jit(jax.vmap(policy))

    def run(rng):
        rng, k = jax.random.split(rng)
        state = v_reset(jax.random.split(k, n_episodes))
        alive = np.ones(n_episodes, dtype=bool)
        acc = {key: np.zeros(n_episodes) for key in
               ("ret", "cost", "hazard_cost", "fall_cost", "len")}
        fell = np.zeros(n_episodes, dtype=bool)
        for _ in range(episode_len):
            rng, k = jax.random.split(rng)
            action, _ = v_pol(state.obs, jax.random.split(k, n_episodes))
            state = v_step(state, action)
            m = alive.astype(np.float64)
            acc["ret"] += m * np.asarray(state.reward, dtype=np.float64)
            acc["cost"] += m * np.asarray(state.info["cost"], dtype=np.float64)
            acc["hazard_cost"] += m * (np.asarray(state.info["cost"], dtype=np.float64)
                                       - np.asarray(state.info["fall_cost"], dtype=np.float64))
            acc["fall_cost"] += m * np.asarray(state.info["fall_cost"], dtype=np.float64)
            acc["len"] += m
            done = np.asarray(state.done) > 0.5
            fell |= alive & done
            alive &= ~done
            if not alive.any():
                break
        acc["fell"] = fell.astype(np.float64)
        return acc
    return run


# ----------------------------------------------------------------------------
# step 1: evaluate
# ----------------------------------------------------------------------------
def cmd_evaluate(a):
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(REPO))
    import jax
    from crax import envs
    from training.agents.ppo import checkpoint as ppo_checkpoint

    env_kwargs = json.loads(a.env_kwargs)
    env = envs.get_environment(a.env_name, level=a.level, **env_kwargs)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curves = out_dir / "curves.csv"
    done_keys = set()
    if curves.exists() and not a.overwrite:
        with open(curves) as f:
            done_keys = {(r["condition"], int(r["seed"]), int(r["global_step"]))
                         for r in csv.DictReader(f)}
    new_file = not curves.exists() or a.overwrite
    f = open(curves, "w" if new_file else "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        w.writeheader()

    for cond, seed, run in find_runs(a.runs_root):
        ckpts = list_checkpoints(run)
        if a.final_only:
            ckpts = ckpts[-1:]
        for step, path in ckpts:
            if (cond, seed, step) in done_keys:
                continue
            policy = ppo_checkpoint.load_policy(path, deterministic=a.deterministic)
            rollout = make_rollout(env, policy, a.episode_len, a.episodes)
            acc = rollout(jax.random.PRNGKey(a.eval_seed))  # same episodes for all runs
            row = dict(condition=cond, seed=seed, global_step=step,
                       ret=acc["ret"].mean(), cost=acc["cost"].mean(),
                       hazard_cost=acc["hazard_cost"].mean(),
                       fall_cost=acc["fall_cost"].mean(),
                       fall_rate=acc["fell"].mean(), episode_len=acc["len"].mean())
            w.writerow(row)
            f.flush()
            print(f"{cond} seed{seed} step {step:>12,}: return {row['ret']:.1f} "
                  f"cost {row['cost']:.3f} fall {row['fall_rate']:.2f}", flush=True)
    f.close()


# ----------------------------------------------------------------------------
# step 2: aggregate
# ----------------------------------------------------------------------------
def load_curves(out_dir):
    rows = defaultdict(list)
    with open(Path(out_dir) / "curves.csv") as f:
        for r in csv.DictReader(f):
            rows[(r["condition"], int(r["seed"]))].append(
                {k: (float(v) if k not in ("condition",) else v) for k, v in r.items()})
    for v in rows.values():
        v.sort(key=lambda r: r["global_step"])
    return rows


def run_summary(curve, budget):
    """Per-run scalar metrics from one learning curve (list of rows)."""
    steps = np.array([r["global_step"] for r in curve])
    cost = np.array([r["cost"] for r in curve])
    last = curve[-1]
    feasible = cost <= budget
    first_ok = next((s for s, ok in zip(steps, feasible) if ok), np.nan)
    # first step from which the cost stays within the budget until the end
    stay = np.nan
    for i in range(len(steps)):
        if feasible[i:].all():
            stay = steps[i]
            break
    auc = getattr(np, "trapezoid", getattr(np, "trapz", None))(cost, steps) / max(steps[-1] - steps[0], 1) if len(steps) > 1 else cost[0]
    return dict(ret=last["ret"], cost=last["cost"], hazard_cost=last["hazard_cost"],
                fall_rate=last["fall_rate"], feasible=float(last["cost"] <= budget),
                steps_first_ok=first_ok, steps_stay_ok=stay, mean_cost_over_training=auc)


def bootstrap_diff(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x[~np.isnan(x)], y[~np.isnan(y)]
    if len(x) < 2 or len(y) < 2:
        return np.nan, np.nan, np.nan
    d = [rng.choice(x, len(x)).mean() - rng.choice(y, len(y)).mean() for _ in range(n)]
    return x.mean() - y.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def cmd_aggregate(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(a.out)
    runs = load_curves(out)
    per_cond = defaultdict(list)
    for (cond, seed), curve in runs.items():
        per_cond[cond].append((seed, curve, run_summary(curve, a.budget)))

    metrics = [("ret", "Return", "{:.0f}"), ("cost", "Episode cost", "{:.3f}"),
               ("fall_rate", "Fall rate", "{:.2f}"), ("feasible", "Feasible", "{:.2f}"),
               ("steps_stay_ok", "Steps to stay feasible", "{:.2e}"),
               ("mean_cost_over_training", "Mean cost over training", "{:.3f}")]

    # ---- table (final checkpoint, mean +- std over seeds) -------------------
    order = sorted(per_cond, key=lambda c: (c.split("_")[0], c))
    lines = []
    header = ["condition", "n"] + [m[1] for m in metrics]
    lines.append(header)
    for c in order:
        vals = [s for _, _, s in per_cond[c]]
        row = [c, str(len(vals))]
        for key, _, fmt in metrics:
            x = np.array([v[key] for v in vals], float)
            row.append("n/a" if np.isnan(x).all() else
                       f"{fmt.format(np.nanmean(x))} ± {fmt.format(np.nanstd(x))}")
        lines.append(row)
    with open(out / "summary.csv", "w", newline="") as f:
        csv.writer(f).writerows(lines)
    w = [max(len(r[i]) for r in lines) for i in range(len(header))]
    for r in lines:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))

    # ---- LaTeX table ----------------------------------------------------------
    tex = ["\\begin{tabular}{lcrrrr}", "\\toprule",
           "Condition & $n$ & Return & Cost & Fall rate & Feasible \\\\", "\\midrule"]
    for r in lines[1:]:
        tex.append(f"{r[0].replace('_', chr(92) + '_')} & {r[1]} & {r[2]} & {r[3]} & {r[4]} & {r[5]} \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "table.tex").write_text("\n".join(tex).replace("±", "$\\pm$") + "\n")

    # ---- pairwise comparisons to the reference (bootstrap 95% CI) ---------------
    pairs = [("RQ1: curriculum vs. final level only", "ours_pid", "b1_pid"),
             ("RQ1 (Lagrangian)", "ours_lag", "b1_lag"),
             ("Ablation: PID vs. Lagrangian (curriculum)", "ours_pid", "ours_lag"),
             ("Ablation: PID vs. Lagrangian (final level only)", "b1_pid", "b1_lag")]
    pairs += [(f"RQ2: constraint vs. penalty ({c})", "b1_pid", c) for c in order if c.startswith("b2_")]
    pairs += [(f"RQ2: constraint vs. penalty ({c}), both with curriculum", "ours_pid", c)
              for c in order if c.startswith("b3_")]
    res = []
    for title, ca, cb in pairs:
        if ca not in per_cond or cb not in per_cond:
            continue
        for key, label, _ in metrics[:4] + metrics[4:5]:
            xa = [s[key] for _, _, s in per_cond[ca]]
            xb = [s[key] for _, _, s in per_cond[cb]]
            d, lo, hi = bootstrap_diff(xa, xb)
            res.append([title, ca, cb, label, f"{d:.4g}", f"[{lo:.4g}, {hi:.4g}]"])
    with open(out / "comparisons.csv", "w", newline="") as f:
        csv.writer(f).writerows([["question", "A", "B", "metric", "mean(A)-mean(B)", "95% bootstrap CI"]] + res)
    print("\nPairwise comparisons written to", out / "comparisons.csv",
          "(only n >= 2 seeds per condition give a CI)")

    # ---- learning curves ----------------------------------------------------------
    def plot(metric, ylabel, fname, groups, hline=None):
        fig, ax = plt.subplots(figsize=(5.2, 3.2))
        for c in groups:
            if c not in per_cond:
                continue
            grid = np.unique(np.concatenate([[r["global_step"] for r in cv]
                                             for _, cv, _ in per_cond[c]]))
            ys = np.array([np.interp(grid, [r["global_step"] for r in cv],
                                     [r[metric] for r in cv]) for _, cv, _ in per_cond[c]])
            m, s = ys.mean(0), ys.std(0)
            ax.plot(grid, m, label=c)
            ax.fill_between(grid, m - s, m + s, alpha=0.15)
        if hline is not None:
            ax.axhline(hline, color="k", ls="--", lw=0.8, label="budget $d$")
        ax.set_xlabel("environment steps (total, all stages)")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=200)
        plt.close(fig)

    g1 = [c for c in order if c.startswith(("ours_", "b1_"))]
    g2 = [c for c in order if c in ("ours_pid", "b1_pid") or c.startswith(("b2_", "b3_"))]
    for tag, grp in (("rq1", g1), ("rq2", g2)):
        plot("ret", "Return (final task)", f"{tag}_return.png", grp)
        plot("cost", "Episode cost (final task)", f"{tag}_cost.png", grp, hline=a.budget)
        plot("fall_rate", "Fall rate (final task)", f"{tag}_fall_rate.png", grp)
    print("Plots written to", out)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("--runs_root", default="models/rq")
    e.add_argument("--out", default="results/rq")
    e.add_argument("--env_name", default="safe_bullets_humanoid")
    e.add_argument("--level", type=int, default=3)
    e.add_argument("--env_kwargs", default='{}',
                   help="JSON env kwargs of the final level (none needed for level 3)")
    e.add_argument("--episodes", type=int, default=128)
    e.add_argument("--episode_len", type=int, default=1000)
    e.add_argument("--eval_seed", type=int, default=12345)
    e.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    e.add_argument("--final_only", action="store_true", help="only the last checkpoint of each run")
    e.add_argument("--overwrite", action="store_true")
    g = sub.add_parser("aggregate")
    g.add_argument("--out", default="results/rq")
    g.add_argument("--budget", type=float, default=0.1,
                   help="episodic cost budget d used for 'feasible' (same as --safety_bound)")
    a = p.parse_args()
    {"evaluate": cmd_evaluate, "aggregate": cmd_aggregate}[a.cmd](a)


if __name__ == "__main__":
    main()