"""Make the report figures from the training logs.

Example:
    python src/make_figures.py --runs T1_q_kl:"q, gSDE" T4_noq_kl:"no q" --tag batchT --baseline-success 0.70
    python src/make_figures.py --runs U1_q_s0:"q" --tag final --eval-json runs/eval_final.json
Outputs (PDF + PNG) in report/figures/: learning_<tag>, losses_<tag>; table_<tag>.tex if --eval-json is given.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(ROOT, "report", "figures")
COLORS = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]
SMOOTH = 10  # rolling-mean window for the loss curves (in logged iterations)
KL_TARGET = 0.02
XLABEL = {False: "timesteps (millions)", True: "normalised training time"}

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "legend.fontsize": 8, "xtick.labelsize": 7.5,
                     "ytick.labelsize": 7.5, "axes.titlesize": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "lines.linewidth": 1.2})


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(FIG_DIR, name + ".pdf"))


def read_csv(run, fname):
    path = os.path.join(ROOT, "runs", run, fname)
    if not os.path.exists(path):
        print(f"missing {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:  # file may be mid-write
        print(f"cannot read {path}: {e}")
        return None
    return df if len(df) else None


def parse_runs(items):
    runs = []
    for it in items:
        name, _, label = it.partition(":")
        runs.append((name, label or name))
    return runs


def x_axis(run, steps, normalise):
    """Timesteps in millions, or the fraction of the run's training (0 at its start, 1 at its end)."""
    if not normalise:
        return steps / 1e6
    prog = read_csv(run, "progress.csv")
    ts = prog["time/total_timesteps"].to_numpy()
    start = ts[0] - (ts[1] - ts[0])          # warm starts begin above 0
    with open(os.path.join(ROOT, "runs", run, "config.json")) as f:
        total = json.load(f)["steps"]
    return (steps - start) / total


def learning_figure(runs, tag, baseline, normalise=False):
    fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for i, (name, label) in enumerate(runs):
        df = read_csv(name, "eval.csv")
        if df is None:
            continue
        x = x_axis(name, df["timesteps"], normalise)
        ax[0].plot(x, 100 * df["success_rate"], color=COLORS[i % len(COLORS)], label=label)
        ax[1].plot(x, 100 * df["median_pos_err"], color=COLORS[i % len(COLORS)], label=label)
    if baseline is not None:
        ax[0].axhline(100 * baseline, color="k", ls="--", lw=1, label="Jacobian control (Ex 6)")
    ax[0].set_ylabel("success rate at 1 cm (%)")
    ax[0].set_ylim(-3, 103)
    ax[1].set_ylabel("median final position error (cm)")
    ax[1].set_yscale("log")
    ax[1].axhline(1.0, color="gray", ls=":", lw=1)
    for a in ax:
        a.set_xlabel(XLABEL[normalise])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    save(fig, f"learning_{tag}")


def losses_figure(runs, tag, normalise=False):
    fig, axs = plt.subplots(2, 3, figsize=(7, 4.2))
    ax = axs.ravel()
    panels = [("train/value_loss", "value loss", True), ("train/policy_gradient_loss", "policy grad. loss", False),
              ("train/entropy_loss", "entropy loss", False), ("train/approx_kl", "approx. KL", False),
              ("train/clip_fraction", "clip fraction", False)]
    any_std = False
    for i, (name, label) in enumerate(runs):
        df = read_csv(name, "progress.csv")
        if df is None or "time/total_timesteps" not in df:
            continue
        c = COLORS[i % len(COLORS)]
        x = x_axis(name, df["time/total_timesteps"], normalise)
        for a, (col, _, _) in zip(ax[:5], panels):
            if col in df:
                a.plot(x, df[col].rolling(SMOOTH, min_periods=1).mean(), color=c, label=label)
        if "train/std" in df and df["train/std"].notna().any():
            any_std = True
            ax[5].plot(x, df["train/std"].rolling(SMOOTH, min_periods=1).mean(), color=c, label=label)
        elif "train/explained_variance" in df:
            ax[5].plot(x, df["train/explained_variance"].rolling(SMOOTH, min_periods=1).mean(), color=c, label=label)
    for a, (_, t, logy) in zip(ax[:5], panels):
        a.set_ylabel(t + f" (avg {SMOOTH})")
        if logy:
            a.set_yscale("log")
    ax[3].axhline(KL_TARGET, color="k", ls="--", lw=1, label="target KL")
    ax[5].set_ylabel(("action std" if any_std else "explained variance") + f" (avg {SMOOTH})")
    for a in ax:
        a.set_xlabel(XLABEL[normalise])
        a.yaxis.label.set_size(7.5)
    h, l = ax[3].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=min(len(l), 3), frameon=False)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    save(fig, f"losses_{tag}")


def table_from_json(path, tag):
    with open(path) as f:
        res = json.load(f)["results"]
    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             r"method & success (\%) & pos.\ median (cm) & pos.\ p90 (cm) & rot.\ median ($^\circ$) & steps \\",
             r"\hline"]
    for k, m in res.items():
        name = k.replace("_", r"\_")
        steps = "--" if np.isnan(m["steps_to_reach"]) else f"{m['steps_to_reach']:.1f}"
        lines.append(f"{name} & {100 * m['success_rate']:.1f} & {100 * m['pos_err_median']:.2f} & "
                     f"{100 * m['pos_err_p90']:.2f} & {m['rot_err_deg_median']:.1f} & {steps} \\\\")
    lines += [r"\hline", r"\end{tabular}", ""]
    os.makedirs(FIG_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f"table_{tag}.tex")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="NAME:LABEL (run dir under runs/)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--baseline-success", type=float, default=None)
    ap.add_argument("--eval-json", default=None, help="runs/eval_<tag>.json from evaluate.py -> table_<tag>.tex")
    ap.add_argument("--normalise", action="store_true", help="x axis = fraction of each run's training (0 to 1)")
    args = ap.parse_args()
    runs = parse_runs(args.runs)
    learning_figure(runs, args.tag, args.baseline_success, args.normalise)
    losses_figure(runs, args.tag, args.normalise)
    if args.eval_json:
        table_from_json(args.eval_json, args.tag)


if __name__ == "__main__":
    main()
