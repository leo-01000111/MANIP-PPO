"""Live view of running trainings: losses, success rate, steps, elapsed time and predicted finish time.

Reads runs/<name>/progress.csv, eval.csv and config.json every `--every` seconds (read-only, safe while training).
    python src/live_monitor.py P1_pose_keypoint P2_pose_rotcur P3_pose_restricted
Close the window or press Ctrl+C to stop. The trainings are not affected.
"""
import argparse
import datetime as dt
import json
import os

import matplotlib.pyplot as plt
import pandas as pd

PANELS = [("eval", "success_rate", "success rate (eval)", False),
          ("prog", "train/value_loss", "value loss", True),
          ("prog", "train/policy_gradient_loss", "policy loss", False),
          ("prog", "train/entropy_loss", "entropy loss", False),
          ("prog", "train/approx_kl", "approx. KL", False),
          ("prog", "train/clip_fraction", "clip fraction", False),
          ("prog", "train/std", "action std", False),
          ("eval", "median_pos_err", "median final pos. error (m)", True),
          ("eval", "mean_rot_err", "mean final orientation error (deg)", False)]
STYLES = ["-", "--", ":", "-."]          # one line style per run, so lines on top of each other stay visible
SCALE = {"mean_rot_err": 57.29577951308232}   # rad -> deg


def read(run):
    d = os.path.join("runs", run)
    try:
        prog = pd.read_csv(os.path.join(d, "progress.csv"))
    except Exception:  # noqa: file missing or being written
        prog = None
    try:
        ev = pd.read_csv(os.path.join(d, "eval.csv"))
    except Exception:  # noqa
        ev = None
    cfg = json.load(open(os.path.join(d, "config.json")))
    return prog, ev, cfg


def status(run, prog, cfg):
    """Steps done / to do, elapsed time and predicted finish (from the average step rate of the run)."""
    if prog is None or len(prog) < 3 or "time/total_timesteps" not in prog:
        return f"{run}: starting ..."
    ts, el = prog["time/total_timesteps"].to_numpy(), prog["time/time_elapsed"].to_numpy()
    start = ts[0] - (ts[1] - ts[0])                     # step count when this run began (warm starts begin > 0)
    done, todo = ts[-1] - start, cfg["steps"]
    rate = (ts[-1] - ts[0]) / max(el[-1] - el[0], 1e-6)  # steps per second, whole run (includes eval pauses)
    left = max(todo - done, 0) / max(rate, 1e-6)
    eta = (dt.datetime.now() + dt.timedelta(seconds=left)).strftime("%H:%M")
    return (f"{run}: {done / 1e6:5.2f} / {todo / 1e6:.0f} M steps ({100 * done / todo:4.1f} %), "
            f"elapsed {el[-1] / 3600:4.1f} h, {rate:5.0f} steps/s, finish ~{eta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--every", type=float, default=30.0, help="refresh period (s)")
    ap.add_argument("--smooth", type=int, default=10, help="rolling mean over this many updates")
    a = ap.parse_args()

    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
    plt.show(block=False)                       # show once; later refreshes do not bring it to the front
    while plt.fignum_exists(fig.number):
        lines = []
        for ax in axes.flat:
            ax.clear()
        for i, run in enumerate(a.runs):
            prog, ev, cfg = read(run)
            lines.append(status(run, prog, cfg))
            for ax, (src, col, title, logy) in zip(axes.flat, PANELS):
                df = ev if src == "eval" else prog
                if df is None or col not in df or "timesteps" not in "".join(df.columns):
                    continue
                x = df["timesteps"] if src == "eval" else df["time/total_timesteps"]
                y = df[col] if src == "eval" else df[col].rolling(a.smooth, min_periods=1).mean()
                y = y * SCALE.get(col, 1.0)
                ax.plot(x / 1e6, y, color=f"C{i}", label=run, lw=1.6 - 0.2 * i, ls=STYLES[i % len(STYLES)])
                ax.set_title(title, fontsize=9)
                ax.set_yscale("log" if logy else "linear")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=8)
        axes.flat[0].legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("timesteps (M)", fontsize=8)
        fig.suptitle("\n".join(lines) + f"\nupdated {dt.datetime.now():%H:%M:%S}", fontsize=9, family="monospace")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.canvas.draw_idle()                  # redraw without raising the window (plt.pause steals the focus)
        fig.canvas.start_event_loop(a.every)


if __name__ == "__main__":
    main()
