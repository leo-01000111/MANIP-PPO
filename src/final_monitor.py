"""Live view of the final chained runs (position stage, then pose stage) in one window, no restart needed.

Each chain NAME has two run folders: runs/NAME_pos and runs/NAME_pose (the second appears when the pose stage starts).
Both stages are drawn on the same axes (the step count continues across the warm start); a dashed vertical line
marks the switch. Eval panels: success / errors are position-only before the line and full pose after it.
    python src/final_monitor.py F_q_s0 F_q_s1 F_noqmem_s0 F_noqmem_s1
"""
import argparse
import datetime as dt
import os

import matplotlib.pyplot as plt
import pandas as pd

from live_monitor import PANELS, SCALE, STYLES

POS_STEPS, POSE_STEPS = 18_000_000, 12_000_000     # as in final_chain.sh


def read_stage(run):
    d = os.path.join("runs", run)
    out = []
    for f in ("progress.csv", "eval.csv"):
        try:
            out.append(pd.read_csv(os.path.join(d, f)))
        except Exception:  # noqa: not there yet, or being written
            out.append(None)
    return out


def chain_status(name, stages):
    """Overall progress over both stages, elapsed time and predicted finish of the whole chain."""
    done, elapsed, rate, stage = 0, 0.0, None, "position"
    for (prog, _), todo, label in zip(stages, (POS_STEPS, POSE_STEPS), ("position", "pose")):
        if prog is None or len(prog) < 3 or "time/total_timesteps" not in prog:
            continue
        ts, el = prog["time/total_timesteps"].to_numpy(), prog["time/time_elapsed"].to_numpy()
        start = ts[0] - (ts[1] - ts[0])
        done += min(ts[-1] - start, todo)
        elapsed += el[-1]
        rate = (ts[-1] - ts[0]) / max(el[-1] - el[0], 1e-6)
        stage = label
    if rate is None:
        return f"{name}: starting ..."
    total = POS_STEPS + POSE_STEPS
    eta = dt.datetime.now() + dt.timedelta(seconds=max(total - done, 0) / max(rate, 1e-6))
    return (f"{name}: {stage:8s} stage, {done / 1e6:5.2f} / {total / 1e6:.0f} M steps ({100 * done / total:4.1f} %), "
            f"elapsed {elapsed / 3600:4.1f} h, {rate:4.0f} steps/s, all done ~{eta:%H:%M}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("chains", nargs="+")
    ap.add_argument("--every", type=float, default=30.0, help="refresh period (s)")
    ap.add_argument("--smooth", type=int, default=10, help="rolling mean over this many updates")
    a = ap.parse_args()

    plt.ion()
    fig, axes = plt.subplots(3, 3, figsize=(14, 9.5))
    plt.show(block=False)                       # show once; later refreshes do not bring it to the front
    while plt.fignum_exists(fig.number):
        lines, switches = [], []
        for ax in axes.flat:
            ax.clear()
        for i, name in enumerate(a.chains):
            stages = [read_stage(name + "_pos"), read_stage(name + "_pose")]
            lines.append(chain_status(name, stages))
            pose_prog = stages[1][0]
            if pose_prog is not None and len(pose_prog) > 1 and "time/total_timesteps" in pose_prog:
                ts = pose_prog["time/total_timesteps"]
                switches.append((ts.iloc[0] - (ts.iloc[1] - ts.iloc[0])) / 1e6)
            for ax, (src, col, title, logy) in zip(axes.flat, PANELS):
                parts = []
                for prog, ev in stages:
                    df = ev if src == "eval" else prog
                    if df is None or col not in df:
                        continue
                    x = df["timesteps"] if src == "eval" else df["time/total_timesteps"]
                    y = df[col] if src == "eval" else df[col].rolling(a.smooth, min_periods=1).mean()
                    parts.append((x / 1e6, y * SCALE.get(col, 1.0)))
                for k, (x, y) in enumerate(parts):      # each stage its own segment, same colour and style
                    ax.plot(x, y, color=f"C{i}", lw=1.6 - 0.2 * i, ls=STYLES[i % len(STYLES)],
                            label=name if k == 0 else None)
                ax.set_title(title, fontsize=9)
                ax.set_yscale("log" if logy else "linear")
                ax.grid(alpha=0.3)
                ax.tick_params(labelsize=8)
        for ax in axes.flat:                            # the switch position -> pose (one line per chain that switched)
            for s in sorted(set(round(v, 2) for v in switches)):
                ax.axvline(s, color="k", ls="--", lw=1.2, alpha=0.7)
        if switches:
            axes.flat[0].text(min(switches), axes.flat[0].get_ylim()[1], " pose stage ->", fontsize=8,
                              va="top", ha="left")
        axes.flat[0].legend(fontsize=7, loc="upper left")
        for ax in axes[-1]:
            ax.set_xlabel("timesteps (M)", fontsize=8)
        fig.suptitle("\n".join(lines) + f"\nupdated {dt.datetime.now():%H:%M:%S}   "
                     "(left of the dashed line: position task; right: full pose, 1 cm + 5 deg)",
                     fontsize=9, family="monospace")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.canvas.draw_idle()                  # redraw without raising the window
        fig.canvas.start_event_loop(a.every)


if __name__ == "__main__":
    main()
