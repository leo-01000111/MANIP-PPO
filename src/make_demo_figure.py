"""Frames of the RViz demo for the report (LaTeX `animate`), plus a still first frame that browsers show.

Frame 0 (the poster) is a graph of the tool position in time: the PPO policy as recorded in the lab simulator
(joint states -> fMe), and the Jacobian controller of Ex 6 simulated on the same model from the same start to the
same target (same speed limit, same 0.05 s step). The next frames are RViz screenshots of the recorded motion.

    python src/make_demo_figure.py JS_CSV FRAMES_DIR --target 0.503 -1.393 1.247
Outputs report/figures/demo/f0.png (graph) and f1.jpg ... (screenshots cropped and downscaled).
"""
import argparse
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from baseline import JacobianController  # noqa: E402
from kr16 import KR16, QMAX, QMIN, VMAX  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures", "demo")
CROP = (16, 104, 2550, 1500)        # RViz 3D view inside the 2560 x 1600 screenshot
WIDTH = 960                          # output frame width (px)
DT, TOL = 0.05, 0.01


def jacobian_run(q0, target, t_end):
    world, ctrl = KR16(), JacobianController(KR16(), "position")
    M_star = np.eye(4)
    M_star[:3, 3] = target
    q, ts, ps = np.array(q0, float), [0.0], [world.fMe(q0)[:3, 3]]
    reached = None
    while ts[-1] < t_end:
        if reached is None and np.linalg.norm(ps[-1] - target) < TOL:
            reached = ts[-1]
        a = ctrl.act(q, M_star) if reached is None else np.zeros(6)
        q = np.clip(q + a * ctrl.speed_scale * VMAX * DT, QMIN, QMAX)
        ts.append(ts[-1] + DT)
        ps.append(world.fMe(q)[:3, 3])
    return np.array(ts), np.array(ps), reached


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("js_csv")
    ap.add_argument("frames")
    ap.add_argument("--target", type=float, nargs=3, required=True)
    a = ap.parse_args()
    target = np.array(a.target)

    d = np.loadtxt(a.js_csv, delimiter=",")
    t, q = d[:, 0] - d[0, 0], d[:, 1:7]
    moving = np.where(np.abs(np.diff(q, axis=0)).sum(axis=1) > 1e-4)[0]
    keep = t >= t[moving[0]] - 0.3                   # same start as the replayed frames
    t, q = t[keep] - t[moving[0]], q[keep]           # time 0 = the arm starts moving
    world = KR16()
    p_ppo = np.array([world.fMe(x)[:3, 3] for x in q])
    err = np.linalg.norm(p_ppo - target, axis=1)
    t_ppo = t[np.argmax(err < TOL)] if (err < TOL).any() else None
    t_jac, p_jac, t_j = jacobian_run(q[0], target, t[-1])
    t_end = max(v for v in (t_ppo, t_j, 1.0) if v is not None) + 1.0
    sel = t <= t_end
    t, p_ppo = t[sel], p_ppo[sel]
    sel = t_jac <= t_end
    t_jac, p_jac = t_jac[sel], p_jac[sel]

    os.makedirs(OUT, exist_ok=True)
    for f in glob.glob(os.path.join(OUT, "f*.*")):
        os.remove(f)
    w, h = CROP[2] - CROP[0], CROP[3] - CROP[1]
    size = (WIDTH, round(WIDTH * h / w))
    fig, ax = plt.subplots(1, 3, figsize=(size[0] / 150, size[1] / 150), dpi=150, sharex=True)
    for i, c in enumerate("xyz"):
        ax[i].plot(t, p_ppo[:, i], color="#1f77b4", lw=1.6, label="PPO (lab simulator)")
        ax[i].plot(t_jac, p_jac[:, i], color="#d62728", lw=1.4, ls="--", label="Jacobian control (Ex 6)")
        ax[i].axhline(target[i], color="k", ls=":", lw=1, label="target")
        ax[i].set_title(f"tool {c} (m)", fontsize=9)
        ax[i].set_xlabel("time (s)", fontsize=8)
        ax[i].tick_params(labelsize=7)
        ax[i].grid(alpha=0.3)
    h_, l_ = ax[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=8)
    note = [f"PPO reached 1 cm at {t_ppo:.1f} s" if t_ppo is not None else "PPO: not reached",
            f"Jacobian at {t_j:.1f} s" if t_j is not None else "Jacobian: not reached"]
    fig.suptitle(",   ".join(note) + "   (Adobe Reader: click to play the RViz replay)", fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(os.path.join(OUT, "f0.png"), dpi=150)
    plt.close(fig)
    print("poster:", note)

    shots = sorted(glob.glob(os.path.join(a.frames, "f*.png")))
    for k, s in enumerate(shots, start=1):
        Image.open(s).convert("RGB").crop(CROP).resize(size, Image.Resampling.LANCZOS).save(os.path.join(OUT, f"f{k}.jpg"), quality=88)
    print("frames", len(shots) + 1, "size", size, "->", OUT)


if __name__ == "__main__":
    main()
