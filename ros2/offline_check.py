"""Offline test of the PPO loop with a fake simulator (no ROS, no torch).

  python ros2/offline_check.py --policy runs/<x>/policy.npz [--target 0.8 -1.18 0.49] [--q0 ...] [--csv out.csv]
  python ros2/offline_check.py --dummy        # random small network, only tests that the loop runs

The fake simulator moves q to the commanded position (clipped to the joint limits) and returns
fMe(q) as the "measured" tool pose. Writes q(t) to a CSV if --csv is given.
"""
import argparse
import os
import tempfile

import numpy as np

import ppo_core

ppo_core.add_src_to_path()
from kr16 import KR16, QMAX, QMIN  # noqa: E402


def make_dummy(path, obs_q=True, multiscale=False, seed=0, history=0):
    rng = np.random.default_rng(seed)
    sizes = [62 + 12 * history, 16, 16, 6]
    d = {"n_layers": np.array(3), "obs_q": np.array(obs_q), "multiscale_obs": np.array(multiscale),
         "history": np.array(history)}
    for i in range(3):
        d[f"W{i}"] = rng.normal(0, 0.3, (sizes[i + 1], sizes[i]))
        d[f"b{i}"] = np.zeros(sizes[i + 1])
    np.savez(path, **d)


def run(npz, xyz, tu, q0, timeout=10.0, tol=0.01):
    world = KR16()
    ctl = ppo_core.PPOController(npz, ppo_core.pose_matrix(xyz, tu))
    q, rows = np.array(q0, float), []
    for k in range(int(timeout / ppo_core.DT)):
        M = world.fMe(q)                       # "sensor"
        e_p, e_r = ctl.errors(M)
        rows.append([k * ppo_core.DT, *q, e_p])
        if e_p < tol:
            break
        q = ctl.next_q(q, ctl.step(q, M))      # fake simulator: perfect position tracking
    return np.array(rows), e_p, e_r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--target", type=float, nargs=3, default=ppo_core.TARGET_A[:3])
    ap.add_argument("--tu", type=float, nargs=3, default=ppo_core.TARGET_A[3:])
    ap.add_argument("--q0", type=float, nargs=6, default=[0, -0.5, 0.5, 0, 0.5, 0])
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--csv")
    a = ap.parse_args()
    path = a.policy
    if a.dummy or not path:
        path = os.path.join(tempfile.mkdtemp(), "dummy.npz")
        make_dummy(path)
    rows, e_p, e_r = run(path, a.target, a.tu, a.q0, a.timeout)
    q = rows[-1, 1:7]
    assert np.all(q >= QMIN - 1e-9) and np.all(q <= QMAX + 1e-9)
    print("steps %d, time %.2f s, final position error %.4f m, orientation error %.2f deg" % (
        len(rows), rows[-1, 0], e_p, np.rad2deg(e_r)))
    if a.csv:
        np.savetxt(a.csv, rows, delimiter=",", header="t,q1,q2,q3,q4,q5,q6,pos_err", comments="")
