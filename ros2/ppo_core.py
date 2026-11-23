"""ROS-free core of the PPO reaching loop, shared by ppo_reach_node.py and offline_check.py.

Needs only numpy and the files src/kr16.py, src/obs.py, src/policy_numpy.py (no torch, no gymnasium).
"""
import os
import sys

import numpy as np

DT = 0.05            # s, control period used in training
SPEED_SCALE = 0.5    # training: qdot = a * SPEED_SCALE * VMAX

# Lab target A (robot_kr16.h): x y z, then the vector theta*u (rad)
TARGET_A = [0.812110097, -1.187465983, 0.4915271277, 0.9423584377, -2.852157604, 0.1075927581]


def add_src_to_path(src_dir=""):
    """Make `import kr16, obs, policy_numpy` work. Default: ../src next to this folder, then ./src."""
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ([src_dir] if src_dir else []) + [os.path.join(here, "..", "src"), os.path.join(here, "src")]:
        if d and os.path.exists(os.path.join(d, "policy_numpy.py")):
            sys.path.insert(0, os.path.abspath(d))
            return os.path.abspath(d)
    raise FileNotFoundError("cannot find src/policy_numpy.py; set the src_dir parameter")


def pose_matrix(xyz, tu):
    from kr16 import rot_tu
    M = np.eye(4)
    M[:3, :3] = rot_tu(np.asarray(tu, float))
    M[:3, 3] = xyz
    return M


class PPOController:
    """One call of `step` = one control tick: measurements in, joint velocity (rad/s) out."""

    def __init__(self, npz_path, M_star):
        from kr16 import VMAX, QMIN, QMAX
        from policy_numpy import NumpyPolicy
        self.pol = NumpyPolicy(npz_path)
        self.M_star = np.asarray(M_star, float)
        self.prev_a = np.zeros(6, dtype=np.float32)
        self.vmax, self.qmin, self.qmax = VMAX, QMIN, QMAX

    def errors(self, M):
        from obs import orientation_error
        e_p = float(np.linalg.norm(self.M_star[:3, 3] - M[:3, 3]))
        e_r = float(np.linalg.norm(orientation_error(M[:3, :3], self.M_star[:3, :3])))
        return e_p, e_r

    def step(self, q, M):
        """q: measured joints (6,), M: measured tool pose (4x4). Returns qdot (6,) in rad/s."""
        o = self.pol.build_obs(np.asarray(q, float), M, self.M_star, self.prev_a)
        a = self.pol.act(o)
        self.prev_a = a
        return a.astype(float) * SPEED_SCALE * self.vmax

    def next_q(self, q, qdot):
        """Position setpoint for one period, inside the joint limits (same as the training env)."""
        return np.clip(np.asarray(q, float) + qdot * DT, self.qmin, self.qmax)
