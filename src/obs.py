"""Observation builder shared by the environment and the numpy-only policy (no torch, no gymnasium).

Layout of the observation vector (all entries roughly of order 1):
    [ 0: 6]  q mapped to [-1, 1] by the joint limits   (zeros when obs_q is False)
    [ 6: 9]  tool position p / 2 m
    [ 9:15]  tool orientation, first two columns of R
    [15:18]  target position p* / 2 m
    [18:24]  target orientation, first two columns of R*
    [24:27]  position error (p* - p) / 0.5 m, clipped to [-3, 3]
    [27:30]  orientation error theta*u of R* * R^T (base frame) / pi
    [30:36]  previous action
    [36:49]  multi-scale position block of e = p* - p     (zeros when multiscale is False)
             e / max(|e|, 1e-9) (3), log10(max(|e|, 1e-4)) / 4 (1), tanh(e / s) for s = 1, 0.1, 0.01 m (9)
    [49:62]  multi-scale orientation block of theta*u     (zeros when multiscale is False)
             same layout, with theta = |theta*u| and s = 1, 0.1, 0.01 rad
Optional history (history = h > 0): h entries of HIST_ENTRY = 12 values are appended after the 62 above, newest first
(see history_entry); zeros at the start of an episode.
Every entry is a measurement of the tool or the target; no model, Jacobian or wrist position.
"""
import numpy as np

from kr16 import QMAX, QMIN, tu_of

OBS_DIM = 62
HIST_ENTRY = 12   # values per past step in the optional history (see history_entry)
SCALES_POS = (1.0, 0.1, 0.01)   # m
SCALES_ROT = (1.0, 0.1, 0.01)   # rad


def orientation_error(R, R_star):
    """Rotation vector theta*u of R_star * R^T: the rotation that brings R onto R_star, in the base frame."""
    return tu_of(R_star @ R.T)


def multiscale_block(e, scales):
    """13 values for an error vector e: direction (3), log10 of the norm / 4 (1), tanh(e / s) per scale (9)."""
    n = float(np.linalg.norm(e))
    out = [e / max(n, 1e-9), [np.log10(max(n, 1e-4)) / 4.0]]
    out += [np.tanh(e / s) for s in scales]
    return np.concatenate(out)


def build_obs(q, M, M_star, prev_action, obs_q=True, multiscale=False):
    """Observation from joint angles q, tool pose M (4x4), target pose M_star (4x4), previous action."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    if obs_q:
        obs[0:6] = 2.0 * (q - QMIN) / (QMAX - QMIN) - 1.0
    p, R = M[:3, 3], M[:3, :3]
    ps, Rs = M_star[:3, 3], M_star[:3, :3]
    obs[6:9] = p / 2.0
    obs[9:15] = R[:, :2].T.reshape(-1)          # column 1 then column 2
    obs[15:18] = ps / 2.0
    obs[18:24] = Rs[:, :2].T.reshape(-1)
    obs[24:27] = np.clip((ps - p) / 0.5, -3.0, 3.0)
    obs[27:30] = orientation_error(R, Rs) / np.pi
    obs[30:36] = prev_action
    if multiscale:
        obs[36:49] = multiscale_block(ps - p, SCALES_POS)
        obs[49:62] = multiscale_block(orientation_error(R, Rs), SCALES_ROT)
    return obs


def obs_dim(history=0):
    """Length of the observation with `history` past steps appended."""
    return OBS_DIM + int(history) * HIST_ENTRY


def history_entry(M_prev, M, action):
    """12 values describing one control step: tool motion and the action that caused it.

    [0:3]  (p - p_prev) / 0.1 m, clipped to [-3, 3]
    [3:6]  theta*u of R * R_prev^T (base frame) / 0.1 rad, clipped to [-3, 3]
    [6:12] the action in [-1, 1]
    """
    e = np.empty(HIST_ENTRY, dtype=np.float32)
    e[0:3] = (M[:3, 3] - M_prev[:3, 3]) / 0.1
    e[3:6] = tu_of(M[:3, :3] @ M_prev[:3, :3].T) / 0.1
    e[0:6] = np.clip(e[0:6], -3.0, 3.0)
    e[6:12] = np.asarray(action, dtype=np.float32)
    return e
