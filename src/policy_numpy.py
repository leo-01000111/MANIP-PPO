"""Pure-numpy deterministic policy (no torch): runs in the ROS2 VM.

The weights come from export_numpy.py. `build_obs` is the same function the environment uses
(obs.py), re-exported here so that the VM needs only this file, obs.py and kr16.py.
"""
import numpy as np

from obs import HIST_ENTRY, OBS_DIM, build_obs, history_entry, obs_dim, orientation_error  # noqa: F401  (re-exported)


class NumpyPolicy:
    """Actor mean of the SB3 MlpPolicy (also with gSDE: the deterministic action is action_net(latent_pi)): tanh hidden layers, then a linear action layer, clipped to [-1, 1]."""

    def __init__(self, npz_path):
        d = np.load(npz_path)
        n = int(d["n_layers"])
        self.W = [d[f"W{i}"] for i in range(n)]       # each W has shape (out, in), like torch
        self.b = [d[f"b{i}"] for i in range(n)]
        # observation settings of the training run (use them when calling build_obs)
        self.obs_q = bool(d["obs_q"]) if "obs_q" in d else True
        self.multiscale_obs = bool(d["multiscale_obs"]) if "multiscale_obs" in d else False
        self.history = int(d["history"]) if "history" in d else 0     # past steps appended to the observation

    def build_obs(self, q, M, M_star, prev_action):
        """Observation with the settings this policy was trained with."""
        return build_obs(q, M, M_star, prev_action, self.obs_q, self.multiscale_obs)

    def act(self, obs):
        x = np.asarray(obs, dtype=np.float64)
        for W, b in zip(self.W[:-1], self.b[:-1]):
            x = np.tanh(W @ x + b)
        return np.clip(self.W[-1] @ x + self.b[-1], -1.0, 1.0).astype(np.float32)
