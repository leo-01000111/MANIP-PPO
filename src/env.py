"""Gymnasium environment: reach a target pose with the Kuka KR16, kinematics hidden from the agent."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from kr16 import KR16
from obs import HIST_ENTRY, build_obs, history_entry, obs_dim, orientation_error

EPS_P = 0.001            # m, floor of the position error in the log reward
EPS_R = np.deg2rad(0.1)  # rad, floor of the orientation error in the log reward


class KR16ReachEnv(gym.Env):
    """The agent commands joint velocities and sees only measurements (see obs.py for the layout).

    Action  : a in [-1, 1]^6, qdot = a * speed_scale * VMAX, q_next = clip(q + qdot * dt, qmin, qmax).
    Observation (62 values, plus 12 per step of `history` if > 0 (tool motion and action of the last steps, newest first); same size for every task and for obs_q True/False):
        q in [-1, 1] (zeros if obs_q is False), tool p and R columns, target p* and R* columns,
        position error, orientation error theta*u of R* * R^T, previous action, then a multi-scale block
        (26 values, zeros if multiscale_obs is False).
    Reward  : -|p - p*| - w_rot * |theta*u error| - 0.01 * |a|^2, plus 1.0 and termination on success.
            reward_mode "progress": 10 * decrease of the error per step.
            reward_mode "log_progress": potential-based shaping with potential -log(d), so halving the
            distance pays the same at 1 m and at 1 cm (errors floored at 1 mm and 0.1 deg).
    Success : |p - p*| < pos_tol (task "position"), and also |theta*u error| < rot_tol (task "pose").
    Reset   : start q and target joint vector q* are uniform in the joint limits shrunk by 10 % towards
              their middle; the target pose is fMe(q*), so it is always reachable.
    `world` is the hidden KR16 model; only the environment and the Jacobian baseline use it.
    """
    metadata = {"render_modes": []}

    def __init__(self, obs_q=True, task="position", w_rot=0.0, a3=0.68, dt=0.05, max_steps=100,
                 speed_scale=0.5, pos_tol=0.01, rot_tol=np.deg2rad(5), reward_mode="distance", multiscale_obs=False,
                 history=0, limit_penalty=0.0, time_penalty=0.0):
        assert task in ("position", "pose") and reward_mode in ("distance", "progress", "log_progress")
        self.multiscale_obs = multiscale_obs
        self.history = int(history)
        self.time_penalty = float(time_penalty)     # reward cost per step (favours reaching sooner)
        self.limit_penalty = float(limit_penalty)   # reward cost per joint pushed against its limit in a step
        self.reward_mode = reward_mode
        self.obs_q, self.task, self.w_rot = obs_q, task, w_rot
        self.dt, self.max_steps, self.speed_scale = dt, max_steps, speed_scale
        self.pos_tol, self.rot_tol = pos_tol, rot_tol
        self.world = KR16(a3)
        mid = (self.world.qmax + self.world.qmin) / 2
        half = 0.9 * (self.world.qmax - self.world.qmin) / 2
        self._sample_lo, self._sample_hi = mid - half, mid + half
        self.obs_dim = obs_dim(self.history)
        self.hist_buf = np.zeros((self.history, HIST_ENTRY), dtype=np.float32)
        self.observation_space = spaces.Box(-5.0, 5.0, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(6,), dtype=np.float32)

    def set_w_rot(self, w_rot):
        """Change the orientation weight (curriculum); callable through VecEnv.env_method."""
        self.w_rot = float(w_rot)

    def set_pos_tol(self, pos_tol):
        """Change the position tolerance (accuracy curriculum); callable through VecEnv.env_method."""
        self.pos_tol = float(pos_tol)

    def _sample_q(self):
        return self.np_random.uniform(self._sample_lo, self._sample_hi)

    def _errors(self):
        """Position error norm and orientation error norm between tool and target."""
        pos_err = float(np.linalg.norm(self.M[:3, 3] - self.M_star[:3, 3]))
        rot_err = float(np.linalg.norm(orientation_error(self.M[:3, :3], self.M_star[:3, :3])))
        return pos_err, rot_err

    def _obs(self):
        obs = build_obs(self.q, self.M, self.M_star, self.prev_action, self.obs_q, self.multiscale_obs)
        if self.history:
            obs = np.concatenate([obs, self.hist_buf.reshape(-1)])
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.q = self._sample_q()
        self.M = self.world.fMe(self.q)
        self.M_star = self.world.fMe(self._sample_q())
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.hist_buf = np.zeros((self.history, HIST_ENTRY), dtype=np.float32)
        self.steps = 0
        pos_err, rot_err = self._errors()
        self._prev_err = (pos_err, rot_err)
        return self._obs(), {"pos_err": pos_err, "rot_err": rot_err, "is_success": False, "q": self.q.copy()}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        qdot = a * self.speed_scale * self.world.vmax
        q_raw = self.q + qdot * self.dt
        self.q = np.clip(q_raw, self.world.qmin, self.world.qmax)
        n_at_limit = int(np.sum(q_raw != self.q))   # joints the command tried to push past a limit
        M_prev, self.M = self.M, self.world.fMe(self.q)
        if self.history:      # newest entry first
            self.hist_buf = np.roll(self.hist_buf, 1, axis=0)
            self.hist_buf[0] = history_entry(M_prev, self.M, a)
        self.prev_action = a.astype(np.float32)
        self.steps += 1

        pos_err, rot_err = self._errors()
        success = pos_err < self.pos_tol and (self.task == "position" or rot_err < self.rot_tol)
        if self.reward_mode == "distance":
            reward = -pos_err - self.w_rot * rot_err
        elif self.reward_mode == "log_progress":
            lp = lambda x, eps: np.log(max(x, eps))
            reward = (lp(self._prev_err[0], EPS_P) - lp(pos_err, EPS_P)
                      + self.w_rot * (lp(self._prev_err[1], EPS_R) - lp(rot_err, EPS_R)))
        else:
            # progress: reward the decrease of the error in this step (potential-based shaping, 10 per m)
            reward = 10.0 * ((self._prev_err[0] - pos_err) + self.w_rot * (self._prev_err[1] - rot_err))
        self._prev_err = (pos_err, rot_err)
        reward += -0.01 * float(a @ a) + (1.0 if success else 0.0) - self.limit_penalty * n_at_limit - self.time_penalty
        truncated = (not success) and self.steps >= self.max_steps
        info = {"pos_err": pos_err, "rot_err": rot_err, "is_success": bool(success), "q": self.q.copy()}
        return self._obs(), reward, bool(success), truncated, info
