"""Lab Jacobian velocity controller (Exercise 6): the "full knowledge" reference.

Unlike the PPO agent, it uses a kinematic model of the robot (fMe and the Jacobian), here `model`.
The model may differ from the world of the environment (wrong link length a3 in the robustness test).
"""
import numpy as np

from kr16 import KR16, VMAX, tu_of


class JacobianController:
    """qdot = pinv(J) * v with v proportional to the pose error, scaled to the env's action range."""

    def __init__(self, model: KR16, task="position", lam=8.0, speed_scale=0.5, vmax=VMAX):
        self.model, self.task, self.lam = model, task, lam
        self.speed_scale, self.vmax = speed_scale, np.asarray(vmax)

    def act(self, q, target_pose, tool_pose=None):
        """Action in [-1, 1]^6 for joint angles q (encoders) and target pose (4x4, base frame).

        By default the tool pose comes from the controller's model, fMe(q), as in the lab.
        `tool_pose` (a measured pose) replaces it: then the model only provides the Jacobian.
        """
        M = self.model.fMe(q) if tool_pose is None else tool_pose
        J = self.model.fJe(q)
        if self.task == "position":
            v = self.lam * (target_pose[:3, 3] - M[:3, 3])
            qdot = np.linalg.pinv(J[:3]) @ v
        else:
            # lab law: E = Md^-1 * M, v = -lam * Rd * t, w = -lam * R * theta*u
            E = np.linalg.inv(target_pose) @ M
            v = np.concatenate([-self.lam * target_pose[:3, :3] @ E[:3, 3],
                                -self.lam * M[:3, :3] @ tu_of(E[:3, :3])])
            qdot = np.linalg.pinv(J) @ v
        a = qdot / (self.speed_scale * self.vmax)
        peak = np.max(np.abs(a))
        if peak > 1.0:                              # scale the whole vector: keeps the direction
            a = a / peak
        return a.astype(np.float32)
