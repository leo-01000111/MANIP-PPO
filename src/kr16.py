"""Kuka KR16 forward kinematics: the hidden "world" of the reaching environment.

Ported from the validated lab code (reference/lab_python/make_velocity_figures.py).
The agent never calls this module; only the environment (to move the arm) and the
Jacobian baseline (as its own model of the robot) do.

Convention: modified Denavit-Hartenberg (MDH), frame i-1 -> i is
    Rot(x, alpha) * Trans(x, a) * Rot(z, theta) * Trans(z, r).
"""
from math import acos, cos, pi, sin

import numpy as np

# Fixed tool offset: wMe = Rot(x, pi) * Trans(z, 0.158)
WME = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -0.158], [0, 0, 0, 1.0]])
# Joint velocity limits (rad/s) and joint position limits (rad)
VMAX = np.array([2.722713633111154] * 3 + [5.759586531581287] * 2 + [10.733774899765127])
QMAX = np.array([3.2289, 0.6109, 2.6878, 6.1087, 2.2689, 6.1087])
QMIN = np.array([-3.2289, -2.7053, -2.2689, -6.1087, -2.2689, -6.1087])


def mdh(alpha, a, theta, r):
    """Homogeneous transform of one MDH link (closed form of the product of 4 elementary moves)."""
    ca, sa, ct, st = cos(alpha), sin(alpha), cos(theta), sin(theta)
    return np.array([[ct, -st, 0.0, a],
                     [ca * st, ca * ct, -sa, -sa * r],
                     [sa * st, sa * ct, ca, ca * r],
                     [0.0, 0.0, 0.0, 1.0]])


def skew(u):
    """Matrix of the cross product: skew(u) * x = u x x."""
    return np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])


def rot_tu(tu):
    """Rotation matrix from the vector theta*u (Rodrigues formula)."""
    th = np.linalg.norm(tu)
    if th < 1e-12:
        return np.eye(3)
    K = skew(np.asarray(tu) / th)
    return np.eye(3) + sin(th) * K + (1 - cos(th)) * (K @ K)


def tu_of(R):
    """Vector theta*u of a rotation matrix (inverse of rot_tu), theta in [0, pi]."""
    th = acos(min(1.0, max(-1.0, (R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2)))
    if th < 1e-12:
        return np.zeros(3)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if pi - th < 1e-6:
        # near theta = pi the antisymmetric part vanishes: take the axis from the symmetric part
        B = (R + np.eye(3)) / 2
        i = int(np.argmax(np.diag(B)))
        u = B[:, i] / np.sqrt(B[i, i])
        if u @ w < 0:       # keep the sign given by the antisymmetric part
            u = -u
        return th * u
    return th * w / (2 * sin(th))


class KR16:
    """Forward kinematics and Jacobian of the KR16. `a3` is the length of link 3 (nominal 0.68 m)."""

    def __init__(self, a3=0.68):
        self.a3 = a3
        self.qmin, self.qmax, self.vmax = QMIN.copy(), QMAX.copy(), VMAX.copy()
        # (alpha, a, theta offset, r) of each joint; theta = q_i + offset
        self._table = [(pi, 0.0, 0.0, -0.675), (pi / 2, 0.26, 0.0, 0.0), (0.0, a3, pi / 2, 0.0),
                       (-pi / 2, 0.035, 0.0, -0.67), (pi / 2, 0.0, 0.0, 0.0), (-pi / 2, 0.0, 0.0, 0.0)]

    def fMe(self, q):
        """4x4 pose of the tool frame in the base frame for joint angles q (6,)."""
        M = np.eye(4)
        for (alpha, a, off, r), qi in zip(self._table, q):
            M = M @ mdh(alpha, a, qi + off, r)
        return M @ WME

    def fJe(self, q, h=1e-7):
        """6x6 Jacobian in the base frame (numerical): [linear velocity; angular velocity] = J * qdot."""
        q = np.asarray(q, dtype=float)
        M = self.fMe(q)
        J = np.zeros((6, 6))
        for i in range(6):
            dq = q.copy()
            dq[i] += h
            Mp = self.fMe(dq)
            J[:3, i] = (Mp[:3, 3] - M[:3, 3]) / h
            J[3:, i] = tu_of(Mp[:3, :3] @ M[:3, :3].T) / h
        return J
