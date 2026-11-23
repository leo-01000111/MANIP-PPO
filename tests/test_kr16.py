"""Tests of the KR16 world model and of the Gymnasium environment."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from env import KR16ReachEnv
from kr16 import KR16, QMAX, QMIN, rot_tu, tu_of

# Lab targets (robot_kr16.h) as [x, y, z, theta*u] and their IGM solutions (rounded to 4 decimals)
POSE_A = [0.812110097, -1.187465983, 0.4915271277, 0.9423584377, -2.852157604, 0.1075927581]
POSE_B = [1.282877001, 0.3344962596, 1.415167601, -1.142592072, 1.424640835, 1.004968206]
QA = np.array([0.9545, -0.4901, 0.9822, 0.1735, 1.0564, 1.5088])
QB = np.array([-0.1440, -1.0140, 0.6810, -1.3110, 1.2950, 1.1620])


def check_pose(M, pose):
    assert np.linalg.norm(M[:3, 3] - pose[:3]) < 5e-3
    assert np.linalg.norm(tu_of(M[:3, :3] @ rot_tu(pose[3:]).T)) < 1e-2


def test_home_position():
    p = KR16().fMe(np.zeros(6))[:3, 3]
    assert np.allclose(p, [1.768, 0.0, 0.64], atol=1e-6)


def test_lab_poses():
    robot = KR16()
    check_pose(robot.fMe(QA), POSE_A)
    check_pose(robot.fMe(QB), POSE_B)


def test_jacobian_matches_finite_differences():
    rng = np.random.default_rng(0)
    robot = KR16()
    for _ in range(5):
        q = rng.uniform(QMIN * 0.8, QMAX * 0.8)
        dq = 1e-4 * rng.standard_normal(6)
        M0, M1 = robot.fMe(q), robot.fMe(q + dq)
        J = robot.fJe(q)
        assert np.allclose(M1[:3, 3] - M0[:3, 3], J[:3] @ dq, atol=1e-7)
        assert np.allclose(tu_of(M1[:3, :3] @ M0[:3, :3].T), J[3:] @ dq, atol=1e-7)


def test_a3_changes_reach():
    # at q = 0 the arm is stretched along x: a3 adds directly to x (link 3 lies along the arm)
    p0 = KR16(0.68).fMe(np.zeros(6))[:3, 3]
    p1 = KR16(0.75).fMe(np.zeros(6))[:3, 3]
    assert np.allclose(p1 - p0, [0.07, 0, 0], atol=1e-9)


def test_env_reset_step():
    env = KR16ReachEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == (env.obs_dim,) and obs.dtype == np.float32
    assert np.all(np.abs(obs) <= 3.0) and info["is_success"] is False
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert obs.shape == (env.obs_dim,) and np.isfinite(r)
    assert set(info) >= {"pos_err", "rot_err", "is_success", "q"}
    assert np.all(info["q"] >= QMIN) and np.all(info["q"] <= QMAX)


def test_env_truncates_at_max_steps():
    env = KR16ReachEnv(max_steps=5)
    env.reset(seed=2)
    for k in range(5):
        _, _, term, trunc, _ = env.step(np.zeros(6))
    assert trunc and not term


def test_env_success_terminates():
    env = KR16ReachEnv()
    env.reset(seed=3)
    env.M_star = env.world.fMe(env.q)        # target = current pose: success at the next (zero) step
    _, r, term, _, info = env.step(np.zeros(6))
    assert term and info["is_success"] and r > 0.9


def test_pose_task_needs_orientation():
    env = KR16ReachEnv(task="pose")
    env.reset(seed=4)
    env.M_star = env.world.fMe(env.q)
    env.M_star[:3, :3] = env.M_star[:3, :3] @ rot_tu([0, 0, 0.5])    # same position, rotated 0.5 rad
    _, _, term, _, info = env.step(np.zeros(6))
    assert not term and info["rot_err"] > 0.4


def test_check_env():
    check_env(KR16ReachEnv(), skip_render_check=True)
    check_env(KR16ReachEnv(task="pose", obs_q=False), skip_render_check=True)


def _step_reward(env, p_from, p_to):
    """Reward of one step that moves the tool from distance p_from to p_to of the target (zero action)."""
    env.reset(seed=0)
    env._prev_err = (p_from, 0.0)
    env.M = env.world.fMe(env.q)
    env.M_star = env.M.copy()
    env.M_star[0, 3] += p_to            # the step does not move the arm (zero action): distance is p_to
    _, r, _, _, info = env.step(np.zeros(6))
    assert abs(info["pos_err"] - p_to) < 1e-9
    return r


def test_log_progress_scale_invariance():
    env = KR16ReachEnv(reward_mode="log_progress")
    r1 = _step_reward(env, 1.0, 0.5)
    r2 = _step_reward(env, 0.02, 0.01)
    assert r1 > 0 and abs(r1 - r2) < 1e-9 and abs(r1 - np.log(2)) < 1e-9
    assert _step_reward(env, 0.5, 1.0) < 0                    # moving away is penalised


def test_multiscale_block():
    from obs import OBS_DIM, build_obs
    assert OBS_DIM == 62
    off, on = KR16ReachEnv(), KR16ReachEnv(multiscale_obs=True)
    o0, _ = off.reset(seed=3)
    o1, _ = on.reset(seed=3)
    assert o0.shape == o1.shape == (62,)
    assert np.all(o0[36:] == 0) and np.any(o1[36:] != 0)
    assert np.allclose(o0[:36], o1[:36])
    e = on.M_star[:3, 3] - on.M[:3, 3]
    assert np.allclose(o1[36:39], e / np.linalg.norm(e), atol=1e-6)
    assert np.isclose(o1[39], np.log10(np.linalg.norm(e)) / 4, atol=1e-6)
    assert np.allclose(o1[40:43], np.tanh(e), atol=1e-6) and np.allclose(o1[46:49], np.tanh(e / 0.01), atol=1e-6)
    assert np.all(np.abs(o1) <= 3.0)
    assert np.allclose(build_obs(on.q, on.M, on.M_star, on.prev_action, True, True), o1)


def test_set_pos_tol():
    env = KR16ReachEnv(pos_tol=0.01)
    env.reset(seed=0)
    env.M_star = env.M.copy()
    env.M_star[0, 3] += 0.05
    assert not env.step(np.zeros(6))[2]                       # 5 cm > 1 cm tolerance
    env.set_pos_tol(0.1)
    assert env.pos_tol == 0.1 and env.step(np.zeros(6))[2]    # now a success
