"""Tests of the optional features: keypoint reward, orientation tolerance, restricted target orientation."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from env import KP_LEN, R_REF, KR16ReachEnv, keypoint_error
from kr16 import rot_tu, tu_of


def test_keypoint_error_zero_and_position_only():
    env = KR16ReachEnv(task="pose")
    env.reset(seed=0)
    assert keypoint_error(env.M, env.M) == 0.0
    M2 = env.M.copy()
    M2[:3, 3] += [0.1, -0.2, 0.05]
    assert np.isclose(keypoint_error(env.M, M2), np.linalg.norm([0.1, -0.2, 0.05]))


def test_keypoint_error_pure_rotation():
    env = KR16ReachEnv(task="pose")
    env.reset(seed=1)
    prev = 0.0
    for t in (0.1, 0.5, 1.0, 2.0):
        M2 = env.M.copy()
        M2[:3, :3] = env.M[:3, :3] @ rot_tu([0, 0, t])      # rotation about the tool z axis
        e = keypoint_error(env.M, M2)
        assert np.isclose(e, 4 * KP_LEN * np.sin(t / 2) / 3) and e > prev
        prev = e


def test_keypoint_reward_and_info():
    env = KR16ReachEnv(task="pose", reward_mode="keypoint", w_rot=5.0)
    env.reset(seed=2)
    env.M_star = env.M.copy()
    env.M_star[0, 3] += 0.5
    env._prev_kp = keypoint_error(env.M, env.M_star)
    env.M_star[0, 3] -= 0.25                                  # target moves closer by half the distance
    _, r, term, _, info = env.step(np.zeros(6))
    assert abs(info["kp_err"] - 0.25) < 1e-9 and abs(r - np.log(2)) < 1e-9 and not term


def test_set_rot_tol():
    env = KR16ReachEnv(task="pose", rot_tol=np.deg2rad(5))
    env.reset(seed=0)
    env.M_star = env.M.copy()
    env.M_star[:3, :3] = env.M[:3, :3] @ rot_tu([0, 0, np.deg2rad(20)])
    assert not env.step(np.zeros(6))[2]
    env.set_rot_tol(np.deg2rad(30))
    assert env.step(np.zeros(6))[2]


def test_restricted_target_orientation():
    rng_range = np.deg2rad(60)
    env = KR16ReachEnv(task="pose", target_rot_range=rng_range)
    for s in range(50):
        env.reset(seed=s)
        assert np.linalg.norm(tu_of(R_REF.T @ env.M_star[:3, :3])) <= rng_range + 1e-9


def test_mean_tries_60deg(capsys):
    env = KR16ReachEnv(task="pose", target_rot_range=np.deg2rad(60))
    tries = []
    for s in range(200):
        env.reset(seed=s)
        tries.append(env.last_target_tries)
    with capsys.disabled():
        print(f"\nmean tries for 60 deg: {np.mean(tries):.1f} (max {max(tries)})")
    assert np.mean(tries) < 200


def test_defaults_unchanged():
    env = KR16ReachEnv(reward_mode="log_progress", task="pose", w_rot=0.5)
    obs, _ = env.reset(seed=7)
    assert env.target_rot_range is None and obs.shape == (62,)
    rng = np.random.default_rng(0)
    for _ in range(5):
        pe, re_ = env._prev_err
        a = rng.uniform(-1, 1, 6)
        _, r, _, _, info = env.step(a)
        lp = lambda x, eps: np.log(max(x, eps))
        from env import EPS_P, EPS_R
        exp = (lp(pe, EPS_P) - lp(info["pos_err"], EPS_P) + 0.5 * (lp(re_, EPS_R) - lp(info["rot_err"], EPS_R))
               - 0.01 * float(np.clip(a, -1, 1) @ np.clip(a, -1, 1)))
        assert abs(r - exp) < 1e-9 and "kp_err" not in info


def test_check_env_new_modes():
    env = KR16ReachEnv(task="pose", reward_mode="keypoint", target_rot_range=np.deg2rad(60))
    check_env(env, skip_render_check=True)
    env.reset(seed=0)
    for _ in range(300):
        _, r, term, trunc, _ = env.step(env.action_space.sample())
        assert np.isfinite(r)
        if term or trunc:
            env.reset()
