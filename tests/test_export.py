"""Torch policy and the numpy export must give the same actions (needs a trained model; skipped if absent)."""
import os
import sys

import numpy as np
import pytest

def make_model(tmp_path, sde, multiscale):
    """Tiny freshly trained PPO (no dependence on saved runs, whose obs size may be outdated)."""
    import json
    from stable_baselines3 import PPO
    from env import KR16ReachEnv

    kw = dict(multiscale_obs=multiscale, reward_mode="log_progress")
    model = PPO("MlpPolicy", KR16ReachEnv(**kw), n_steps=64, batch_size=32, use_sde=sde, seed=0, device="cpu",
                policy_kwargs=dict(log_std_init=-1.0))
    model.learn(128)
    zip_path = str(tmp_path / "m.zip")
    model.save(zip_path)
    with open(tmp_path / "config.json", "w") as f:
        json.dump(dict(obs_q=True, env_kwargs=dict(obs_q=True, **kw)), f)
    return model, zip_path, kw


@pytest.mark.parametrize("sde,multiscale", [(False, False), (True, True)])
def test_numpy_policy_matches_torch(tmp_path, sde, multiscale):
    from env import KR16ReachEnv
    from export_numpy import export
    from policy_numpy import NumpyPolicy, build_obs

    model, zip_path, kw = make_model(tmp_path, sde, multiscale)
    npz = str(tmp_path / "p.npz")
    export(zip_path, npz)
    pol = NumpyPolicy(npz)
    assert pol.multiscale_obs == multiscale and pol.obs_q
    env = KR16ReachEnv(**kw)
    obs, _ = env.reset(seed=0)
    for k in range(100):
        # the shared builder must reproduce the env observation exactly
        assert np.allclose(pol.build_obs(env.q, env.M, env.M_star, env.prev_action), obs, atol=1e-6)
        assert np.allclose(build_obs(env.q, env.M, env.M_star, env.prev_action, True, multiscale), obs, atol=1e-6)
        a_torch = model.predict(obs, deterministic=True)[0]
        assert np.allclose(a_torch, pol.act(obs), atol=1e-5)
        obs, _, term, trunc, _ = env.step(np.random.default_rng(k).uniform(-1, 1, 6))
        if term or trunc:
            obs, _ = env.reset(seed=k)


def test_policy_numpy_does_not_import_torch():
    import subprocess
    src = os.path.join(os.path.dirname(__file__), "..", "src")
    code = "import sys; sys.path.insert(0, r'%s'); import policy_numpy; print('torch' in sys.modules)" % src
    assert subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip() == "False"


# ---- observation history ----

def test_obs_shape_history():
    from env import KR16ReachEnv
    from obs import OBS_DIM, HIST_ENTRY, obs_dim
    assert OBS_DIM == 62 and obs_dim(4) == 62 + 4 * HIST_ENTRY
    for h in (0, 4):
        env = KR16ReachEnv(history=h)
        o, _ = env.reset(seed=0)
        assert o.shape == (obs_dim(h),) == env.observation_space.shape and o.dtype == np.float32
        o, *_ = env.step(env.action_space.sample())
        assert o.shape == (obs_dim(h),)


def test_history_buffer_order():
    from env import KR16ReachEnv
    from obs import history_entry
    env = KR16ReachEnv(history=4)
    o, _ = env.reset(seed=1)
    assert np.all(o[62:] == 0)
    rng = np.random.default_rng(0)
    entries = []
    for k in range(5):
        M0 = env.M.copy()
        a = rng.uniform(-1, 1, 6)
        o, *_ = env.step(a)
        entries.append(history_entry(M0, env.M, np.clip(a, -1, 1)))
        h = o[62:].reshape(4, 12)
        if k == 0:
            assert np.allclose(h[0], entries[0]) and np.all(h[1:] == 0)
    expect = np.array(entries[::-1][:4])          # last 4, newest first
    assert np.allclose(o[62:].reshape(4, 12), expect, atol=1e-6)
    assert np.all(np.abs(expect[:, :6]) <= 3.0)
    o, _ = env.reset(seed=2)
    assert np.all(o[62:] == 0)


def test_history_default_unchanged():
    from env import KR16ReachEnv
    a, b = KR16ReachEnv(), KR16ReachEnv(history=2)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert np.allclose(oa, ob[:62])
    act = np.full(6, 0.3)
    assert np.allclose(a.step(act)[0], b.step(act)[0][:62])


def test_numpy_policy_matches_torch_history(tmp_path):
    import json
    from stable_baselines3 import PPO
    from env import KR16ReachEnv
    from export_numpy import export
    from policy_numpy import NumpyPolicy

    kw = dict(history=4, reward_mode="log_progress")
    env = KR16ReachEnv(**kw)
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, seed=0, device="cpu")
    model.learn(128)
    zip_path = str(tmp_path / "m.zip")
    model.save(zip_path)
    with open(tmp_path / "config.json", "w") as f:
        json.dump(dict(obs_q=True, env_kwargs=dict(obs_q=True, **kw)), f)
    npz = str(tmp_path / "p.npz")
    export(zip_path, npz)
    pol = NumpyPolicy(npz)
    assert pol.history == 4
    obs, _ = env.reset(seed=0)
    for k in range(60):
        assert np.allclose(model.predict(obs, deterministic=True)[0], pol.act(obs), atol=1e-5)
        obs, _, term, trunc, _ = env.step(np.random.default_rng(k).uniform(-1, 1, 6))
        if term or trunc:
            obs, _ = env.reset(seed=k)


def test_ros2_core_history_matches_env(tmp_path):
    """PPOController builds the same observation (incl. history) as the env for the same poses and actions."""
    from env import KR16ReachEnv
    ros2 = os.path.join(os.path.dirname(__file__), "..", "ros2")
    sys.path.insert(0, ros2)
    try:
        import ppo_core, offline_check
    finally:
        sys.path.remove(ros2)
    npz = str(tmp_path / "d.npz")
    offline_check.make_dummy(npz, history=4)
    env = KR16ReachEnv(history=4)
    obs, _ = env.reset(seed=5)
    ctl = ppo_core.PPOController(npz, env.M_star)
    seen = []
    orig = ctl.pol.act
    ctl.pol.act = lambda o: (seen.append(o.copy()), orig(o))[1]
    for k in range(7):
        ctl.step(env.q, env.M)
        a = ctl.prev_a                             # the action just sent (in [-1, 1])
        assert np.allclose(seen[-1], obs, atol=1e-6)
        obs, *_ = env.step(a)
    # new target clears the history
    ctl.set_target(env.M_star)
    ctl.step(env.q, env.M)
    assert np.all(seen[-1][62:] == 0)
