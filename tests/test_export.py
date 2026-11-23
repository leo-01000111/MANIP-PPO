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
