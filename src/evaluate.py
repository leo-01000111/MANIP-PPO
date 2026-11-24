"""Compare PPO models and the Jacobian baseline on one fixed test set.

Example:
    python src/evaluate.py --models runs/pos_q/final_model.zip:ppo_q --baseline --task position
Each model needs the config.json written by train.py next to it (it gives the env settings: obs_q, multiscale_obs, reward_mode, pos_tol).
Outputs runs/eval_<tag>.json (metrics) and runs/eval_<tag>_traj.npz (3 trajectories per method).
"""
import argparse
import json
import os
import sys

import numpy as np
from stable_baselines3 import PPO

sys.path.insert(0, os.path.dirname(__file__))
from baseline import JacobianController  # noqa: E402
from env import KR16ReachEnv  # noqa: E402
from kr16 import KR16  # noqa: E402

N_TRAJ = 3


def rollout(env, policy, seed):
    """Run one episode with policy(obs, env) -> action; return final info, steps, q(t), p(t)."""
    obs, info = env.reset(seed=seed)
    qs, ps = [env.q.copy()], [env.M[:3, 3].copy()]
    done, n = False, 0
    while not done:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        qs.append(env.q.copy())
        ps.append(env.M[:3, 3].copy())
        done, n = term or trunc, n + 1
    return info, n, np.array(qs), np.array(ps), env.M_star[:3, 3].copy()


def evaluate_method(label, make_env, policy, episodes, seed):
    env = make_env()
    seeds = np.random.default_rng(seed).integers(0, 2**31 - 1, episodes)
    res = [rollout(env, policy, int(s)) for s in seeds]
    pos = np.array([r[0]["pos_err"] for r in res])
    rot = np.rad2deg([r[0]["rot_err"] for r in res])
    ok = np.array([r[0]["is_success"] for r in res])
    steps = np.array([r[1] for r in res])
    metrics = dict(success_rate=float(ok.mean()), pos_err_median=float(np.median(pos)),
                   pos_err_p90=float(np.percentile(pos, 90)), rot_err_deg_median=float(np.median(rot)),
                   rot_err_deg_p90=float(np.percentile(rot, 90)),
                   steps_to_reach=float(steps[ok].mean()) if ok.any() else float("nan"))
    traj = {}
    for i in range(N_TRAJ):
        traj[f"{label}_q_{i}"], traj[f"{label}_p_{i}"], traj[f"{label}_target_{i}"] = res[i][2], res[i][3], res[i][4]
    return metrics, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help="path[:label] of PPO zip files")
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--task", choices=["position", "pose"], default="position")
    ap.add_argument("--a3", type=float, default=0.68, help="link length of the world")
    ap.add_argument("--model-a3", type=float, default=0.68, help="link length in the Jacobian controller's model")
    ap.add_argument("--lam", type=float, default=8.0)
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--max-steps", type=int, default=100, help="episode length for the baseline (PPO models use their config)")
    ap.add_argument("--target-rot-range", type=float, default=None, help="baseline: restricted target orientation (deg)")
    args = ap.parse_args()

    results, trajs = {}, {}
    for item in args.models:
        path, _, label = item.partition(":")
        label = label or os.path.basename(os.path.dirname(path))
        with open(os.path.join(os.path.dirname(path), "config.json")) as f:
            cfg = json.load(f)
        # the model runs with the env settings it was trained with (obs_q, multiscale_obs, pos_tol, ...)
        ekw = dict(cfg.get("env_kwargs", {}), obs_q=cfg["obs_q"], task=args.task, a3=args.a3)
        ekw.pop("w_rot", None)
        model = PPO.load(path, device="cpu")
        make_env = lambda ekw=ekw: KR16ReachEnv(**ekw)
        policy = lambda obs, env: model.predict(obs, deterministic=True)[0]
        results[label], t = evaluate_method(label, make_env, policy, args.episodes, args.seed)
        trajs.update(t)
    if args.baseline:
        ctrl = JacobianController(KR16(args.model_a3), args.task, args.lam)
        rr = None if args.target_rot_range is None else np.deg2rad(args.target_rot_range)
        make_env = lambda: KR16ReachEnv(task=args.task, a3=args.a3, max_steps=args.max_steps, target_rot_range=rr)
        results["jacobian"], t = evaluate_method("jacobian", make_env, lambda obs, env: ctrl.act(env.q, env.M_star),
                                                 args.episodes, args.seed)
        trajs.update(t)

    tag = args.tag or f"{args.task}_a3_{args.a3}"
    os.makedirs("runs", exist_ok=True)
    with open(f"runs/eval_{tag}.json", "w") as f:
        json.dump(dict(args=vars(args), results=results), f, indent=2)
    np.savez(f"runs/eval_{tag}_traj.npz", **trajs)

    print(f"{'method':14s} {'success':>8s} {'pos med':>9s} {'pos p90':>9s} {'rot med(deg)':>13s} {'steps':>7s}")
    for k, m in results.items():
        print(f"{k:14s} {m['success_rate']:8.3f} {m['pos_err_median']:9.4f} {m['pos_err_p90']:9.4f} "
              f"{m['rot_err_deg_median']:13.2f} {m['steps_to_reach']:7.1f}")


if __name__ == "__main__":
    main()
