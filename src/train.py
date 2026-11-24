"""PPO training of the reaching agent (stable-baselines3).

Examples:
    python src/train.py --name pos_q --steps 2000000 --task position
    python src/train.py --name pose_q --steps 1000000 --task pose --w-rot 0.1 --init-from runs/pos_q/final_model.zip
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.logger import configure

sys.path.insert(0, os.path.dirname(__file__))
from env import KR16ReachEnv  # noqa: E402

EVAL_SEED = 12345
N_EVAL = 200

HYPERPARAMS = dict(n_steps=256, batch_size=1024, n_epochs=10, gamma=0.98, gae_lambda=0.95,
                   clip_range=0.2, ent_coef=0.0, learning_rate=3e-4)
POLICY_KWARGS = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh)


def run_episode(model, env, seed):
    """One deterministic episode on a fixed start/target pair; returns the last info and the length."""
    obs, _ = env.reset(seed=seed)
    done, n, info = False, 0, {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        done, n = term or trunc, n + 1
    return info, n


class FixedSetEval(BaseCallback):
    """Every eval_freq steps: 200 deterministic episodes on the same start/target pairs -> eval.csv."""

    def __init__(self, env_kwargs, out_dir, eval_freq):
        super().__init__()
        self.env = KR16ReachEnv(**env_kwargs)
        self.env.reset(seed=EVAL_SEED)                       # seeds the env generator once
        self.seeds = np.random.default_rng(EVAL_SEED).integers(0, 2**31 - 1, N_EVAL)
        self.out_dir, self.eval_freq, self.best = out_dir, eval_freq, -1.0
        self.last = 0
        self.path = os.path.join(out_dir, "eval.csv")
        with open(self.path, "w", newline="") as f:
            csv.writer(f).writerow(["timesteps", "success_rate", "mean_pos_err", "median_pos_err",
                                    "mean_rot_err", "mean_ep_len"])

    def evaluate(self):
        res = [run_episode(self.model, self.env, int(s)) for s in self.seeds]
        pos = np.array([r[0]["pos_err"] for r in res])
        rot = np.array([r[0]["rot_err"] for r in res])
        succ = np.mean([r[0]["is_success"] for r in res])
        row = [self.num_timesteps, succ, pos.mean(), np.median(pos), rot.mean(), np.mean([r[1] for r in res])]
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if succ > self.best:
            self.best = succ
            self.model.save(os.path.join(self.out_dir, "best_model"))
        print(f"[eval] t={self.num_timesteps} success={succ:.3f} median pos err={np.median(pos):.4f} m", flush=True)

    def _on_step(self):
        if self.num_timesteps - self.last >= self.eval_freq:
            self.last = self.num_timesteps
            self.evaluate()
        return True

    def _on_training_end(self):
        if self.num_timesteps != self.last:
            self.evaluate()


class TolCurriculum(BaseCallback):
    """A tolerance decays geometrically from tol_start to tol_final over the first `frac` of the run.

    `setter` is the env method to call ("set_pos_tol" in m, "set_rot_tol" in rad); `tag` names the logged value.

    tol(t) = tol_start * (tol_final / tol_start)^min(t / (frac * total), 1), t counted from the start of this run.
    Updated at every rollout start through venv.env_method (works through VecNormalize, which forwards it).
    """

    def __init__(self, tol_start, tol_final, frac, total_steps, setter="set_pos_tol", tag="train/pos_tol"):
        super().__init__()
        self.setter, self.tag = setter, tag
        self.tol_start, self.tol_final, self.frac, self.total = tol_start, tol_final, frac, total_steps
        self.t0 = None

    def tol(self, t):
        x = min(t / max(self.frac * self.total, 1), 1.0)
        return self.tol_start * (self.tol_final / self.tol_start) ** x

    def _on_training_start(self):
        self.t0 = self.num_timesteps

    def _on_rollout_start(self):
        v = self.tol(self.num_timesteps - self.t0)
        self.training_env.env_method(self.setter, v)
        self.logger.record(self.tag, v)

    def _on_step(self):
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--obs-q", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--task", choices=["position", "pose"], default="position")
    ap.add_argument("--w-rot", type=float, default=0.0)
    ap.add_argument("--a3", type=float, default=0.68)
    ap.add_argument("--init-from", default=None)
    ap.add_argument("--eval-freq", type=int, default=50_000)
    ap.add_argument("--reward", choices=["distance", "progress", "log_progress", "keypoint"], default="log_progress")
    ap.add_argument("--multiscale-obs", action=argparse.BooleanOptionalAction, default=True, help="fill the multi-scale error block of the observation")
    ap.add_argument("--gamma", type=float, default=HYPERPARAMS["gamma"], help="discount factor (also overrides a warm-started model)")
    ap.add_argument("--max-steps", type=int, default=100, help="episode length in control steps of 0.05 s")
    ap.add_argument("--time-penalty", type=float, default=0.0, help="reward cost per step (favours short, fast reaches)")
    ap.add_argument("--reset-std", type=float, default=None, help="warm start: reset the exploration std to this value (re-opens exploration for a new task)")
    ap.add_argument("--limit-penalty", type=float, default=0.0, help="reward cost per joint pushed against its limit in a step")
    ap.add_argument("--history", type=int, default=0, help="number of past steps (tool motion + action, 12 values each) appended to the observation")
    ap.add_argument("--sde", action=argparse.BooleanOptionalAction, default=True, help="gSDE exploration (use_sde=True)")
    ap.add_argument("--sde-freq", type=int, default=-1, help="sde_sample_freq")
    ap.add_argument("--pos-tol", type=float, default=0.01, help="final position tolerance (m)")
    ap.add_argument("--tol-start", type=float, default=None, help="initial tolerance of the accuracy curriculum (m)")
    ap.add_argument("--tol-frac", type=float, default=0.5, help="fraction of the steps over which the tolerance decays")
    ap.add_argument("--rot-tol-start", type=float, default=None, help="initial orientation tolerance (deg) of the accuracy curriculum, decays to 5 deg over --tol-frac")
    ap.add_argument("--target-rot-range", type=float, default=None, help="deg: sample only targets within this angle of the tool-down orientation Rot(x, pi)")
    ap.add_argument("--log-std-init", type=float, default=-1.0, help="initial log of the action noise std")
    ap.add_argument("--norm-reward", action=argparse.BooleanOptionalAction, default=True, help="VecNormalize on rewards only (obs untouched)")
    ap.add_argument("--threads", type=int, default=0, help="torch threads (0 = torch default)")
    ap.add_argument("--vec", choices=["dummy", "subproc"], default="subproc")
    ap.add_argument("--n-steps", type=int, default=HYPERPARAMS["n_steps"], help="rollout length per env")
    ap.add_argument("--batch-size", type=int, default=HYPERPARAMS["batch_size"])
    ap.add_argument("--n-epochs", type=int, default=HYPERPARAMS["n_epochs"])
    ap.add_argument("--lr", type=float, default=HYPERPARAMS["learning_rate"])
    ap.add_argument("--lr-decay", action="store_true", help="learning rate decays linearly to 0")
    ap.add_argument("--target-kl", type=float, default=None, help="stop the epochs of an update when KL > 1.5 * target")
    ap.add_argument("--ent-coef", type=float, default=HYPERPARAMS["ent_coef"])
    args = ap.parse_args()
    HYPERPARAMS.update(n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                       learning_rate=args.lr, ent_coef=args.ent_coef)

    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)
    HYPERPARAMS["gamma"] = args.gamma
    env_kwargs = dict(max_steps=args.max_steps, limit_penalty=args.limit_penalty, time_penalty=args.time_penalty, obs_q=args.obs_q, task=args.task, w_rot=args.w_rot, a3=args.a3, reward_mode=args.reward,
                      multiscale_obs=args.multiscale_obs, pos_tol=args.pos_tol, history=args.history)
    if args.target_rot_range is not None:      # only stored when used, so old configs stay identical
        env_kwargs["target_rot_range"] = float(np.deg2rad(args.target_rot_range))
    if args.threads:
        torch.set_num_threads(args.threads)
    config = dict(vars(args), env_kwargs=env_kwargs, hyperparams=HYPERPARAMS)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    set_random_seed(args.seed)
    vec_cls = SubprocVecEnv if args.vec == "subproc" else DummyVecEnv
    train_kwargs = dict(env_kwargs, pos_tol=args.tol_start if args.tol_start else args.pos_tol)
    if args.rot_tol_start:
        train_kwargs["rot_tol"] = float(np.deg2rad(args.rot_tol_start))
    venv = make_vec_env(KR16ReachEnv, n_envs=args.n_envs, seed=args.seed, env_kwargs=train_kwargs,
                        vec_env_cls=vec_cls, monitor_dir=out_dir,
                        monitor_kwargs=dict(info_keywords=("is_success", "pos_err")))
    vn_path = os.path.join(os.path.dirname(args.init_from), "vecnormalize.pkl") if args.init_from else None
    if vn_path and os.path.exists(vn_path):
        venv = VecNormalize.load(vn_path, venv)           # warm start: keep the reward statistics
        venv.training, venv.norm_reward, venv.gamma = True, True, args.gamma
    elif args.norm_reward:
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=HYPERPARAMS["gamma"])
    if args.init_from:
        model = PPO.load(args.init_from, env=venv, device="cpu", custom_objects={"seed": args.seed, "gamma": args.gamma})
        if args.reset_std:
            with torch.no_grad():
                model.policy.log_std.fill_(float(np.log(args.reset_std)))
    else:
        model = PPO("MlpPolicy", venv, policy_kwargs=dict(POLICY_KWARGS, log_std_init=args.log_std_init), seed=args.seed, device="cpu",
                    verbose=0, use_sde=args.sde, sde_sample_freq=args.sde_freq, target_kl=args.target_kl,
                    **dict(HYPERPARAMS, learning_rate=(lambda f: f * args.lr) if args.lr_decay else args.lr))
    model.set_logger(configure(out_dir, ["stdout", "csv"]))

    t0 = time.time()
    callbacks = [FixedSetEval(env_kwargs, out_dir, args.eval_freq)]       # always at the final tolerance
    if args.tol_start:
        callbacks.append(TolCurriculum(args.tol_start, args.pos_tol, args.tol_frac, args.steps))
    if args.rot_tol_start:
        callbacks.append(TolCurriculum(np.deg2rad(args.rot_tol_start), np.deg2rad(5), args.tol_frac,
                                       args.steps, setter="set_rot_tol", tag="train/rot_tol"))
    model.learn(args.steps, callback=callbacks, reset_num_timesteps=not args.init_from)
    model.save(os.path.join(out_dir, "final_model"))
    if isinstance(venv, VecNormalize):
        venv.save(os.path.join(out_dir, "vecnormalize.pkl"))
    dt = time.time() - t0
    print(f"done: {args.steps} steps in {dt:.0f} s ({args.steps / dt:.0f} fps)")


if __name__ == "__main__":
    main()
