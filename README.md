# MANIP-PPO

A PPO agent (stable-baselines3) learns to reach target poses with the Kuka KR16 arm without ever
seeing its kinematic model. The forward kinematics is the hidden "world" of a Gymnasium environment;
the agent gets only measurements (joint angles, tool pose, target pose) and commands joint velocities.
A lab-style Jacobian controller (Exercise 6, full model knowledge) is the baseline.

## Layout
- `src/kr16.py` hidden world: forward kinematics `fMe`, Jacobian `fJe`, limits (class `KR16(a3)`).
- `src/obs.py` observation builder (numpy only, shared by the env and the exported policy).
- `src/env.py` `KR16ReachEnv`: tasks `position` / `pose`, optional encoders (`obs_q`), reward weight `w_rot`.
- `src/baseline.py` `JacobianController`: pinv(J) velocity law of the lab.
- `src/train.py` PPO training, fixed-set evaluation to `runs/<name>/eval.csv`.
- `src/evaluate.py` compares models and the baseline on one fixed test set.
- `src/export_numpy.py`, `src/policy_numpy.py` torch-free policy for the ROS2 VM.
- `tests/` pytest checks of the world model, the env and the numpy export.

## Run (from the repo root, system Python)
    python -m pytest tests
    python src/train.py --name pos_q --steps 2000000 --task position
    python src/train.py --name pos_noq --steps 2000000 --no-obs-q
    # curriculum: keep training the same network on the pose task with an orientation weight
    python src/train.py --name pose_q --task pose --w-rot 0.1 --steps 1000000 --init-from runs/pos_q/final_model.zip
    # robustness: world with a3 = 0.75
    python src/train.py --name pos_q_a3 --a3 0.75 --steps 500000 --init-from runs/pos_q/final_model.zip
    python src/evaluate.py --models runs/pos_q/final_model.zip:ppo_q --baseline --task position --a3 0.75 --model-a3 0.68
    python src/export_numpy.py runs/pos_q/final_model.zip policy.npz

Optional toggles of `train.py` (all off by default; stored in `config.json`, and `evaluate.py` / `export_numpy.py` read them from it):
- `--reward {distance,progress,log_progress}`: `log_progress` pays log(d_prev) - log(d), the same at 1 m and at 1 cm.
- `--history N`: appends the last N steps (tool motion and action, 12 values each, newest first, zeros at reset) to the observation; default 0.
- `--multiscale-obs`: fills the last 26 observation entries (direction, log-norm, tanh at 3 scales of the position and orientation errors); zeros otherwise.
- `--sde [--sde-freq N]`: gSDE exploration. `--log-std-init x` sets the initial action noise.
- `--tol-start m [--tol-frac f] [--pos-tol m]`: accuracy curriculum, tolerance decays geometrically from `m` to `--pos-tol` (default 0.01) over the first fraction `f` of the steps; the periodic evaluation always uses `--pos-tol`.
- `--norm-reward`: VecNormalize on rewards, saved as `vecnormalize.pkl` and reloaded by `--init-from`.
Observation size is 62 (36 + 26 multi-scale); models trained before this change (36) do not load.

In the VM, copy `kr16.py`, `obs.py`, `policy_numpy.py`, `policy.npz`:
`pol = NumpyPolicy("policy.npz"); pol.act(pol.build_obs(q, M, M_star, prev_action))` gives an action in [-1, 1]^6;
the joint velocity is `action * speed_scale * VMAX`.
