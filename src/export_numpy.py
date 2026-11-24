"""Export the deterministic actor of a trained SB3 PPO model to a numpy .npz.

Usage: python src/export_numpy.py runs/<name>/final_model.zip out.npz
"""
import argparse
import json
import os

import numpy as np
from stable_baselines3 import PPO


def export(model_path, out_path):
    model = PPO.load(model_path, device="cpu")
    pol = model.policy
    layers = [m for m in pol.mlp_extractor.policy_net if hasattr(m, "weight")] + [pol.action_net]
    arrays = {"n_layers": np.array(len(layers))}
    for i, lin in enumerate(layers):
        arrays[f"W{i}"] = lin.weight.detach().numpy().astype(np.float64)
        arrays[f"b{i}"] = lin.bias.detach().numpy().astype(np.float64)
    # observation settings the model was trained with (config.json next to the model), stored for the VM
    cfg_path = os.path.join(os.path.dirname(model_path), "config.json")
    ekw = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        ekw = dict(cfg.get("env_kwargs", {}), obs_q=cfg.get("obs_q", True))
    arrays["obs_q"] = np.array(bool(ekw.get("obs_q", True)))
    arrays["multiscale_obs"] = np.array(bool(ekw.get("multiscale_obs", False)))
    arrays["history"] = np.array(int(ekw.get("history", 0)))
    np.savez(out_path, **arrays)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("out")
    args = ap.parse_args()
    export(args.model, args.out)
    print("saved", args.out)
