"""Evaluate a saved Diffusion Policy checkpoint on ManiSkill state tasks."""

import argparse
import csv
from pathlib import Path

import torch

from train_dp import (
    DEVICE,
    ConditionalUnet1D,
    DDPMScheduler,
    GaussianNormalizer,
    evaluate_policy,
)


def load_checkpoint(path: str, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args = ckpt.get("args", {})
    normalizer_state = ckpt["normalizer"]
    normalizer = GaussianNormalizer(
        normalizer_state["obs_mean"],
        normalizer_state["obs_std"],
        normalizer_state["action_mean"],
        normalizer_state["action_std"],
    )

    obs_horizon = int(args.get("obs_horizon", 2))
    pred_horizon = int(args.get("pred_horizon", 16))
    obs_dim = int(args.get("obs_dim", 43))
    action_dim = int(args.get("action_dim", normalizer.action_dim))
    hidden_dim = int(args.get("hidden_dim", 256))
    num_layers = int(args.get("num_layers", 4))

    policy = ConditionalUnet1D(
        action_dim=action_dim,
        obs_dim=obs_dim * obs_horizon,
        pred_horizon=pred_horizon,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])

    scheduler = DDPMScheduler(
        num_timesteps=int(args.get("num_timesteps", 100)),
        schedule=args.get("schedule", "cosine"),
    )
    return ckpt, args, policy, scheduler, normalizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved policy checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", default=None)
    parser.add_argument("--obs-mode", default=None)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--inference-steps", type=int, default=None)
    parser.add_argument("--sample-init", choices=["random", "zero"], default="random")
    parser.add_argument("--csv-out", type=str, default=None)
    args = parser.parse_args()

    device = DEVICE
    ckpt, train_args, policy, scheduler, normalizer = load_checkpoint(args.checkpoint, device)
    env_id = args.env or train_args.get("env", "PegInsertionSide-v1")
    obs_mode = args.obs_mode or train_args.get("obs_mode", "state")
    action_horizon = int(args.action_horizon or train_args.get("action_horizon", 8))
    inference_steps = int(args.inference_steps or train_args.get("inference_steps", 20))
    max_steps = args.max_steps if args.max_steps is not None else train_args.get("eval_max_steps")

    result = evaluate_policy(
        env_id=env_id,
        policy_net=policy,
        scheduler=scheduler,
        normalizer=normalizer,
        obs_horizon=int(train_args.get("obs_horizon", 2)),
        pred_horizon=int(train_args.get("pred_horizon", 16)),
        num_episodes=args.episodes,
        obs_mode=obs_mode,
        device=device,
        num_inference_steps=inference_steps,
        max_episode_steps=max_steps,
        action_horizon=action_horizon,
        sample_init=args.sample_init,
    )

    row = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "epoch": ckpt.get("epoch"),
        "env": env_id,
        "episodes": args.episodes,
        "max_steps": max_steps,
        "action_horizon": action_horizon,
        "inference_steps": inference_steps,
        "sample_init": args.sample_init,
        **result,
    }
    print(row)

    if args.csv_out:
        csv_path = Path(args.csv_out)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        exists = csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


if __name__ == "__main__":
    main()
