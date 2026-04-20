"""Record Diffusion Policy rollout videos with OpenCV."""

import argparse
import shutil
from pathlib import Path

import cv2
import gymnasium as gym
import mani_skill.envs  # noqa: F401
import numpy as np
import torch

from collect_demos import obs_to_numpy
from evaluate_checkpoint import load_checkpoint
from train_dp import DEVICE


def extract_state_obs(raw_obs) -> np.ndarray:
    state_obs = raw_obs["state"] if isinstance(raw_obs, dict) else raw_obs
    state_obs = obs_to_numpy(state_obs)
    return np.asarray(state_obs, dtype=np.float32).reshape(-1)


def as_frame(rendered) -> np.ndarray:
    frame = np.asarray(rendered)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


def write_frame(writer, frame: np.ndarray) -> None:
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def rollout_to_video(args, policy, scheduler, normalizer, train_args, seed: int, out_path: Path):
    env_kwargs = {}
    if args.max_steps is not None:
        env_kwargs["max_episode_steps"] = args.max_steps

    env = gym.make(
        args.env,
        obs_mode=args.obs_mode,
        render_mode="rgb_array",
        render_backend=args.render_backend,
        **env_kwargs,
    )
    obs, info = env.reset(seed=seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    first_obs = extract_state_obs(obs)
    obs_horizon = int(train_args.get("obs_horizon", 2))
    pred_horizon = int(train_args.get("pred_horizon", 16))
    obs_history = [first_obs.copy() for _ in range(obs_horizon)]

    first_frame = as_frame(env.render())
    height, width = first_frame.shape[:2]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*args.codec),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        env.close()
        raise RuntimeError(f"Could not open video writer for {out_path}")

    write_frame(writer, first_frame)
    done = False
    success = False
    step = 0
    total_reward = 0.0
    rollout_limit = args.max_steps or 500

    policy.eval()
    with torch.no_grad():
        while not done and step < rollout_limit:
            obs_seq = np.stack(obs_history[-obs_horizon:], axis=0)
            obs_t = torch.from_numpy(obs_seq).float().unsqueeze(0).to(DEVICE)
            obs_t = normalizer.normalize_obs(obs_t)

            action_seq = scheduler.sample(
                policy,
                obs_t,
                pred_horizon,
                normalizer.action_dim,
                num_inference_steps=args.inference_steps,
                init_noise=args.sample_init,
            )
            action_seq = normalizer.denormalize_action(action_seq)

            exec_horizon = min(args.action_horizon, pred_horizon)
            for action_i in range(exec_horizon):
                action_np = action_seq[0, action_i].cpu().numpy()
                action_np = np.clip(action_np, -1, 1)
                obs, reward, terminated, truncated, info = env.step(action_np)
                total_reward += float(reward)
                step += 1
                success = bool(info.get("success", False))
                done = bool(terminated or truncated or success)

                obs_history.append(extract_state_obs(obs))
                obs_history.pop(0)
                write_frame(writer, as_frame(env.render()))

                if done or step >= rollout_limit:
                    break

    writer.release()
    env.close()
    return success, step, total_reward


def main():
    parser = argparse.ArgumentParser(description="Record a policy rollout video.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--env", required=True)
    parser.add_argument("--obs-mode", default="state")
    parser.add_argument("--out", required=True)
    parser.add_argument("--attempts", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--action-horizon", type=int, default=8)
    parser.add_argument("--inference-steps", type=int, default=20)
    parser.add_argument("--sample-init", choices=["random", "zero"], default="random")
    parser.add_argument("--render-backend", default="sapien_cpu")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--codec", default="mp4v")
    args = parser.parse_args()

    _, train_args, policy, scheduler, normalizer = load_checkpoint(args.checkpoint, DEVICE)
    out_path = Path(args.out)
    tmp_dir = out_path.parent / "_tmp_attempts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    best = None
    for attempt in range(args.attempts):
        seed = args.seed_start + attempt
        tmp_path = tmp_dir / f"{out_path.stem}_seed{seed}.mp4"
        success, steps, reward = rollout_to_video(
            args, policy, scheduler, normalizer, train_args, seed, tmp_path
        )
        print(f"attempt={attempt + 1} seed={seed} success={success} steps={steps} reward={reward:.2f}")
        best = (tmp_path, success, steps, reward, seed)
        if success or not args.require_success:
            shutil.copy2(tmp_path, out_path)
            print(f"saved {out_path}")
            return

    if best is not None:
        tmp_path, success, steps, reward, seed = best
        shutil.copy2(tmp_path, out_path)
        print(f"saved last attempt {out_path} seed={seed} success={success} steps={steps} reward={reward:.2f}")


if __name__ == "__main__":
    main()
