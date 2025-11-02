import os
import glob
import argparse
import torch
import numpy as np
import cv2
from typing import Literal, Union

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.envs.snake_envs import SnakeFirstPersonEnv
from core.snake6110.snakegame import SnakeGame, MoveResult
from core.snake6110.level import EmptyLevel

# === CNN activation capture ===
activations = []


def hook_fn(module, input, output):
    activations.append(output.detach().cpu())


def show_feature_map_opencv(tensor, name="CNN Activations"):
    fmap = tensor.squeeze(0)
    num_channels, h, w = fmap.shape

    images = []
    for i in range(num_channels):
        img = fmap[i].numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        images.append(img)

    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    blank = np.zeros_like(images[0])
    images += [blank] * (rows * cols - num_channels)

    rows_imgs = [cv2.hconcat(images[i * cols:(i + 1) * cols]) for i in range(rows)]
    grid = cv2.vconcat(rows_imgs)

    cv2.imshow(name, grid)
    cv2.waitKey(1)


def show_obs_opencv(
    obs: Union[np.ndarray, dict],
    name: str = "Observation",
    mode: Literal["channel", "vertical", "horizontal"] = "horizontal",
) -> None:
    overlay_text = ""
    if isinstance(obs, dict):
        if "pixel" not in obs:
            raise ValueError("Missing 'pixel' key in observation dict.")
        raw_obs = obs["pixel"]
        keys_of_interest = ["snake_length", "score", "step", "episode", "food_count"]
        lines = [f"{k.replace('_', ' ').capitalize()}: {obs[k]}" for k in keys_of_interest if k in obs]
        overlay_text = "\n".join(lines)
    else:
        raw_obs = obs

    raw_obs = raw_obs.squeeze(0) if raw_obs.ndim == 4 else raw_obs
    c, h, w = raw_obs.shape
    raw_obs = raw_obs.astype(np.float32)
    raw_obs = (raw_obs - raw_obs.min()) / (raw_obs.max() - raw_obs.min() + 1e-5)

    if mode == "channel":
        if c == 1:
            rgb = np.stack([raw_obs[0]] * 3, axis=-1)
        elif c == 2:
            rgb = np.stack([raw_obs[0], raw_obs[1], np.zeros_like(raw_obs[0])], axis=-1)
        else:
            rgb = np.stack([raw_obs[0], raw_obs[c // 2], raw_obs[-1]], axis=-1)
        output_img = (rgb * 255).astype(np.uint8)
        output_img = cv2.resize(output_img, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
    elif mode in ("vertical", "horizontal"):
        images = []
        for i in range(c):
            img = (raw_obs[i] * 255).astype(np.uint8)
            img = cv2.resize(img, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
            images.append(img)
        output_img = cv2.vconcat(images) if mode == "vertical" else cv2.hconcat(images)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if overlay_text:
        for i, line in enumerate(overlay_text.splitlines()):
            cv2.putText(
                output_img,
                line,
                (10, 25 + i * 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.8,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

    cv2.imshow(name, output_img)
    cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdir", type=str, required=True,
                        help="Subdirectory inside 'models/' to load checkpoints from.")
    parser.add_argument("--by", choices=["time", "number"], default="time",
                        help="Select latest checkpoint by 'time' or 'number'")
    parser.add_argument("--fps", type=int, default=30,
                        help="Fixed FPS for SnakeGame playback")
    parser.add_argument("--show-cnn", action="store_true", help="Visualize CNN activations")
    parser.add_argument("--show-obs", action="store_true", help="Visualize observation image")
    return parser.parse_args()


def extract_checkpoint_number(path: str) -> int:
    basename = os.path.basename(path)
    digits = "".join(c for c in basename if c.isdigit())
    return int(digits) if digits else -1


def find_latest_checkpoint(directory: str, method: str) -> str:
    checkpoint_files = glob.glob(os.path.join(directory, "*.zip"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint found.")
    return max(checkpoint_files, key=(extract_checkpoint_number if method == "number" else os.path.getmtime))


def make_env(fps: int):
    def _init():
        level = EmptyLevel(height=13, width=22)
        game = SnakeGame(level, food_count=1, fps=fps)
        return SnakeFirstPersonEnv(game, render_mode="human", view_radius=10, n_stack=1)
    return _init


def register_cnn_hooks(model):
    activations.clear()
    cnn = model.policy.features_extractor.extractors["pixel"].cnn
    for layer in cnn:
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(hook_fn)


def run_loop(env, checkpoint_dir: str, selection_method: str, args):
    current_checkpoint = find_latest_checkpoint(checkpoint_dir, selection_method)
    model = PPO.load(current_checkpoint, env=env)
    print(model.device)
    print("CUDA available:", torch.cuda.is_available())
    register_cnn_hooks(model)

    print("=" * 50)
    print(f"🧠 Loaded checkpoint ({selection_method}): {os.path.basename(current_checkpoint)}")
    print("=" * 50)

    obs = env.reset()
    episode_id = 0
    episode_reward = 0.0

    while True:
        activations.clear()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if args.show_obs:
            show_obs_opencv(obs)

        if args.show_cnn:
            for idx, act in enumerate(activations):
                show_feature_map_opencv(act, name=f"Conv Layer {idx}")

        move_results = info[0].get("move_results", [])
        episode_reward += reward[0]
        print(f"🎮 Episode {episode_id + 1} | Reward: {episode_reward:.2f}", end="\r")

        if done[0]:
            episode_id += 1
            info_dict = info[0]
            print("\n" + "-" * 40)
            print(f"🏁 Episode {episode_id} ended")
            print(f"Cause: {info_dict.get('termination_cause', 'unknown')}")
            print(f"Score: {info_dict.get('final_score', 'n/a')}")
            print(f"Results: {', '.join(r.name for r in move_results)}")
            print("-" * 40 + "\n")

            latest = find_latest_checkpoint(checkpoint_dir, selection_method)
            if latest != current_checkpoint:
                print(f"🔁 Reloading newer checkpoint: {os.path.basename(latest)}")
                current_checkpoint = latest
                model = PPO.load(current_checkpoint, env=env)
                register_cnn_hooks(model)

            episode_reward = 0.0
            obs = env.reset()


def main():
    args = parse_args()
    checkpoint_dir = os.path.join("models", args.subdir, "checkpoints")
    assert os.path.exists(checkpoint_dir), f"Checkpoint not found: {checkpoint_dir}"

    env = DummyVecEnv([make_env(args.fps)])
    run_loop(env, checkpoint_dir, args.by, args)


if __name__ == "__main__":
    main()
