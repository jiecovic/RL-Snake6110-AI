import os
import glob
import argparse
import torch
import numpy as np
import cv2
from typing import Literal, Union

from collections import defaultdict
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from core.envs.snake_envs import SnakePixelDirectionObsEnv, SnakePixelObsEnv, SnakePixelStackedEnv, SnakeFirstPersonEnv
from core.snake6110.snakegame import SnakeGame, MoveResult
from core.snake6110.level import EmptyLevel

# === CNN activation capture ===
activations = []
global_fps = [30]  # Mutable global FPS value that can be updated at runtime


def hook_fn(module, input, output):
    activations.append(output.detach().cpu())


def show_feature_map_opencv(tensor, name="CNN Activations"):
    fmap = tensor.squeeze(0)  # remove batch dimension
    num_channels, h, w = fmap.shape

    # Normalize and convert each channel to a color image
    images = []
    for i in range(num_channels):
        img = fmap[i].numpy()
        img = (img - img.min()) / (img.max() - img.min() + 1e-5)
        img = (img * 255).astype(np.uint8)
        img = cv2.resize(img, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
        img = cv2.applyColorMap(img, cv2.COLORMAP_VIRIDIS)
        images.append(img)

    # Determine grid size (rows x cols)
    cols = int(np.ceil(np.sqrt(num_channels)))
    rows = int(np.ceil(num_channels / cols))

    # Fill in blank images if needed
    blank = np.zeros_like(images[0])
    images += [blank] * (rows * cols - num_channels)

    # Stack into grid
    rows_imgs = [cv2.hconcat(images[i * cols:(i + 1) * cols]) for i in range(rows)]
    grid = cv2.vconcat(rows_imgs)

    cv2.imshow(name, grid)

    # === Handle key input for FPS control ===
    key = cv2.waitKey(1)
    if key == ord('+') or key == ord('='):
        global_fps[0] = min(global_fps[0] + 1, 600)
        print(f"\n⏫ Increased FPS to {global_fps[0]}")
    elif key == ord('-') or key == ord('_'):
        global_fps[0] = max(global_fps[0] - 1, 1)
        print(f"\n⏬ Decreased FPS to {global_fps[0]}")


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
        # Extract useful metadata (if available)
        keys_of_interest = ["snake_length", "score", "step", "episode", "food_count"]
        lines = []
        for k in keys_of_interest:
            if k in obs:
                lines.append(f"{k.replace('_', ' ').capitalize()}: {obs[k]}")
        overlay_text = "\n".join(lines)
    else:
        raw_obs = obs

    raw_obs = raw_obs.squeeze(0) if raw_obs.ndim == 4 else raw_obs  # (C, H, W)
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

        rgb = (rgb * 255).astype(np.uint8)
        rgb = cv2.resize(rgb, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
        output_img = rgb

    elif mode in ("vertical", "horizontal"):
        images = []
        for i in range(c):
            img = raw_obs[i]
            img = (img * 255).astype(np.uint8)
            img = cv2.resize(img, (w * 8, h * 8), interpolation=cv2.INTER_NEAREST)
            img = cv2.applyColorMap(img, cv2.COLORMAP_MAGMA)
            images.append(img)

        output_img = cv2.vconcat(images) if mode == "vertical" else cv2.hconcat(images)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    # === Overlay text (if available) ===
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


# === Argument parsing ===
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--subdir",
        type=str,
        required=True,
        help="Subdirectory name inside 'models/' to look for checkpoints (e.g. snake_pixel_dir_ppo)"
    )
    parser.add_argument(
        "--by",
        choices=["time", "number"],
        default="time",
        help="How to select the latest checkpoint: by 'time' or 'number'"
    )
    parser.add_argument("--fps", type=int, default=30, help="Initial FPS for human render mode")
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


def make_env():
    def _init():
        level = EmptyLevel(height=11 + 2, width=20 + 2)
        game = SnakeGame(level, food_count=1, fps=global_fps[0])
        # return SnakePixelStackedEnv(game, render_mode="human")
        return SnakeFirstPersonEnv(game, render_mode="human", view_radius=10)

    return _init


def register_cnn_hooks(model):
    activations.clear()
    cnn = model.policy.features_extractor.extractors["pixel"].cnn
    for idx, layer in enumerate(cnn):
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_forward_hook(hook_fn)


def run_loop(env, checkpoint_dir: str, selection_method: str):
    current_checkpoint = find_latest_checkpoint(checkpoint_dir, selection_method)
    model = PPO.load(current_checkpoint, env=env)
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

        # === Update env's FPS live
        env.envs[0].game.fps = global_fps[0]

        # === Visualize observation
        show_obs_opencv(obs)

        # === Visualize CNN activations
        for idx, act in enumerate(activations):
            show_feature_map_opencv(act, name=f"Conv Layer {idx}")

        move_results = info[0].get("move_results", [])
        episode_reward += reward[0]
        print(f"🎮 Episode {episode_id + 1} | Reward: {episode_reward:.2f} | FPS: {global_fps[0]}", end="\r")

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
    global_fps[0] = args.fps

    # === NEW: Use subfolder argument ===
    checkpoint_dir = os.path.join("models", args.subdir, "checkpoints")
    assert os.path.exists(checkpoint_dir), f"Checkpoint not found: {checkpoint_dir}"

    env = DummyVecEnv([make_env()])
    run_loop(env, checkpoint_dir, args.by)


if __name__ == "__main__":
    main()
