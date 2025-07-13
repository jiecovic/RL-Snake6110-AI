import os
import glob
import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from core.envs.snake_envs import SnakePixelDirectionObsEnv
from core.snake6110.snakegame import SnakeGame
from core.snake6110.level import EmptyLevel

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--by", choices=["time", "number"], default="time",
                        help="How to select the latest checkpoint: by 'time' (default) or by 'number')")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS for human render mode (default: 30)")
    return parser.parse_args()

def extract_checkpoint_number(path: str) -> int:
    basename = os.path.basename(path)
    digits = "".join(c for c in basename if c.isdigit())
    return int(digits) if digits else -1

def find_latest_checkpoint(directory: str, method: str) -> str:
    checkpoint_files = glob.glob(os.path.join(directory, "*.zip"))
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint found in the checkpoint directory.")
    if method == "number":
        return max(checkpoint_files, key=extract_checkpoint_number)
    else:
        return max(checkpoint_files, key=os.path.getmtime)

def make_env(fps: int):
    def _init():
        level = EmptyLevel(height=11, width=20)
        game = SnakeGame(level, food_count=1, fps=fps)
        env = SnakePixelDirectionObsEnv(game, render_mode="human")
        return env
    return _init

def run_loop(env, checkpoint_dir: str, selection_method: str):
    current_checkpoint = find_latest_checkpoint(checkpoint_dir, selection_method)
    model = PPO.load(current_checkpoint, env=env)

    print("=" * 50)
    print(f"🧠 Loaded checkpoint ({selection_method}): {os.path.basename(current_checkpoint)}")
    print("=" * 50)

    obs = env.reset()
    episode_id = 0
    episode_reward = 0.0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        episode_reward += reward[0]
        print(f"🎮 Episode {episode_id + 1} | Reward: {episode_reward:.2f}", end="\r", flush=True)

        if done[0]:
            episode_id += 1
            episode_info = info[0]
            score = episode_info.get("final_score", "n/a")
            cause = episode_info.get("termination_cause", "unknown")

            print("\n" + "-" * 40)
            print(f"🏁 Episode {episode_id} finished")
            print(f"   → Termination Cause : {cause}")
            print(f"   → Final Score       : {score}")
            print("-" * 40 + "\n")

            # Check for newer checkpoint after episode ends
            latest_checkpoint = find_latest_checkpoint(checkpoint_dir, selection_method)
            if latest_checkpoint != current_checkpoint:
                print(f"🔁 Newer checkpoint found: {os.path.basename(latest_checkpoint)}. Reloading model...")
                current_checkpoint = latest_checkpoint
                model = PPO.load(current_checkpoint, env=env)

            episode_reward = 0.0
            obs = env.reset()

def main():
    args = parse_args()
    checkpoint_dir = "models/snake_pixel_dir_ppo/checkpoints"

    assert os.path.exists(checkpoint_dir), f"Checkpoint folder not found: {checkpoint_dir}"

    env = DummyVecEnv([make_env(args.fps)])
    run_loop(env, checkpoint_dir, args.by)

if __name__ == "__main__":
    main()
