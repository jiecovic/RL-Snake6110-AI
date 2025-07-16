# train.py
import os
from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv, VecFrameStack
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

from collections import defaultdict
from typing import List

from core.envs.snake_envs import SnakePixelDirectionObsEnv, SnakePixelObsEnv, SnakePixelStackedEnv, SnakeFirstPersonEnv
from core.snake6110.snakegame import SnakeGame
from core.snake6110.level import EmptyLevel
from core.policies.custom_policies import CustomCombinedExtractor
from core.custom_cnns.custom_cnn import SnakeCNN_3Layers, SnakeCNN_Deep
from core.custom_callbacks import TerminationCauseLogger

# === Config ===
RUN_NAME = "snake_pixel_dir_ppo"
NUM_ENVS = 8
TOTAL_TIMESTEPS = 100_000_000
SEED = 42
CHECKPOINT_FREQ = 500_000 // NUM_ENVS

LOG_ROOT = "logs"
MODEL_ROOT = "models"
CHECKPOINT_ROOT = "checkpoints"

LOG_DIR = os.path.join(LOG_ROOT, RUN_NAME)
MODEL_DIR = os.path.join(MODEL_ROOT, RUN_NAME)
CHECKPOINT_DIR = os.path.join(MODEL_DIR, CHECKPOINT_ROOT)

TENSORBOARD_LOG_DIR = LOG_DIR  # SB3 uses this separately

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def make_env(rank: int, seed: int = 0):
    def _init():
        level = EmptyLevel(height=13, width=22)
        game = SnakeGame(level, food_count=1, fps=0)
        # env = SnakePixelDirectionObsEnv(game, render_mode="none")
        # env = SnakePixelObsEnv(game, render_mode="none")
        # env = SnakePixelStackedEnv(game, render_mode="none", n_stack=2)
        env = SnakeFirstPersonEnv(game, render_mode="none", n_stack=2, view_radius=10)
        env.reset(seed=seed + rank)
        return env

    return _init


if __name__ == "__main__":
    set_random_seed(SEED)

    print("=" * 40)
    print(" 🐍 Starting PPO training for Snake ")
    print("=" * 40)

    # Create vectorized environment
    env_fns = [make_env(rank=i, seed=SEED) for i in range(NUM_ENVS)]
    vec_env = SubprocVecEnv(env_fns)
    # vec_env = DummyVecEnv(env_fns)

    # vec_env = VecFrameStack(vec_env, n_stack=2, channels_order='first')
    vec_env = VecMonitor(vec_env)

    # Print environment info
    print(f"Num Envs         : {NUM_ENVS}")
    print(f"Obs Space        : {vec_env.observation_space}")
    print(f"Action Space     : {vec_env.action_space}")

    # Set up SB3 logger (TensorBoard only)
    # logger = configure(TENSORBOARD_LOG_DIR, ["tensorboard"])

    # Create PPO model with progress bar enabled
    model = PPO(
        policy=MultiInputActorCriticPolicy,
        # policy="CnnPolicy",
        ent_coef=0.03,
        env=vec_env,
        n_steps=8 * 2048 // NUM_ENVS,
        batch_size=128,
        n_epochs=2,
        gamma=0.999,
        learning_rate=3e-4,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        verbose=0,
        seed=SEED,
        # policy_kwargs=dict(
        #     features_extractor_class=SnakeCNN_3Layers,
        #     features_extractor_kwargs=dict(features_dim=512),
        #     net_arch=[256, 128],
        # )
        policy_kwargs=dict(
            net_arch=[256, 128],
            # net_arch=[128, 64],
            features_extractor_class=CustomCombinedExtractor,
            features_extractor_kwargs=dict(
                cnn_constructor=SnakeCNN_3Layers,
                # cnn_constructor=SnakeCNN_Deep,
                cnn_output_dim=512
            )
        )
    )
    # model.set_logger(logger)

    # Print PPO settings
    print("-" * 40)
    print("✅ PPO Model Info")
    print("-" * 40)
    print(f"Policy Class     : {model.policy_class.__name__}")
    print(f"Ent Coeff        : {model.ent_coef}")
    print(f"Learning Rate    : {model.lr_schedule(1e5)}")
    print(f"Batch Size       : {model.batch_size}")
    print(f"Steps per Update : {model.n_steps}")
    print(f"Gamma            : {model.gamma}")
    print(f"GAE Lambda       : {model.gae_lambda}")
    print(f"Clip Range       : {model.clip_range}")
    print(f"Max Grad Norm    : {model.max_grad_norm}")
    print("-" * 40)

    print(model.policy)


    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=CHECKPOINT_DIR,
        name_prefix="ppo_checkpoint"
    )

    callbacks = CallbackList([
        checkpoint_callback,
        TerminationCauseLogger()
    ])

    # Ask for confirmation before starting training
    start = input("Do you want to start training? [Y/n]: ").strip().lower()
    if start not in ("", "y", "yes"):
        print("❌ Training aborted.")
        vec_env.close()
        exit(0)

    print("🚀 Starting training...")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
        callback=callbacks,
        tb_log_name=f"ppo_{RUN_NAME}"
    )

    print("💾 Saving final model...")
    model.save(os.path.join(MODEL_DIR, "ppo_final"))

    vec_env.close()
    print("✅ Training complete and environment closed.")
