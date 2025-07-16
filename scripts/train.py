import os
import json
from typing import Dict

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from core.envs.snake_envs import SnakeFirstPersonEnv
from core.snake6110.snakegame import SnakeGame
from core.snake6110.level import EmptyLevel
from core.policies.custom_policies import CustomCombinedExtractor
from core.custom_cnns.custom_cnn import SnakeCNN_3Layers
from core.custom_callbacks import TerminationCauseLogger

# === Config dict (single source of truth) ===
config = {
    "base_run_name": "snake_ppo",
    "num_envs": 8,
    "total_timesteps": 100_000_000,
    "seed": 42,
    "checkpoint_freq": 500_000,
    "env_params": {
        "height": 13,
        "width": 22,
        "food_count": 1,
    },
    "ppo_params": {
        "policy": MultiInputActorCriticPolicy,
        "n_steps": 8 * 2048 // 8,
        "batch_size": 128,
        "n_epochs": 2,
        "gamma": 0.999,
        "ent_coef": 0.03,
        "learning_rate": 3e-4,
        "verbose": 0,
        "seed": 42,
        "policy_kwargs": {
            "net_arch": [256, 128],
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "cnn_constructor": SnakeCNN_3Layers,
                "cnn_output_dim": 512
            }
        }
    }
}


def serialize_classes(obj):
    if isinstance(obj, dict):
        return {k: serialize_classes(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_classes(i) for i in obj]
    elif isinstance(obj, type):
        return obj.__name__
    else:
        return obj


def get_unique_run_dir(base_name: str, base_path: str = "logs") -> str:
    i = 1
    while True:
        candidate = f"{base_name}_{i}"
        full_path = os.path.join(base_path, candidate)
        if not os.path.exists(full_path):
            return full_path
        i += 1


def save_config_manifest(config: Dict, run_dir: str):
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(serialize_classes(config), f, indent=4)
    print(f"[INFO] Saved run manifest to {manifest_path}")


if __name__ == "__main__":
    # Setup RNG seed
    set_random_seed(config["seed"])

    # Environment factory
    def make_env(rank: int, seed: int = 0):
        def _init():
            env_conf = config["env_params"]
            level = EmptyLevel(height=env_conf["height"], width=env_conf["width"])
            game = SnakeGame(level, food_count=env_conf["food_count"], fps=0)
            env = SnakeFirstPersonEnv(game, render_mode="none", n_stack=2, view_radius=10)
            env.reset(seed=seed + rank)
            return env
        return _init

    print("=" * 40)
    print(" 🐍 Starting PPO training for Snake ")
    print("=" * 40)

    env_fns = [make_env(i, config["seed"]) for i in range(config["num_envs"])]
    # vec_env = DummyVecEnv(env_fns)
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    print(f"Num Envs         : {config['num_envs']}")
    print(f"Obs Space        : {vec_env.observation_space}")
    print(f"Action Space     : {vec_env.action_space}")

    model = PPO(
        env=vec_env,
        **{k: v for k, v in config["ppo_params"].items() if k != "policy"},
        policy=config["ppo_params"]["policy"],
    )

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

    # === Confirm ===
    start = input("Do you want to start training? [Y/n]: ").strip().lower()
    if start not in ("", "y", "yes"):
        print("❌ Training aborted.")
        vec_env.close()
        exit(0)

    # === Create folders AFTER confirmation ===
    run_dir = get_unique_run_dir(config["base_run_name"])
    print(f"Run directory: {run_dir}")

    log_dir = run_dir
    model_dir = os.path.join("models", os.path.basename(run_dir))
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    tensorboard_log_dir = log_dir

    for path in (log_dir, model_dir, checkpoint_dir):
        os.makedirs(path, exist_ok=True)

    # Inject dynamic paths into config
    config["ppo_params"]["tensorboard_log"] = tensorboard_log_dir
    config["checkpoint_dir"] = checkpoint_dir
    config["model_dir"] = model_dir

    save_config_manifest(config, run_dir)

    # Update model tensorboard log path
    model.tensorboard_log = tensorboard_log_dir

    # Callbacks
    callbacks = CallbackList([
        CheckpointCallback(
            save_freq=config["checkpoint_freq"] // config["num_envs"],
            save_path=checkpoint_dir,
            name_prefix="ppo_checkpoint"
        ),
        TerminationCauseLogger()
    ])

    print("🚀 Starting training...")
    model.learn(
        total_timesteps=config["total_timesteps"],
        progress_bar=True,
        callback=callbacks,
        tb_log_name=f"ppo_{os.path.basename(run_dir)}"
    )

    print("💾 Saving final model...")
    model.save(os.path.join(model_dir, "ppo_final"))
    vec_env.close()

    print("✅ Training complete and environment closed.")
