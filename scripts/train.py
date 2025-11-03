import os
import json
import argparse
from typing import Dict
import glob
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from core.envs.snake_envs import SnakeFirstPersonEnv
from core.snake6110.snakegame import SnakeGame
from core.snake6110.level import EmptyLevel
from core.policies.custom_policies import CustomCombinedExtractor
from core.custom_cnns.custom_cnn import SnakeCNN_Simple
from core.custom_callbacks import TerminationCauseLogger


# === Config dict (single source of truth) ===
config = {
    "base_run_name": "snake_ppo",
    "num_envs": 16,
    "total_timesteps": 100_000_000,
    "seed": 42,
    "checkpoint_freq": 1_000_000,
    "resume_checkpoint": None,  # Optional override via CLI

    "env_params": {
        "height": 13,
        "width": 22,
        "food_count": 1,
    },

    "snake_env_params": {
        "render_mode": "none",
        "n_stack": 1,
        "view_radius": 10
    },

    "env_classes": {
        "level": EmptyLevel,
        "game": SnakeGame,
        "env": SnakeFirstPersonEnv
    },

    "ppo_params": {
        "policy": MultiInputActorCriticPolicy,
        "n_steps": 2048,
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
                "cnn_constructor": SnakeCNN_Simple,
                "cnn_output_dim": 512
            }
        }
    }
}


# === CLI + resume resolution ===

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None,
                        help="Checkpoint path, run dir, or 'latest:run_name'")
    return parser.parse_args()


def resolve_checkpoint_path(resume_arg: str) -> str:
    if resume_arg.endswith(".zip") and os.path.isfile(resume_arg):
        return resume_arg

    if resume_arg.startswith("latest:"):
        run_name = resume_arg.split("latest:")[1]
        run_dir = os.path.join("models", run_name, "checkpoints")
    elif os.path.isdir(resume_arg):
        run_dir = os.path.join(resume_arg, "checkpoints")
    else:
        raise FileNotFoundError(f"Could not interpret resume path: {resume_arg}")

    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {run_dir}")

    checkpoints = glob.glob(os.path.join(run_dir, "*.zip"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in: {run_dir}")

    latest = max(checkpoints, key=os.path.getmtime)
    print(f"🔁 Resolved latest checkpoint: {latest}")
    return latest


# === Manifest + utility ===

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


# === Main ===

if __name__ == "__main__":
    args = parse_args()
    if args.resume:
        resolved_path = resolve_checkpoint_path(args.resume)
        config["resume_checkpoint"] = resolved_path

    set_random_seed(config["seed"])

    def make_env(rank: int, seed: int = 0):
        def _init():
            env_conf = config["env_params"]
            snake_env_conf = config["snake_env_params"]
            cls_conf = config["env_classes"]

            level = cls_conf["level"](height=env_conf["height"], width=env_conf["width"])
            game = cls_conf["game"](level, food_count=env_conf["food_count"], fps=0)
            env = cls_conf["env"](
                game,
                render_mode=snake_env_conf["render_mode"],
                n_stack=snake_env_conf["n_stack"],
                view_radius=snake_env_conf["view_radius"]
            )
            env.reset(seed=seed + rank)
            return env
        return _init

    print("=" * 40)
    print(" 🐍 Starting PPO training for Snake ")
    print("=" * 40)

    env_fns = [make_env(i, config["seed"]) for i in range(config["num_envs"])]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecMonitor(vec_env)

    if config["resume_checkpoint"]:
        print(f"📦 Resuming from checkpoint: {config['resume_checkpoint']}")
        model = PPO.load(config["resume_checkpoint"], env=vec_env)
    else:
        model = PPO(
            env=vec_env,
            **{k: v for k, v in config["ppo_params"].items() if k != "policy"},
            policy=config["ppo_params"]["policy"],
        )

    label_width = 22
    print("-" * 40)
    print("📊 Environment Info")
    print("-" * 40)
    print(f"{'Num Envs':<{label_width}}: {config['num_envs']}")
    print(f"{'Obs Space':<{label_width}}: {vec_env.observation_space}")
    print(f"{'Action Space':<{label_width}}: {vec_env.action_space}")
    print("-" * 40)
    print("✅ PPO Model Info")
    print("-" * 40)
    print(f"{'Policy Class':<{label_width}}: {model.policy_class.__name__}")
    print(f"{'Ent Coeff':<{label_width}}: {model.ent_coef}")
    print(f"{'Learning Rate':<{label_width}}: {model.lr_schedule(1e5)}")
    print(f"{'Batch Size':<{label_width}}: {model.batch_size}")
    print(f"{'Steps per Update':<{label_width}}: {model.n_steps}")
    print(f"{'Gamma':<{label_width}}: {model.gamma}")
    print(f"{'GAE Lambda':<{label_width}}: {model.gae_lambda}")
    print(f"{'Clip Range':<{label_width}}: {model.clip_range(1e5)}")
    print(f"{'Max Grad Norm':<{label_width}}: {model.max_grad_norm}")
    print("-" * 40)
    print("🦾 Model + Device Info")
    print("-" * 40)
    print(f"{'🖥️  Model Device':<{label_width}}: {model.device}")
    print("-" * 40)
    print("🔥 PyTorch & CUDA Info")
    print("-" * 40)
    print(f"{'📦 PyTorch Version':<{label_width}}: {torch.__version__}")
    print(f"{'⚙️  CUDA Available':<{label_width}}: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"{'🎮 GPU Name':<{label_width}}: {torch.cuda.get_device_name(0)}")
        print(f"{'🧮 CUDA Version (built)':<{label_width}}: {torch.version.cuda}")
        print(f"{'🔧 cuDNN Version':<{label_width}}: {torch.backends.cudnn.version()}")
    else:
        print(f"{'🚫 CUDA Status':<{label_width}}: No CUDA-enabled GPU found.")

    print("\n🦾 Policy:")
    print(model.policy)

    start = input("Do you want to start training? [Y/n]: ").strip().lower()
    if start not in ("", "y", "yes"):
        print("❌ Training aborted.")
        vec_env.close()
        exit(0)

    run_dir = get_unique_run_dir(config["base_run_name"])
    print(f"Run directory: {run_dir}")

    log_dir = run_dir
    model_dir = os.path.join("models", os.path.basename(run_dir))
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    tensorboard_log_dir = log_dir

    for path in (log_dir, model_dir, checkpoint_dir):
        os.makedirs(path, exist_ok=True)

    config["ppo_params"]["tensorboard_log"] = tensorboard_log_dir
    config["checkpoint_dir"] = checkpoint_dir
    config["model_dir"] = model_dir
    save_config_manifest(config, run_dir)

    if not config["resume_checkpoint"]:
        model.tensorboard_log = tensorboard_log_dir

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
