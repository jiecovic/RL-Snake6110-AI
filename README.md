# 🐍 Snake RL — PPO Agent for Classic Nokia-Style Snake

This project reproduces the **classic Snake game from the Nokia 6110** (original grid size and feel)  
and explores how **reinforcement learning agents** can learn to play it using **Proximal Policy Optimization (PPO)**.

The repository is intentionally designed as an **experimental RL playground**: a place to try different
observation representations, network architectures, and training setups while keeping the underlying
game mechanics fixed and reproducible.

The ambition is **learning directly from pixels**, but the project also includes
**symbolic and hybrid observation pipelines** used for experimentation and architectural comparisons with different feature extraction architectures.

---

## ✨ Highlights

- 🎮 **Faithful Snake implementation** (PyGame)
- 🤖 **PPO agents** (Stable-Baselines3 + Gymnasium)
- 🧠 **Multiple observation paradigms**
  - Raw pixel-based inputs
  - Snake-centric POV grids
  - Symbolic / embedded representations (for experimental architectures)
- 🧪 **Current feature extraction models**
  - CNN-based feature extractors
  - Vision Transformers (ViT)
  - Embedding + MLP variants
- 📦 **Reproducible runs via config snapshots**
- 📊 **TensorBoard logging**
- 👀 **Live watch mode with hot checkpoint reload**

---

## 🎯 Objective

The **ultimate objective** of this project is to train a reinforcement learning agent
**purely from visual input (pixels)**, without relying on handcrafted state abstractions,
world models, or privileged game information.

At the same time, the codebase deliberately supports **experimental alternatives**, including:
- symbolic grid encodings,
- compact embeddings,
- hybrid POV representations,

to study their impact on learning speed, stability, and asymptotic performance.

The agent should learn to:
- efficiently collect food,
- avoid self and wall collisions,
- and ultimately **win the game** by filling the entire grid,

with all behavior emerging from interaction with the environment and the reward signal.

This makes the repository less a single “final solution” and more a **controlled research and
engineering sandbox for RL experiments on a classic game**.

---

## 🧠 Reward Shaping

The environment's `step(action)` returns rewards as follows:

- **Step penalty** (encourage efficiency):  
  `reward -= tiny_reward`
- **Food eaten**:
  - `+1.0` base reward
  - **Speed bonus** up to `+2.0`, scaled by how quickly food is found:
    ```
    reward += 2 * (1 - steps_since_last_food / max_steps)
    ```
  - resets food counters and visited-node tracking
- **Fatal move** (wall/self collision):  
  `−5.0`, episode terminates
- **Win condition** (board fully filled):  
  `+10.0`, episode terminates
- **Timeout / truncation**:  
  episode truncates if `steps_since_last_food >= max_steps`

The `info` dict includes:
- `final_score`
- `termination_cause` (human-readable)

---

## 🚀 Usage

All entry points are installed as **CLI tools** via `pyproject.toml`.

### 🏋️ Train an Agent

Training uses **Hydra + Pydantic** configs under `configs/`.

Note: only PPO is implemented today (train.algo.type=ppo).

Run an example main config:

```
snake-train --config-name main_pov_pixel_ppo
```

Common overrides:

```
snake-train --config-name main_pov_pixel_ppo run.seed=123 run.num_envs=8 run.total_timesteps=5_000_000
```

🔒 **Reproducibility note**  
At training start, configuration artifacts are written to:

```
runs/<run_id>/config_snapshot.yaml
runs/<run_id>/config_hydra.yaml
runs/<run_id>/config_validated.yaml
```

`config_snapshot.yaml` is the **single source of truth** for:
- evaluation
- watch mode
- reproducing the run

---

### 📊 Evaluate a Trained Run

```
snake-eval --run snake_ppo_001
```

Options:

```
snake-eval --run snake_ppo_001 --which best --episodes 100
```

---

### 👀 Watch a Trained Agent (Live Reload)

```
snake-watch --run snake_ppo_001
```

Hot reload during training:

```
snake-watch --run snake_ppo_001 --reload 30
```

---

### 🕹️ Play Snake as a Human

```
snake-play
```

Custom settings:

```
snake-play --width 30 --height 20 --fps 15
```

---

## 🧪 Configuration Files

Hydra config groups live in `configs/`:
- `env/` (env id + params + observation)
- `level/` (grid sizes)
- `reward/` (reward shaping)
- `feature_extractor/` (feature extractor params)
- `train/` (algorithm + eval scheduling)
- `run/` (seed, num_envs, total_timesteps, checkpointing)

Main configs live at the top level: `configs/main_*.yaml`.

Policy MLP `net_arch` now lives in `train.algo.params.policy_kwargs.net_arch`.
`train.eval` applies to both periodic evals during training and the final eval pass.

⚠️ **Important**  
The provided example configs are **experimental** and intended as research starting points.

---

## 📂 Run Structure

```
runs/
└── snake_ppo_001/
    ├── config_snapshot.yaml
    ├── config_hydra.yaml
    ├── config_validated.yaml
    ├── checkpoints/
    │   ├── latest.zip
    │   ├── best.zip
    │   └── final.zip
    ├── tb/
    ├── eval_final.json
    └── status.txt
```

Legacy runs under `experiments/` are still supported for loading.

---

## 📦 Installation

```
pip install .
```


PyTorch install:
https://pytorch.org/get-started/locally/

---

## 📝 License

MIT License — see LICENSE







