# RL-Snake6110-AI

# 🐍 Snake RL — PPO Agent for Classic Nokia-Style Snake

This project is my first attempt to **reproduce the famous Snake game from the Nokia 6110** —  
same classic grid size and window layout — and to **train a reinforcement learning agent** to play it using **Proximal Policy Optimization (PPO)**.

## Overview

- **Engine:** PyGame  
- **Algorithm:** PPO (using Stable-Baselines3 and Gymnasium)  
- **Goal:** Learn to efficiently collect food, avoid fatal collisions, and ultimately **win the game** by filling the entire grid with the snake’s body tiles.  
- **TensorBoard support** included for performance tracking.


### Reward shaping 

The environment’s `step(action)` returns rewards as follows:

- **Tiny step penalty** every move to encourage efficiency: `reward -= tiny_reward`.
- **Food eaten:**
  - `+1.0` base,
  - **speed bonus** up to `+2.0`, scaled by how quickly food is found since the last bite: `reward += 2 * (1 - steps_since_last_food / max_steps)`
  - resets `visited_nodes` and the food-step counter.
- **Fatal move** (wall/self/etc.): `−5.0` and **terminate**.
- **Win** (board cleared / win condition): `+10.0`, terminate (no penalties).
- **Timeout/truncation:** episode **truncates** when `steps_since_last_food >= max_steps`.

`info` includes `final_score` and a human-readable `termination_cause` when an episode ends.  
Rendering is called each step for visual playback.


---

## 🚀 Usage

### 🏋️‍♂️ Train the Agent
Start PPO training from scratch:
```bash
python -m scripts.train
```
Training logs and model checkpoints will be saved automatically under a timestamped subdirectory.

---

### 🎮 Run a Trained Checkpoint
Render or evaluate a saved model:
```bash
python -m scripts.run_checkpoint --subdir snake_ppo_x
```
Replace `snake_ppo_x` with the name of your training run folder (e.g. `snake_ppo_3`, `snake_ppo_best`, etc.).

---


### 🐍 Run the Pretrained Model (Included in Repo)
A pretrained PPO agent is provided under the repository folder `snake_test/`.

Run it directly to watch the trained model play:

```bash
python -m scripts.run_checkpoint --subdir snake_test
```
This will load the existing checkpoint from snake_test/ and start a rendering session in the PyGame window.

---

### 🎮 Demo Video

Demonstration of a fully trained agent performing a complete successful run.

[▶️ Watch Demo Video](demo-videos/FullRun-Demo-Win.mp4)

## 🎮 Demo Gif
![Snake RL Demo](snake_demo.gif)




## 🔮 Future Work & Ideas

- **Extend to other grid sizes and layouts** — larger arenas, variable resolutions, or grids with dynamic obstacles.  
- **Introduce new game rules or mechanics** — e.g., multiple foods, or wall generation for added complexity.  
- **Experiment with different observation spaces** — current setup uses a *snake-centric (POV)* observation, which scales well to varying map sizes, but global or hybrid states could offer richer representations.  
- **Evaluate alternative RL algorithms** — TBD
- **Hyperparameter Optimization (HPO) and Fine-Tuning** — automate PPO parameter search (learning rate, entropy coef, rollout size, etc.) and fine-tune trained agents for stability and better asymptotic performance.
- **Explore model-based extensions** — planning or world-model approaches to improve sample efficiency; current agent is purely *model-free*.  
- **Curriculum learning** — gradually increase grid size or introduce obstacles during training to accelerate convergence.  
  *(This was already explored in an earlier draft version and yielded promising results — worth revisiting for future iterations.)*  


