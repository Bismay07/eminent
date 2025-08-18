import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from DrivingEnvHybrid import DrivingEnvHybrid  # Ensure this is in the same folder

# ---------------- Config ----------------
TIMESTEPS = 500_000          # total training steps
CHECKPOINT_FREQ = 50_000     # save every N steps
CHECKPOINT_DIR = "checkpoints"
MODEL_SAVE_PATH = "ppo_driving_final"

# Make checkpoint folder if it doesn't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------------- Create environment ----------------
# Vectorized env speeds up PPO training
env = make_vec_env(
    lambda: DrivingEnvHybrid(render_mode="rgb_array", obs_size=96, num_rays=24, screen_size=800, render_every=10),
    n_envs=4
)

# ---------------- PPO Model ----------------
model = PPO(
    "MultiInputPolicy",  # Supports dict observation spaces {"image":..., "sensors":...}
    env,
    verbose=1,
    tensorboard_log="./ppo_driving_tensorboard",
    learning_rate=2.5e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01,
)

# ---------------- Checkpoint Callback ----------------
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_driving"
)

# ---------------- Train ----------------
model.learn(
    total_timesteps=TIMESTEPS,
    callback=checkpoint_callback
)

# Save final model
model.save(MODEL_SAVE_PATH)
print(f"Training finished. Model saved to {MODEL_SAVE_PATH}.zip")

# ---------------- Optional: Test ----------------
obs = env.reset()
for _ in range(500):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated.any() or truncated.any():
        obs = env.reset()
