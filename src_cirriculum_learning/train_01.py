import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym
from customEnv_01 import CircuitEnv  # Replace with your file name

# Create environment
def create_env():
    return CircuitEnv(max_components=15, value_buckets=5, folder_path="../data")

# Using vectorized environments for faster training
env = make_vec_env(create_env, n_envs=1)

# Define the policy network architecture if needed
policy_kwargs = dict(
    net_arch=[dict(pi=[256, 256], vf=[256, 256])]
)

# Initialize the PPO agent
model = PPO(
    "MultiInputPolicy",   # because observation_space is a Dict
    env,
    verbose=1,
    learning_rate=3e-4,
    batch_size=64,
    n_steps=2048,
    policy_kwargs=policy_kwargs,
    tensorboard_log="./ppo_circuit_tensorboard/"
)

# Callbacks for saving models and evaluation
checkpoint_callback = CheckpointCallback(save_freq=5000, save_path='./models/', name_prefix='ppo_circuit')
eval_callback = EvalCallback(env, best_model_save_path='./best_model/',
                             log_path='./eval_logs/', eval_freq=10000,
                             deterministic=True, render=False)

# Train the agent
model.learn(total_timesteps=1_000_000, callback=[checkpoint_callback, eval_callback])

# Save the final model
model.save("ppo_circuit_final")

print("Training completed. Model saved!")
