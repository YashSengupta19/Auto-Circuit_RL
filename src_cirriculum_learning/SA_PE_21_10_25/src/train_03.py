import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from customEnv_03 import CircuitEnv
from feature_extractor_network import MultiInputFeatureExtractor

# === Function to create environments ===
def make_env(env_id: int):
    def _init():
        return CircuitEnv(work_dir=f"ltspice_env_{env_id}")
    return _init

if __name__ == "__main__":   # Critical on Windows

    n_envs = 10  # Number of parallel environments
    env_fns = [make_env(i) for i in range(n_envs)]
    
    # Create vectorized environment
    env = SubprocVecEnv(env_fns)

    # === Define policy configuration ===
    policy_kwargs = dict(
        features_extractor_class=MultiInputFeatureExtractor,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    # === Initialize PPO model ===
    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=256,   # Increase batch size for multiple envs
        n_steps=2048,     # Longer rollout per environment
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_circuit_tensorboard/"
    )

    # === Checkpoint callback ===
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./models/',
        name_prefix='ppo_circuit'
    )

    # === Evaluation environment ===
    eval_env = CircuitEnv(work_dir="./ltspice_env_eval")  # Separate eval environment
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./eval_logs/',
        eval_freq=10_000,
        deterministic=True,
        render=False
    )

    # === Train model for longer timesteps ===
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback]
    )

    # === Save final model ===
    model.save("ppo_circuit_final")
    print("✅ Training completed. Model saved successfully!")
