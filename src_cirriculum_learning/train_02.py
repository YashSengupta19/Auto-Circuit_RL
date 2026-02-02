import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gymnasium as gym
from customEnv_02 import CircuitEnv


def make_env(rank):
    def _init():
        return CircuitEnv(
            max_components=3,
            value_buckets=5,
            folder_path="../3_data",
            work_dir=f"./ltspice_env_{rank}"   # unique folder per worker
        )
    return _init


if __name__ == "__main__":   # <-- CRITICAL on Windows
    n_envs = 1  # adjust to CPU cores
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])

    policy_kwargs = dict(
        net_arch=[dict(pi=[256, 256], vf=[256, 256])]
    )

    model = PPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        batch_size=64,
        n_steps=2048,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./ppo_circuit_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path='./models/',
        name_prefix='ppo_circuit'
    )

    # ✅ Use a single env for evaluation to avoid multiprocessing nesting
    eval_env = make_env(999)()
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./best_model/',
        log_path='./eval_logs/',
        eval_freq=10000,
        deterministic=True,
        render=False
    )

    model.learn(
        total_timesteps=1_000_000,
        callback=[checkpoint_callback, eval_callback]
    )

    model.save("ppo_circuit_final")
    print("Training completed. Model saved!")
