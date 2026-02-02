import torch
import numpy as np
from stable_baselines3 import PPO
import gymnasium as gym
from customEnv_03 import CircuitEnv


def evaluate_model(model, env, num_episodes=1, render=False):
    """
    Run evaluation for a trained PPO model.
    """
    all_rewards = []
    all_lengths = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1

            if render:
                env.render()

        all_rewards.append(total_reward)
        all_lengths.append(steps)
        print(f"Episode {ep + 1}/{num_episodes} - Reward: {total_reward:.3f}, Steps: {steps}")

    print("\n✅ Evaluation complete!")
    print(f"Average Reward: {np.mean(all_rewards):.3f}")
    print(f"Average Episode Length: {np.mean(all_lengths):.2f}")
    return all_rewards, all_lengths


if __name__ == "__main__":
    # 🔹 Load test environment
    test_env = CircuitEnv(
        max_components=5,
        value_buckets=5,
        folder_path="../test_data",
        work_dir="./ltspice_test_env"
    )

    # 🔹 Load trained PPO model
    model_path = r"C:\Users\jishu\OneDrive\Desktop\SA_PE\src_cirriculum_learning\models\ppo_circuit_1000000_steps.zip"   # or "./ppo_circuit_final.zip"

    model = PPO.load(model_path)

    # 🔹 Evaluate the model
    rewards, lengths = evaluate_model(model, test_env)

    # 🔹 Optionally save results
    np.savez("test_evaluation_results.npz", rewards=rewards, lengths=lengths)
    print("Saved evaluation results to test_evaluation_results.npz")
