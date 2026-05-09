import os
import numpy as np
from stable_baselines3 import PPO
from rover_simulator.navigation.rl_env import RoverGymEnv


def evaluate(model_path: str, episodes: int = 5):
    model = PPO.load(model_path)
    env = RoverGymEnv()
    results = []
    for ep in range(episodes):
        res = env.reset()
        if isinstance(res, tuple):
            obs = res[0]
        else:
            obs = res
        done = False
        total_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            step_res = env.step(action)
            # support both Gym (obs, r, done, info) and Gymnasium (obs, r, terminated, truncated, info)
            if len(step_res) == 4:
                obs, r, done, info = step_res
            else:
                obs, r, terminated, truncated, info = step_res
                done = bool(terminated or truncated)
            total_reward += r
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, info = env.step(action)
            total_reward += r
        results.append(total_reward)
        print(f"Episode {ep} reward: {total_reward}, info: {info}")
    print(f"Mean reward: {np.mean(results)}")


if __name__ == '__main__':
    if os.path.exists("models/ppo_rover.zip"):
        evaluate("models/ppo_rover.zip", episodes=3)
    else:
        print("models/ppo_rover.zip not found. Train first: python rl/train.py")
