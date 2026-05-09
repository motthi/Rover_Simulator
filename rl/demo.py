import os
# Force headless backend for Qt / matplotlib to avoid GUI plugin errors
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import matplotlib
matplotlib.use('Agg')
import numpy as np
from rover_simulator.navigation.rl_env import RoverGymEnv
from stable_baselines3 import PPO
import csv
import matplotlib.pyplot as plt


def run_demo(model_path: str = "models/ppo_rover.zip", episodes: int = 5, out_dir: str = "rl/demos"):
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # load model once
    model = PPO.load(model_path)
    print("Loaded model:", model)

    for ep in range(episodes):
        env = RoverGymEnv()
        res = env.reset()
        if isinstance(res, tuple):
            obs, _ = res
        else:
            obs = res

        actions_log = []
        poses_log = []
        rewards_log = []

        done = False
        step_idx = 0
        # manual loop to log actions and rewards
        while not done and step_idx < env.max_steps:
            action, _ = model.predict(obs, deterministic=True)
            step_res = env.step(action)
            # handle both Gym and Gymnasium step return formats
            if len(step_res) == 4:
                obs, r, done, info = step_res
            else:
                obs, r, terminated, truncated, info = step_res
                done = bool(terminated or truncated)

            actions_log.append(action.tolist())
            rewards_log.append(float(r))
            # record rover real pose if available
            if env.rover is not None:
                poses_log.append(env.rover.real_pose.tolist())
            step_idx += 1

        # determine result string
        result = info.get('success', False)
        result_str = 'Succeed' if result else (info.get('collision', False) and 'Collided' or ('Truncated' if step_idx >= env.max_steps else 'Unknown'))

        # save CSV log
        csv_path = os.path.join(out_dir, f"run_{ep}_{result_str}.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'action_v', 'action_w', 'reward', 'pose_x', 'pose_y', 'pose_th'])
            for i in range(len(actions_log)):
                a = actions_log[i]
                r = rewards_log[i]
                p = poses_log[i] if i < len(poses_log) else [None, None, None]
                writer.writerow([i, a[0], a[1], r, p[0], p[1], p[2]])

        # save a static plot of the path
        history = env.rover.history
        fig, ax = plt.subplots(figsize=(6, 6))
        try:
            from rover_simulator.utils.draw import draw_obstacles
            draw_obstacles(ax, env.world.obstacles, 0.0)
        except Exception:
            pass

        if len(history.real_poses) > 0:
            poses = np.array(history.real_poses)
            ax.plot(poses[:, 0], poses[:, 1], '-o', markersize=2)

        start = poses[0] if len(history.real_poses) > 0 else np.array([0, 0, 0])
        goal = env._goal
        ax.scatter([start[0]], [start[1]], c='green', label='start')
        ax.scatter([goal[0]], [goal[1]], c='red', label='goal')
        ax.set_aspect('equal')
        ax.set_title(f"Episode {ep} - {result_str}")
        ax.legend()

        out_path = os.path.join(out_dir, f"run_{ep}_{result_str}.png")
        fig.tight_layout()
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved demo image: {out_path}, csv: {csv_path}")


if __name__ == '__main__':
    run_demo(episodes=5)
