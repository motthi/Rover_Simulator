import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from rover_simulator.navigation.rl_env import RoverGymEnv
from rl.callbacks import EpisodeInfoCallback, RolloutCheckpointCallback
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList


def make_env(map_file=None):
    def _init():
        return RoverGymEnv(map_file=map_file)
    return _init


def main(timesteps: int = 200000, model_dir: str = "models"):
    env = DummyVecEnv([make_env()])
    env = VecMonitor(env)   # Wrap with VecMonitor to record episode rewards/lengths for TensorBoard

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./rl_logs")
    
    # attach callbacks: episode info logging, periodic checkpoint, optional rollout checkpoint
    callbacks = []
    # episode info -> tensorboard
    # periodic checkpoint every N timesteps (default 50k)
    # create checkpoint callback saving into model_dir
    checkpoint_cb = CheckpointCallback(save_freq=50000, save_path=model_dir, name_prefix="ppo_rover")
    # rollout checkpoint (save after every rollout)
    rollout_cb = RolloutCheckpointCallback(save_dir=model_dir, name_prefix="ppo_rover")

    callbacks.append(EpisodeInfoCallback())
    callbacks.append(checkpoint_cb)
    callbacks.append(rollout_cb)
    cb_list = CallbackList(callbacks)

    model.learn(total_timesteps=timesteps, callback=cb_list)
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_rover")
    model.save(model_path)
    print(f"Saved model to {model_path}.zip")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--timesteps', type=int, default=200000, help='Total timesteps to train')
    parser.add_argument('--model-dir', type=str, default='models', help='Directory to save model')
    args = parser.parse_args()
    
    main(timesteps=args.timesteps, model_dir=args.model_dir)
