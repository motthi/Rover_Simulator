from stable_baselines3 import PPO
from rover_simulator.core import Controller


class RLController(Controller):
    """Controller wrapper that uses a trained SB3 policy to produce (v, w).

    Usage:
      controller = RLController(model_path="path/to/model.zip", env=env)
      rover.controller = controller
      controller.attach_env(env)
    """

    def __init__(self, env=None):
        super().__init__()
        self.model = None
        self.env = env

    def attach_env(self, env):
        self.env = env

    def calculate_control_inputs(self):
        if self.model is None or self.env is None:
            # fallback
            return 0.0, 0.0

        obs = self.env._get_obs()
        action, _ = self.model.predict(obs, deterministic=True)
        v = float(action[0])
        w = float(action[1])
        return v, w

    def set_model(self, model):
        self.model = model
