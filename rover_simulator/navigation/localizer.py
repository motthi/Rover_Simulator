import numpy as np
from scipy.stats import norm
from rover_simulator.utils import state_transition
from rover_simulator.core import Localizer


class ImaginalLocalizer(Localizer):
    def __init__(self) -> None:
        super().__init__()

    def estimate_pose(self, previous_pose: np.ndarray, control_inputs: np.ndarray, time_interval: float):
        return state_transition(previous_pose, control_inputs, time_interval)


class NoisyLocalizer(ImaginalLocalizer):
    def __init__(self, noise: np.ndarray = np.array([0.2, 0.05]), bias: np.ndarray = np.array([1.1, 1.01])) -> None:
        super().__init__()
        if not noise.shape == (2, ):
            raise ValueError("array 'noise' is not of the right shape (2,), given array's shape is {}".format(noise.shape))
        if not bias.shape == (2, ):
            raise ValueError("array 'noise' is not of the right shape (2,), given array's shape is {}".format(bias.shape))
        self.noise_pose_pdf = norm(scale=noise[0])
        self.noise_theta_pdf = norm(scale=noise[1])
        self.bias = bias

    def estimate_pose(self, previous_pose: np.ndarray, control_inputs: np.ndarray, time_interval: float):
        control_inputs = self.add_noise(control_inputs)
        control_inputs = self.add_bias(control_inputs)
        return state_transition(previous_pose, control_inputs, time_interval)

    def add_noise(self, control_inputs: np.ndarray):
        nu, omega = control_inputs
        nu += self.noise_pose_pdf.rvs()
        omega += self.noise_theta_pdf.rvs()
        return np.array([nu, omega])

    def add_bias(self, control_inputs: np.ndarray):
        nu, omega = control_inputs
        return np.array([nu * self.bias[0], omega * self.bias[1]])
