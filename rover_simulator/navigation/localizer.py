import math
import numpy as np
from scipy.stats import norm
from rover_simulator.utils import state_transition
from rover_simulator.core import Localizer
from scipy.stats import multivariate_normal


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


def matM(nu, omega, time, stds) -> np.ndarray:
    return np.diag([stds["nn"]**2 * abs(nu) / time + stds["no"]**2 * abs(omega) / time,
                    stds["on"]**2 * abs(nu) / time + stds["oo"]**2 * abs(omega) / time])


def matA(nu, omega, time, theta) -> np.ndarray:
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega * time), math.cos(theta + omega * time)
    return np.array([
        [(stw - st) / omega, -nu / (omega**2) * (stw - st) + nu / omega * time * ctw],
        [(-ctw + ct) / omega, -nu / (omega**2) * (-ctw + ct) + nu / omega * time * stw],
        [0, time]
    ])


def matF(nu, omega, time, theta) -> np.ndarray:
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu / omega * (math.cos(theta + omega * time) - math.cos(theta))
    F[1, 2] = nu / omega * (math.sin(theta + omega * time) - math.sin(theta))
    return F


def matH(pose, landmark_pos):  # kf4funcs
    mx, my = landmark_pos
    mux, muy, mut = pose
    q = (mux - mx)**2 + (muy - my)**2
    return np.array([[(mux - mx) / np.sqrt(q), (muy - my) / np.sqrt(q), 0.0], [(my - muy) / q, (mux - mx) / q, -1.0]])


def matQ(distance_dev, direction_dev):
    return np.diag(np.array([distance_dev**2, direction_dev**2]))


class KalmanFilter:
    def __init__(
        self,
        init_pose, init_cov=None, motion_noise_stds={"nn": 0.1, "no": 0.0001, "on": 0.013, "oo": 0.02}
    ):
        if init_cov is None:
            init_cov = np.diag([1e-10, 1e-10, 1e-10])
        self.belief = multivariate_normal(mean=init_pose, cov=init_cov)
        self.pose = self.belief.mean
        self.motion_noise_stds = motion_noise_stds

    def estimate_pose(self, previous_pose: np.ndarray, control_inputs: np.ndarray, dt: float):
        nu, omega = control_inputs
        if abs(omega) < 1e-5:
            omega = 1e-5  # 値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる

        M = matM(nu, omega, dt, self.motion_noise_stds)
        A = matA(nu, omega, dt, self.belief.mean[2])
        F = matF(nu, omega, dt, self.belief.mean[2])
        self.belief.cov = F.dot(self.belief.cov).dot(F.T) + A.dot(M).dot(A.T)
        self.belief.mean = state_transition(previous_pose, control_inputs, dt)
        self.pose = self.belief.mean
        return self.pose
