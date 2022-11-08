import math
import numpy as np
from scipy.stats import norm
from rover_simulator.utils.cmotion.cmotion import state_transition, covariance_transition
from rover_simulator.core import Localizer
from scipy.stats import multivariate_normal


class ImaginalLocalizer(Localizer):
    def __init__(self) -> None:
        super().__init__()

    def estimate_pose(self, previous_pose: np.ndarray, v: np.ndarray, w: np.ndarray, time_interval: float):
        return np.array(state_transition(previous_pose, v, w, time_interval))


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

    def estimate_pose(self, previous_pose: np.ndarray, v: np.ndarray, w: np.ndarray, time_interval: float):
        v, w = self.add_noise(v, w)
        v, w = self.add_bias(v, w)
        return state_transition(previous_pose, v, w, time_interval)

    def add_noise(self, v: float, w: float):
        v += self.noise_pose_pdf.rvs()
        w += self.noise_theta_pdf.rvs()
        return np.array([v, w])

    def add_bias(self, v: float, w: float):
        return np.array([v * self.bias[0], w * self.bias[1]])


def matM(v: float, w: float, dt, stds) -> np.ndarray:
    return np.diag([
        stds[0]**2 * abs(v) / dt + stds[1]**2 * abs(w) / dt,
        stds[2]**2 * abs(v) / dt + stds[3]**2 * abs(w) / dt
    ])


def matA(v, w, dt, theta) -> np.ndarray:
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + w * dt), math.cos(theta + w * dt)
    return np.array([
        [(stw - st) / w, -v / (w**2) * (stw - st) + v / w * dt * ctw],
        [(-ctw + ct) / w, -v / (w**2) * (-ctw + ct) + v / w * dt * stw],
        [0, dt]
    ])


def matF(v, w, dt, theta) -> np.ndarray:
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = v / w * (math.cos(theta + w * dt) - math.cos(theta))
    F[1, 2] = v / w * (math.sin(theta + w * dt) - math.sin(theta))
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
        self.motion_noise_stds = np.array([motion_noise_stds["nn"], motion_noise_stds["no"], motion_noise_stds["on"], motion_noise_stds["oo"]])

    def estimate_pose(self, previous_pose: np.ndarray, v: float, w: float, dt: float):
        if abs(w) < 1e-5:
            w = 1e-5  # 値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる
        self.belief.cov = covariance_transition(previous_pose, self.belief.cov, self.motion_noise_stds, v, w, dt)
        self.belief.mean = state_transition(previous_pose, v, w, dt)
        self.pose = self.belief.mean
        return self.pose
