import math
import numpy as np


def state_transition(pose: np.ndarray, control_inputs: np.ndarray, dt: float) -> np.ndarray:
    nu, omega = control_inputs
    t0 = pose[2]
    if math.fabs(omega) < 1e-10:
        new_pose = pose + np.array([nu * np.cos(t0), nu * np.sin(t0), omega]) * dt
    else:
        new_pose = pose + np.array([nu / omega * (np.sin(t0 + omega * dt) - np.sin(t0)), nu / omega * (-np.cos(t0 + omega * dt) + np.cos(t0)), omega * dt])
    while new_pose[2] > np.pi:
        new_pose[2] -= 2 * np.pi
    while new_pose[2] < -np.pi:
        new_pose[2] += 2 * np.pi
    return new_pose


def covariance_transition(
    pose: np.ndarray, cov: np.ndarray, stds: tuple,
    control_inputs: np.ndarray,
    dt: float
) -> np.ndarray:
    nu, omega = control_inputs
    M = matM(nu, omega, dt, stds)
    A = matA(nu, omega, dt, pose[2])
    F = matF(nu, omega, dt, pose[2])
    return F.dot(cov).dot(F.T) + A.dot(M).dot(A.T)


def matM(nu: float, omega: float, time: float, stds: tuple) -> np.ndarray:
    return np.diag([
        stds["nn"]**2 * abs(nu) / time + stds["no"]**2 * abs(omega) / time,
        stds["on"]**2 * abs(nu) / time + stds["oo"]**2 * abs(omega) / time
    ])


def matA(nu: float, omega: float, time: float, theta: float) -> np.ndarray:
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + omega * time), math.cos(theta + omega * time)
    return np.array([
        [(stw - st) / omega, -nu / (omega**2) * (stw - st) + nu / omega * time * ctw],
        [(-ctw + ct) / omega, -nu / (omega**2) * (-ctw + ct) + nu / omega * time * stw],
        [0, time]
    ])


def matF(nu: float, omega: float, time: float, theta: float) -> np.ndarray:
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = nu / omega * (math.cos(theta + omega * time) - math.cos(theta))
    F[1, 2] = nu / omega * (math.sin(theta + omega * time) - math.sin(theta))
    return F
