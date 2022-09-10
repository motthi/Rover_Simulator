import math
import numpy as np


def state_transition(pose: np.ndarray, v:float, w:float, dt: float) -> np.ndarray:
    t0 = pose[2]
    if math.fabs(w) < 1e-10:
        new_pose = pose + np.array([v * np.cos(t0), v * np.sin(t0), w]) * dt
    else:
        new_pose = pose + np.array([v / w * (np.sin(t0 + w * dt) - np.sin(t0)), v / w * (-np.cos(t0 + w * dt) + np.cos(t0)), w * dt])
    while new_pose[2] > np.pi:
        new_pose[2] -= 2 * np.pi
    while new_pose[2] < -np.pi:
        new_pose[2] += 2 * np.pi
    return new_pose


def covariance_transition(
    pose: np.ndarray, cov: np.ndarray, stds: dict,
    v:float, w:float,
    dt: float
) -> np.ndarray:
    M = matM(v, w, dt, stds)
    A = matA(v, w, dt, pose[2])
    F = matF(v, w, dt, pose[2])
    return F.dot(cov).dot(F.T) + A.dot(M).dot(A.T)


def matM(v: float, w: float, time: float, stds: np.ndarray) -> np.ndarray:
    """Create M

    Args:
        v (float): Control inputs
        w (float): Control inputs (rotate)
        time (float): delta t
        stds (np.ndarray): Motion noise 0: nn, 1: no, 2: on, 3: oo

    Returns:
        np.ndarray: _description_
    """
    return np.diag([
        stds[0]**2 * abs(v) / time + stds[1]**2 * abs(w) / time,
        stds[2]**2 * abs(v) / time + stds[3]**2 * abs(w) / time
    ])


def matA(v: float, w: float, time: float, theta: float) -> np.ndarray:
    st, ct = math.sin(theta), math.cos(theta)
    stw, ctw = math.sin(theta + w * time), math.cos(theta + w * time)
    return np.array([
        [(stw - st) / w, -v / (w**2) * (stw - st) + v / w * time * ctw],
        [(-ctw + ct) / w, -v / (w**2) * (-ctw + ct) + v / w * time * stw],
        [0, time]
    ])


def matF(v: float, w: float, time: float, theta: float) -> np.ndarray:
    F = np.diag([1.0, 1.0, 1.0])
    F[0, 2] = v / w * (math.cos(theta + w * time) - math.cos(theta))
    F[1, 2] = v / w * (math.sin(theta + w * time) - math.sin(theta))
    return F
