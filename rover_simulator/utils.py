import math
import numpy as np
import matplotlib.patches as patches
from typing import List


def round_off(x, digit=0):
    p = 10 ** digit
    s = np.copysign(1, x)
    return (s * x * p * 2 + 1) // 2 / p * s


def isInList(idx: np.ndarray, idx_list: List) -> bool:
    if len(idx_list) == 0:
        return False
    elif np.any(np.all(idx == [chk_idx for chk_idx in idx_list], axis=1)):
        return True
    else:
        return False


def occupancyToColor(occupancy: float) -> str:
    return "#" + format(int(255 * (1 - occupancy)), '02x') * 3


def drawGrid(idx: np.ndarray, grid_width: float, color: str, alpha: float, ax, elems=None, fill=True) -> None:
    xy = idx * grid_width - grid_width / 2
    r = patches.Rectangle(
        xy=(xy),
        height=grid_width,
        width=grid_width,
        facecolor=color,
        alpha=alpha,
        fill=fill
    )
    if elems is not None:
        elems.append(ax.add_patch(r))
    else:
        ax.add_patch(r)


def state_transition(pose: np.ndarray, control_inputs: np.ndarray, time: float) -> np.ndarray:
    nu, omega = control_inputs
    t0 = pose[2]
    if math.fabs(omega) < 1e-10:
        new_pose = pose + np.array([nu * np.cos(t0), nu * np.sin(t0), omega]) * time
    else:
        new_pose = pose + np.array([nu / omega * (np.sin(t0 + omega * time) - np.sin(t0)), nu / omega * (-np.cos(t0 + omega * time) + np.cos(t0)), omega * time])
    while new_pose[2] > np.pi:
        new_pose[2] -= 2 * np.pi
    while new_pose[2] < -np.pi:
        new_pose[2] += 2 * np.pi
    return new_pose


def isInRange(angle: float, rangeMin: float, rangeMax: float):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    while rangeMin > np.pi:
        rangeMin -= 2 * np.pi
    while rangeMin < -np.pi:
        rangeMin += 2 * np.pi
    while rangeMax > np.pi:
        rangeMax -= 2 * np.pi
    while rangeMax < -np.pi:
        rangeMax += 2 * np.pi
    if rangeMin < rangeMax:
        if angle >= rangeMin and angle < rangeMax:
            return True
        else:
            return False
    else:
        if angle >= rangeMin:
            return True
        elif angle < rangeMax:
            return True
        else:
            return False


def lToP(l):
    return 1 - 1 / (1 + np.exp(l))


def pToL(p):
    return np.log(p / (1 - p))


def updateL(l, p):
    return l + np.log(p / (1 - p))


def updateP(p, p_):
    l = pToL(p)
    l = updateL(l, p_)
    return lToP(l)
