import math
import numpy as np
import matplotlib.patches as patches
from typing import List
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse


environment_cmap = LinearSegmentedColormap(
    "environment_map",
    {
        'red': [
            (0.0, 0.0, 0.0),
            (0.5, 1.0, 1.0),
            (1.0, 0.0, 0.0)
        ],
        'green': [
            (0.0, 0.0, 0.0),
            (0.5, 1.0, 1.0),
            (1.0, 0.0, 0.0)
        ],
        'blue': [
            (0.0, 0.0, 1.0),
            (0.5, 1.0, 1.0),
            (1.0, 0.0, 0.0)
        ]
    }  # 0.0 -> -1.0 : Unknow, 0.5 -> 0.0 : Free, 1.0 -> 1.0 : Occupied
)


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


def isInRange(angle: float, rangeMin: float, rangeMax: float):
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


def angle_to_range(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


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


def sigma_ellipse(p, cov, n):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0]) / math.pi * 180
    return Ellipse(p, width=2 * n * math.sqrt(eig_vals[0]), height=2 * n * math.sqrt(eig_vals[1]), angle=ang, fill=False, color="blue", alpha=0.5)
