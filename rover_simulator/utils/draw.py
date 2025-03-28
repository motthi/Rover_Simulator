import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from rover_simulator.core import Obstacle, Sensor


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


def set_fig_params(figsize: tuple = (8, 8), xlim: list = None, ylim: list = None, axes_setting: list = None):
    fig = plt.figure(figsize=figsize)
    plt.subplots_adjust(left=0.1, right=0.95, bottom=0.1, top=0.95)
    if axes_setting:
        ax = fig.add_axes(axes_setting)
    else:
        ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1]) if xlim is not None else None
    ax.set_ylim(ylim[0], ylim[1]) if ylim is not None else None
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    return fig, ax


def save_fig(src: str, fig: Figure, dpi: int = 300):
    fig.savefig(src, dpi=dpi, bbox_inches="tight", pad_inches=0.05)


def save_ani(src: str, ani: FuncAnimation, dpi: int = 300):
    ani.save(src, dpi=dpi)


def draw_rover(ax: Axes, pose: np.ndarray, r: float, color: str = None):
    x, y, theta = pose
    xn = x + r * np.cos(theta)
    yn = y + r * np.sin(theta)
    if color is None:
        color = color
    ax.plot([x, xn], [y, yn], color=color)
    cir = patches.Circle(xy=(x, y), radius=r, fill=False, color=color)
    ax.add_patch(cir)


def draw_obstacles(ax: Axes, obstacles: list[Obstacle], expand_dist: float = 0.0, alpha: float = 1.0, color_enlarge: str = 'gray') -> None:
    for obstacle in obstacles:
        obstacle.draw_expanded(ax, expand_dist, alpha, color_enlarge)
    for obstacle in obstacles:
        obstacle.draw(ax, alpha)


def draw_start(ax: Axes, start_pos: np.ndarray) -> None:
    ax.plot(start_pos[0], start_pos[1], "or", label="Start")


def draw_goal(ax: Axes, goal_pos: np.ndarray) -> None:
    ax.plot(goal_pos[0], goal_pos[1], "xr", label="Goal")


def draw_pose(ax: Axes, pose: np.ndarray, color: str = "black", alpha: float = 1.0) -> None:
    ax.plot(
        [e[0] for e in pose],
        [e[1] for e in pose],
        linewidth=1.0,
        linestyle="-",
        color=color,
        alpha=alpha
    )


def draw_poses(ax: Axes, poses: np.ndarray, color: str = "black", linestyle: str = "-", alpha: float = 1.0) -> None:
    ax.plot(
        [e[0] for e in poses],
        [e[1] for e in poses],
        linewidth=1.0,
        linestyle=linestyle,
        color=color,
        alpha=alpha
    )


def draw_error_ellipses(ax: Axes, poses: np.ndarray, covs: np.ndarray, color: str = "blue", plot_interval: int = 10):
    for i, (p, cov) in enumerate(zip(poses, covs)):
        if i % plot_interval == 0:
            e = sigma_ellipse(p[0:2], cov[0:2, 0:2], 3, color=color)
            ax.add_patch(e)
            x, y, c = p
            sigma3 = math.sqrt(cov[2, 2]) * 3
            xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
            ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
            ax.plot(xs, ys, color=color, alpha=0.5)


def draw_waypoints(ax: Axes, waypoints: np.ndarray, color: str) -> None:
    ax.plot(waypoints[:, 0], waypoints[:, 1], linewidth=1.0, linestyle="-", color=color)


def draw_grid(idx: np.ndarray, grid_width: float, color: str, alpha: float, ax: Axes, elems=None, fill=True) -> None:
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


def draw_grid_map_contour(ax: Axes, grid_map: np.ndarray, grid_num: list, grid_width: float, levels):
    x = np.linspace(0, grid_num[0], grid_num[0]) * grid_width
    y = np.linspace(0, grid_num[1], grid_num[1]) * grid_width
    X, Y = np.meshgrid(x, y)
    cntr = ax.contour(x, y, grid_map.T, levels=levels, cmap="brg")
    ax.clabel(cntr)


def draw_grid_map(ax: Axes, grid_map, cmap=None, vmin=None, vmax=None, alpha=None, extent=None, zorder=None, colorbar=True):
    grid_map = cv2.rotate(grid_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
    im = ax.imshow(
        grid_map,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        alpha=alpha,
        extent=extent,
        zorder=zorder
    )
    plt.colorbar(im) if colorbar is True else None


def occupancyToColor(occupancy: float) -> str:
    return "#" + format(int(255 * (1 - occupancy)), '02x') * 3


def sigma_ellipse(p: np.ndarray, cov: np.ndarray, n, color="blue"):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0]) / math.pi * 180
    return Ellipse(p, width=2 * n * math.sqrt(eig_vals[0]), height=2 * n * math.sqrt(eig_vals[1]), angle=ang, fill=False, color=color, alpha=0.5, zorder=5)


def draw_sensing_results(ax: Axes, poses, sensor_range, sensor_fov, sensing_results, draw_sensing_points_flag, draw_sensing_area_flag):
    for i, sensing_result in enumerate(sensing_results):
        if sensing_result is None:
            continue
        ax.plot(poses[i][0], poses[i][1], marker="o", c="red", ms=5) if draw_sensing_points_flag else None
        if draw_sensing_area_flag:
            x, y, theta = poses[i]
            ax.plot(x, y, marker="o", c="red", ms=5)
            if draw_sensing_area_flag:
                sensing_range = patches.Wedge(
                    (x, y), sensor_range,
                    theta1=np.rad2deg(theta - sensor_fov / 2),
                    theta2=np.rad2deg(theta + sensor_fov / 2),
                    alpha=0.5,
                    color="mistyrose"
                )
                ax.add_patch(sensing_range)


def draw_history_pose(ax: Axes, elems: list, poses: list, rover_r: float, rover_color: str, step: int, start_step: int = 0) -> None:
    xn, yn = poses[start_step + step][0:2] + rover_r * np.array([np.cos(poses[start_step + step][2]), np.sin(poses[start_step + step][2])])
    elems += ax.plot([poses[start_step + step][0], xn], [poses[start_step + step][1], yn], color=rover_color)
    elems += ax.plot(
        [e[0] for e in poses[start_step:start_step + step + 1]],
        [e[1] for e in poses[start_step:start_step + step + 1]],
        linewidth=1.0,
        color=rover_color
    )
    c = patches.Circle(xy=(poses[start_step + step][0], poses[start_step + step][1]), radius=rover_r, fill=False, color=rover_color)
    elems.append(ax.add_patch(c))


def draw_history_pose_with_error_ellipse(ax: Axes, elems: list, poses: list, covs: list, rover_r: float, rover_color: str, error_color: str, step: int, start_step: int) -> None:
    draw_history_pose(ax, elems, poses, rover_r, rover_color, step, start_step)
    e = sigma_ellipse(poses[step][0:2], covs[step][0:2, 0:2], 3, color=error_color)
    elems.append(ax.add_patch(e))
    x, y, c = poses[step]
    sigma3 = math.sqrt(covs[step][2, 2]) * 3
    xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
    ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
    elems += ax.plot(xs, ys, color=error_color, alpha=0.5)


def draw_history_waypoints(ax: Axes, elems: list, waypoints_list: list, step: int) -> None:
    waypoints = waypoints_list[step]
    elems += ax.plot(
        [e[0] for e in waypoints],
        [e[1] for e in waypoints],
        linewidth=1.0,
        linestyle=":",
        color="blue",
        alpha=0.5
    )
