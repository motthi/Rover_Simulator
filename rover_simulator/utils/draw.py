import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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


def set_fig_params(figsize, xlim, ylim):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel("X [m]", fontsize=10)
    ax.set_ylabel("Y [m]", fontsize=10)
    return fig, ax


def draw_rover(ax, pose, r, color=None):
    x, y, theta = pose
    xn = x + r * np.cos(theta)
    yn = y + r * np.sin(theta)
    if color is None:
        c = color
    ax.plot([x, xn], [y, yn], color=c)
    c = patches.Circle(xy=(x, y), radius=r, fill=False, color=c)
    ax.add_patch(c)


def draw_obstacles(ax, obstacles, enlarge_range, alpha=1.0):
    for obstacle in obstacles:
        obstacle.draw(ax, enlarge_range, alpha)


def draw_start(ax, start_pos: np.ndarray) -> None:
    ax.plot(start_pos[0], start_pos[1], "or")


def draw_goal(ax, goal_pos: np.ndarray) -> None:
    ax.plot(goal_pos[0], goal_pos[1], "xr")


def draw_pose(ax, pose, color) -> None:
    ax.plot(
        [e[0] for e in pose],
        [e[1] for e in pose],
        linewidth=1.0,
        linestyle="-",
        color=color
    )


def draw_poses(ax, poses, color, linestyle="-") -> None:
    ax.plot(
        [e[0] for e in poses],
        [e[1] for e in poses],
        linewidth=1.0,
        linestyle=linestyle,
        color=color
    )


def draw_error_ellipses(ax, poses, covs, color="blue", plot_interval=10):
    for i, (p, cov) in enumerate(zip(poses, covs)):
        if i % plot_interval == 0:
            e = sigma_ellipse(p[0:2], cov[0:2, 0:2], 3, color=color)
            ax.add_patch(e)
            x, y, c = p
            sigma3 = math.sqrt(cov[2, 2]) * 3
            xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
            ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
            ax.plot(xs, ys, color=color, alpha=0.5)


def draw_waypoints(ax, waypoints: list, color) -> None:
    ax.plot(
        [e[0] for e in waypoints],
        [e[1] for e in waypoints],
        linewidth=1.0,
        linestyle="-",
        color=color
    )


def draw_grid(idx: np.ndarray, grid_width: float, color: str, alpha: float, ax, elems=None, fill=True) -> None:
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


def draw_grid_map_contour(ax, grid_map, grid_num, grid_width, levels):
    x = np.linspace(0, grid_num[0], grid_num[0]) * grid_width
    y = np.linspace(0, grid_num[1], grid_num[1]) * grid_width
    X, Y = np.meshgrid(x, y)
    cntr = ax.contour(x, y, grid_map.T, levels=levels, cmap="brg")
    ax.clabel(cntr)


def draw_grid_map(ax, grid_map, cmap, vmin, vmax, alpha, extent, zorder, colorbar=True):
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


def sigma_ellipse(p, cov, n, color="blue"):
    eig_vals, eig_vec = np.linalg.eig(cov)
    ang = math.atan2(eig_vec[:, 0][1], eig_vec[:, 0][0]) / math.pi * 180
    return Ellipse(p, width=2 * n * math.sqrt(eig_vals[0]), height=2 * n * math.sqrt(eig_vals[1]), angle=ang, fill=False, color=color, alpha=0.5, zorder=5)


def draw_sensing_results(ax, poses, sensor_range, sensor_fov, sensing_results, draw_sensing_points_flag, draw_sensing_area_flag):
    for i, sensing_result in enumerate(sensing_results):
        if sensing_result is not None:
            if draw_sensing_points_flag:
                ax.plot(poses[i][0], poses[i][1], marker="o", c="red", ms=5)
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
