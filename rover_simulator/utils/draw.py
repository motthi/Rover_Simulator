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


def set_fig_params(figsize:tuple, xlim:list=None, ylim:list=None, axes_setting:list=None):
    fig = plt.figure(figsize=figsize)
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


def save_fig(src, fig, dpi=300):
    fig.savefig(src, dpi=dpi, bbox_inches="tight", pad_inches=0.05)


def save_ani(src, ani, dpi=300):
    ani.save(src, dpi=dpi)


def draw_rover(ax, pose, r, color=None):
    x, y, theta = pose
    xn = x + r * np.cos(theta)
    yn = y + r * np.sin(theta)
    if color is None:
        color = color
    ax.plot([x, xn], [y, yn], color=color)
    cir = patches.Circle(xy=(x, y), radius=r, fill=False, color=color)
    ax.add_patch(cir)


def draw_obstacles(ax, obstacles, enlarge_range, alpha=1.0):
    for obstacle in obstacles:
        obstacle.draw(ax, enlarge_range, alpha)


def draw_start(ax, start_pos: np.ndarray) -> None:
    ax.plot(start_pos[0], start_pos[1], "or")


def draw_goal(ax, goal_pos: np.ndarray) -> None:
    ax.plot(goal_pos[0], goal_pos[1], "xr")


def draw_pose(ax, pose, color, alpha=1.0) -> None:
    ax.plot(
        [e[0] for e in pose],
        [e[1] for e in pose],
        linewidth=1.0,
        linestyle="-",
        color=color,
        alpha=alpha
    )


def draw_poses(ax, poses, color, linestyle="-", alpha=1.0) -> None:
    ax.plot(
        [e[0] for e in poses],
        [e[1] for e in poses],
        linewidth=1.0,
        linestyle=linestyle,
        color=color,
        alpha=alpha
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


def draw_history_pose(ax, elems: list, poses: list, rover_r: float, rover_color: str, step: int, start_step: int) -> None:
    xn, yn = poses[step][0:2] + rover_r * np.array([np.cos(poses[step][2]), np.sin(poses[step][2])])
    elems += ax.plot([poses[step][0], xn], [poses[step][1], yn], color=rover_color)
    elems += ax.plot(
        [e[0] for e in poses[start_step:start_step + step + 1]],
        [e[1] for e in poses[start_step:start_step + step + 1]],
        linewidth=1.0,
        color=rover_color
    )
    c = patches.Circle(xy=(poses[step][0], poses[step][1]), radius=rover_r, fill=False, color=rover_color)
    elems.append(ax.add_patch(c))


def draw_history_pose_with_error_ellipse(ax, elems: list, poses: list, covs: list, rover_r: float, rover_color: str, error_color: str, step: int, start_step: int) -> None:
    draw_history_pose(ax, elems, poses, rover_r, rover_color, step, start_step)
    e = sigma_ellipse(poses[step][0:2], covs[step][0:2, 0:2], 3, color=error_color)
    elems.append(ax.add_patch(e))
    x, y, c = poses[step]
    sigma3 = math.sqrt(covs[step][2, 2]) * 3
    xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
    ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
    elems += ax.plot(xs, ys, color=error_color, alpha=0.5)


def draw_history_sensing_results(
    ax, elems: list,
    real_pose: np.ndarray, estimated_pose: np.ndarray,
    sensed_obstacles: list, rover_r: float, sensor_range: float, sensor_fov: float,
    draw_sensing_points_flag: bool, draw_sensing_area_flag: bool
) -> None:
    ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5) if draw_sensing_points_flag is True else None
    if draw_sensing_area_flag:
        sensing_range = patches.Wedge(
            (real_pose[0], real_pose[1]), sensor_range,
            theta1=np.rad2deg(real_pose[2] - sensor_fov / 2),
            theta2=np.rad2deg(real_pose[2] + sensor_fov / 2),
            alpha=0.5,
            color="mistyrose"
        )
        elems.append(ax.add_patch(sensing_range))

        for sensed_obstacle in sensed_obstacles:
            distance = sensed_obstacle['distance']
            angle = sensed_obstacle['angle'] + estimated_pose[2]
            radius = sensed_obstacle['radius']

            # ロボットと障害物を結ぶ線を描写
            xn, yn = np.array(estimated_pose[0:2]) + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            elems += ax.plot([estimated_pose[0], xn], [estimated_pose[1], yn], color="mistyrose", linewidth=0.8)

            # Draw Enlarged Obstacle Regions
            enl_obs = patches.Circle(xy=(xn, yn), radius=radius + rover_r, fc='blue', ec='blue', alpha=0.3)
            elems.append(ax.add_patch(enl_obs))


def draw_history_waypoints(ax, elems: list, waypoints_list: list, step: int) -> None:
    waypoints = waypoints_list[step]
    elems += ax.plot(
        [e[0] for e in waypoints],
        [e[1] for e in waypoints],
        linewidth=1.0,
        linestyle=":",
        color="blue",
        alpha=0.5
    )
