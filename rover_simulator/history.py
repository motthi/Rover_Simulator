import sys
import math
import copy
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from rover_simulator.core import Obstacle
from rover_simulator.utils import sigma_ellipse

if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class History():
    def __init__(
        self,
        time_interval: float = 0.1,
        rover_r: float = 0.5,
        sensor_range: float = 10.0,
        sensor_fov: float = np.pi / 2,
        rover_color: str = 'black',
        waypoint_color: str = 'blue'
    ) -> None:
        self.steps = []
        self.real_poses = []
        self.estimated_poses = []
        self.sensing_results = []
        self.waypoints = []
        self.sensor_range = sensor_range
        self.sensor_fov = sensor_fov
        self.time_interval = time_interval
        self.rover_r = rover_r
        self.rover_color = rover_color
        self.waypoint_color = waypoint_color

    def append(self, *args, **kwargs) -> None:
        self.steps.append(len(self.steps))
        self.real_poses.append(kwargs['real_pose'])
        self.estimated_poses.append(kwargs['estimated_pose'])
        self.sensing_results.append(kwargs['sensing_result'])
        if 'waypoints' in kwargs:
            self.waypoints.append(copy.copy(kwargs['waypoints']))

    def plot(
        self,
        xlim: List[float], ylim: List[float],
        figsize: Tuple[float, float] = (8, 8),
        obstacles: List[Obstacle] = [],
        enlarge_obstacle: float = 0.0,
        draw_waypoints: bool = False,
        draw_sensing_points: bool = True,
        draw_sensing_area: bool = True
    ):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        # Draw Enlarged Obstacle Regions
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        # Draw Obstacles
        for obstacle in obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
            ax.add_patch(obs)

        # Draw Sensing Results
        for i, sensing_result in enumerate(self.sensing_results):
            if sensing_result is not None:
                if draw_sensing_points:
                    ax.plot(self.real_poses[i][0], self.real_poses[i][1], marker="o", c="red", ms=5)
                if draw_sensing_area:
                    x, y, theta = self.real_poses[i]
                    ax.plot(x, y, marker="o", c="red", ms=5)
                    if draw_sensing_area:
                        sensing_range = patches.Wedge(
                            (x, y), self.sensor_range,
                            theta1=np.rad2deg(theta - self.sensor_fov / 2),
                            theta2=np.rad2deg(theta + self.sensor_fov / 2),
                            alpha=0.5,
                            color="mistyrose"
                        )
                        ax.add_patch(sensing_range)

        # Draw Last Rover Position
        x, y, theta = self.real_poses[-1]
        xn = x + self.rover_r * np.cos(theta)
        yn = y + self.rover_r * np.sin(theta)
        ax.plot([x, xn], [y, yn], color=self.rover_color)
        c = patches.Circle(xy=(x, y), radius=self.rover_r, fill=False, color=self.rover_color)
        ax.add_patch(c)

        x, y, theta = self.estimated_poses[-1]
        xn = x + self.rover_r * np.cos(theta)
        yn = y + self.rover_r * np.sin(theta)
        ax.plot([x, xn], [y, yn], color=self.rover_color)
        c = patches.Circle(xy=(x, y), radius=self.rover_r, fill=False, color=self.rover_color, ec=self.rover_color)
        ax.add_patch(c)

        # Draw History of real_pose
        ax.plot(
            [e[0] for e in self.real_poses],
            [e[1] for e in self.real_poses],
            linewidth=1.0,
            color=self.rover_color
        )

        # Draw History of estimated_pose
        ax.plot(
            [e[0] for e in self.estimated_poses],
            [e[1] for e in self.estimated_poses],
            linewidth=1.0,
            linestyle=":",
            color=self.rover_color
        )

        # Draw Waypoints if draw_waypoints is True
        if draw_waypoints:
            if self.waypoints is not None:
                ax.plot(
                    [e[0] for e in self.waypoints[-1]],
                    [e[1] for e in self.waypoints[-1]],
                    linewidth=1.0,
                    linestyle="-",
                    color=self.waypoint_color
                )

    def animate(
        self,
        xlim: List[float], ylim: List[float],
        start_step: int = 0, end_step: int = None,
        figsize: Tuple[float, float] = (8, 8),
        obstacles: List[Obstacle] = [],
        enlarge_obstacle: float = 0.0,
        save_path: str = None,
        debug: bool = False
    ) -> None:
        end_step = len(self.steps) if end_step is None else end_step
        end_step = end_step if end_step > len(self.steps) else len(self.steps)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        self.xlim = xlim
        self.ylim = ylim

        # Draw Enlarged Obstacle Regions
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        # Draw Obstacles
        for obstacle in obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
            ax.add_patch(obs)

        elems = []

        # Start Animation
        pbar = tqdm(total=end_step - start_step)
        if debug is True:
            for i in range(end_step - start_step):
                self.animate_one_step(i, ax, elems, start_step, pbar)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.animate_one_step, fargs=(ax, elems, start_step, pbar),
                frames=end_step - start_step, interval=int(self.time_interval * 1000),
                repeat=False
            )
            plt.show()
        if save_path is not None:
            self.ani.save(save_path, writer='ffmpeg')

    def animate_one_step(self, i, ax, elems, start_step, pbar):
        while elems:
            elems.pop().remove()

        time_str = "t = %.2f[s]" % (self.time_interval * (start_step + i))
        elems.append(
            ax.text(
                self.xlim[0] * 0.01,
                self.ylim[1] * 1.02,
                time_str,
                fontsize=10
            )
        )

        # Draw History of sensing_result
        real_pose = self.real_poses[start_step + i]
        sensed_obstacles = self.sensing_results[start_step + i]
        estimated_pose = self.estimated_poses[start_step + i]
        if not sensed_obstacles is None:
            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5)

            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5)
            sensing_range = patches.Wedge(
                (real_pose[0], real_pose[1]), self.sensor_range,
                theta1=np.rad2deg(real_pose[2] - self.sensor_fov / 2),
                theta2=np.rad2deg(real_pose[2] + self.sensor_fov / 2),
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
                enl_obs = patches.Circle(xy=(xn, yn), radius=radius + self.rover_r, fc='blue', ec='blue', alpha=0.3)
                elems.append(ax.add_patch(enl_obs))

        # Draw History of waypoints
        if len(self.waypoints) != 0:
            waypoints = self.waypoints[start_step + i]
            elems += ax.plot(
                [e[0] for e in waypoints],
                [e[1] for e in waypoints],
                linewidth=1.0,
                linestyle=":",
                color="blue",
                alpha=0.5
            )

        # Draw History of estimated_pose
        xn, yn = estimated_pose[0:2] + self.rover_r * np.array([np.cos(estimated_pose[2]), np.sin(estimated_pose[2])])
        elems += ax.plot([estimated_pose[0], xn], [estimated_pose[1], yn], color=self.rover_color, alpha=0.5)
        elems += ax.plot(
            [e[0] for e in self.estimated_poses[start_step:start_step + i + 1]],
            [e[1] for e in self.estimated_poses[start_step:start_step + i + 1]],
            linewidth=1.0,
            linestyle=":",
            color=self.rover_color,
            alpha=0.5
        )
        c = patches.Circle(xy=(estimated_pose[0], estimated_pose[1]), radius=self.rover_r, fill=False, color=self.rover_color, alpha=0.5)
        elems.append(ax.add_patch(c))

        # Draw History of real_pose
        xn, yn = real_pose[0:2] + self.rover_r * np.array([np.cos(real_pose[2]), np.sin(real_pose[2])])
        elems += ax.plot([real_pose[0], xn], [real_pose[1], yn], color=self.rover_color)
        elems += ax.plot(
            [e[0] for e in self.real_poses[start_step:start_step + i + 1]],
            [e[1] for e in self.real_poses[start_step:start_step + i + 1]],
            linewidth=1.0,
            color=self.rover_color
        )
        c = patches.Circle(xy=(real_pose[0], real_pose[1]), radius=self.rover_r, fill=False, color=self.rover_color)
        elems.append(ax.add_patch(c))

        pbar.update(1) if not pbar is None else None


class HistoryWithKalmanFilter(History):
    def __init__(self, time_interval: float = 0.1, rover_r: float = 0.5, sensor_range: float = 10, sensor_fov: float = np.pi / 2, rover_color: str = 'black', waypoint_color: str = 'blue') -> None:
        super().__init__(time_interval, rover_r, sensor_range, sensor_fov, rover_color, waypoint_color)
        self.estimated_poses_cov = []

    def append(self, *args, **kwargs) -> None:
        super().append(*args, **kwargs)
        self.estimated_poses_cov.append(kwargs['estimated_pose_cov'])

    def plot(
        self,
        xlim: List[float], ylim: List[float],
        figsize: Tuple[float, float] = (8, 8),
        obstacles: List[Obstacle] = [],
        enlarge_obstacle: float = 0.0,
        draw_waypoints: bool = False,
        draw_sensing_points: bool = True,
        draw_sensing_area: bool = True,
        plot_step_uncertainty: int = 20
    ):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        # Draw Enlarged Obstacle Regions
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        # Draw Obstacles
        for obstacle in obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
            ax.add_patch(obs)

        # Draw Sensing Results
        for i, sensing_result in enumerate(self.sensing_results):
            if sensing_result is not None:
                if draw_sensing_points:
                    ax.plot(self.real_poses[i][0], self.real_poses[i][1], marker="o", c="red", ms=5)
                if draw_sensing_area:
                    x, y, theta = self.real_poses[i]
                    ax.plot(x, y, marker="o", c="red", ms=5)
                    if draw_sensing_area:
                        sensing_range = patches.Wedge(
                            (x, y), self.sensor_range,
                            theta1=np.rad2deg(theta - self.sensor_fov / 2),
                            theta2=np.rad2deg(theta + self.sensor_fov / 2),
                            alpha=0.5,
                            color="mistyrose"
                        )
                        ax.add_patch(sensing_range)

        # Draw Last Rover Position
        x, y, theta = self.real_poses[-1]
        xn = x + self.rover_r * np.cos(theta)
        yn = y + self.rover_r * np.sin(theta)
        ax.plot([x, xn], [y, yn], color=self.rover_color)
        c = patches.Circle(xy=(x, y), radius=self.rover_r, fill=False, color=self.rover_color)
        ax.add_patch(c)

        x, y, theta = self.estimated_poses[-1]
        xn = x + self.rover_r * np.cos(theta)
        yn = y + self.rover_r * np.sin(theta)
        ax.plot([x, xn], [y, yn], color=self.rover_color)
        c = patches.Circle(xy=(x, y), radius=self.rover_r, fill=False, color=self.rover_color, ec=self.rover_color)
        ax.add_patch(c)

        # Draw History of real_pose
        # ax.plot(
        #     [e[0] for e in self.real_poses],
        #     [e[1] for e in self.real_poses],
        #     linewidth=1.0,
        #     color=self.rover_color
        # )

        # Draw History of estimated_pose
        ax.plot(
            [e[0] for e in self.estimated_poses],
            [e[1] for e in self.estimated_poses],
            linewidth=1.0,
            linestyle=":",
            color=self.rover_color
        )
        for i, (p, cov) in enumerate(zip(self.estimated_poses, self.estimated_poses_cov)):
            if i % plot_step_uncertainty == 0:
                e = sigma_ellipse(p[0:2], cov[0:2, 0:2], 3)
                ax.add_patch(e)

                x, y, c = p
                sigma3 = math.sqrt(cov[2, 2]) * 3
                xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
                ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
                ax.plot(xs, ys, color="blue", alpha=0.5)

        # Draw Waypoints if draw_waypoints is True
        if draw_waypoints:
            if self.waypoints is not None:
                ax.plot(
                    [e[0] for e in self.waypoints[-1]],
                    [e[1] for e in self.waypoints[-1]],
                    linewidth=1.0,
                    linestyle="-",
                    color=self.waypoint_color
                )

    def animate_one_step(self, i, ax, elems, start_step, pbar):
        while elems:
            elems.pop().remove()

        time_str = "t = %.2f[s]" % (self.time_interval * (start_step + i))
        elems.append(
            ax.text(
                self.xlim[0] * 0.01,
                self.ylim[1] * 1.02,
                time_str,
                fontsize=10
            )
        )

        # Draw History of sensing_result
        real_pose = self.real_poses[start_step + i]
        sensed_obstacles = self.sensing_results[start_step + i]
        estimated_pose = self.estimated_poses[start_step + i]
        estimated_pose_cov = self.estimated_poses_cov[start_step + i]
        if not sensed_obstacles is None:
            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5)

            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5)
            sensing_range = patches.Wedge(
                (real_pose[0], real_pose[1]), self.sensor_range,
                theta1=np.rad2deg(real_pose[2] - self.sensor_fov / 2),
                theta2=np.rad2deg(real_pose[2] + self.sensor_fov / 2),
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
                enl_obs = patches.Circle(xy=(xn, yn), radius=radius + self.rover_r, fc='blue', ec='blue', alpha=0.3)
                elems.append(ax.add_patch(enl_obs))

        # Draw History of waypoints
        if len(self.waypoints) != 0:
            waypoints = self.waypoints[start_step + i]
            elems += ax.plot(
                [e[0] for e in waypoints],
                [e[1] for e in waypoints],
                linewidth=1.0,
                linestyle=":",
                color="blue",
                alpha=0.5
            )

        # Draw History of estimated_pose
        xn, yn = estimated_pose[0:2] + self.rover_r * np.array([np.cos(estimated_pose[2]), np.sin(estimated_pose[2])])
        elems += ax.plot([estimated_pose[0], xn], [estimated_pose[1], yn], color=self.rover_color, alpha=0.5)
        elems += ax.plot(
            [e[0] for e in self.estimated_poses[start_step:start_step + i + 1]],
            [e[1] for e in self.estimated_poses[start_step:start_step + i + 1]],
            linewidth=1.0,
            linestyle=":",
            color=self.rover_color,
            alpha=0.5
        )
        cir = patches.Circle(xy=(estimated_pose[0], estimated_pose[1]), radius=self.rover_r, fill=False, color=self.rover_color, alpha=0.5)
        elems.append(ax.add_patch(cir))

        p = estimated_pose
        cov = estimated_pose_cov
        e = sigma_ellipse(p[0:2], cov[0:2, 0:2], 3)
        elems.append(ax.add_patch(e))

        x, y, c = p
        sigma3 = math.sqrt(cov[2, 2]) * 3
        xs = [x + math.cos(c - sigma3), x, x + math.cos(c + sigma3)]
        ys = [y + math.sin(c - sigma3), y, y + math.sin(c + sigma3)]
        elems += ax.plot(xs, ys, color="blue", alpha=0.5)

        # Draw History of real_pose
        xn, yn = real_pose[0:2] + self.rover_r * np.array([np.cos(real_pose[2]), np.sin(real_pose[2])])
        elems += ax.plot([real_pose[0], xn], [real_pose[1], yn], color=self.rover_color)
        elems += ax.plot(
            [e[0] for e in self.real_poses[start_step:start_step + i + 1]],
            [e[1] for e in self.real_poses[start_step:start_step + i + 1]],
            linewidth=1.0,
            color=self.rover_color
        )
        c = patches.Circle(xy=(real_pose[0], real_pose[1]), radius=self.rover_r, fill=False, color=self.rover_color)
        elems.append(ax.add_patch(c))

        pbar.update(1) if not pbar is None else None
