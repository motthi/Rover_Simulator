import os
import re
import sys
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from rover_simulator.core import*

if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class World():
    def __init__(self, time_interval: float = 0.1, end_step: int = 10) -> None:
        self.rovers = []
        self.obstacles = []
        self.step = 0
        self.time_interval = time_interval
        self.end_step = end_step
        self.fig = None
        self.ani = None

    def simulate(self):
        for _ in tqdm(range(self.step, self.end_step)):
            self.one_step()
            self.step += 1
            """
            最後にhistory.appendをする必要がある
            現状，最後のone_stepは保存されない
            """

    def one_step(self):
        for rover in self.rovers:
            rover.one_step(self.time_interval)

    def read_objects(self, setting_file_path):
        f = open(setting_file_path)
        f_lines = f.readlines()
        pattern_float_num = r'[+-]?(?:\d+\.?\d*|\.\d+)(?:(?:[eE][+-]?\d+)|(?:\*10\^[+-]?\d+))?'
        pattern = r'Obstacle,\s*(' + pattern_float_num + r'),\s*(' + pattern_float_num + r'),\s*(' + pattern_float_num + r'),\s*(' + pattern_float_num + r')'
        for f_line in f_lines:
            match = re.search(pattern=pattern, string=f_line)
            if match is not None:
                x = float(match.group(1))
                y = float(match.group(2))
                r = float(match.group(3))
                t = int(match.group(4))
                self.append_obstacle(Obstacle(np.array([x, y]), r, t))

    def append_rover(self, rover: Rover):
        self.rovers.append(rover)

    def append_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def reset(self):
        self.step = 0
        self.rovers = []
        self.obstacles = []

    def plot(
        self,
        xlim: List[float], ylim: List[float],
        figsize: Tuple[float, float] = (8, 8),
        enlarge_obstacle: float = 0.0,
        draw_waypoints: bool = False
    ):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        # Draw Enlarged Obstacle Regions
        for obstacle in self.obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        # Draw Obstacles
        for obstacle in self.obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
            ax.add_patch(obs)

        for rover in self.rovers:
            if rover.history is not None:
                # Draw History of real_pose
                ax.plot(
                    [e[0] for e in rover.history.real_poses],
                    [e[1] for e in rover.history.real_poses],
                    linewidth=1.0,
                    color=rover.color
                )
                # Draw History of estimated_pose
                ax.plot(
                    [e[0] for e in rover.history.estimated_poses],
                    [e[1] for e in rover.history.estimated_poses],
                    linewidth=1.0,
                    linestyle=":",
                    color=rover.color
                )

            # Draw Last Rover Position
            x, y, theta = rover.real_pose
            xn = x + rover.r * np.cos(theta)
            yn = y + rover.r * np.sin(theta)
            ax.plot([x, xn], [y, yn], color=rover.color)
            c = patches.Circle(xy=(x, y), radius=rover.r, fill=False, color=rover.color)
            ax.add_patch(c)

            # Draw Waypoints if draw_waypoints is True
            if draw_waypoints:
                if rover.waypoints is not None:
                    ax.plot(
                        [e[0] for e in rover.waypoints],
                        [e[1] for e in rover.waypoints],
                        linewidth=1.0,
                        linestyle="-",
                        color=rover.waypoint_color
                    )

        # plt.show()

    def animate(
        self,
        xlim: List[float], ylim: List[float],
        start_step: int = 0, end_step: int = None,
        figsize: Tuple[float, float] = (8, 8),
        enlarge_obstacle: float = 0.0,
        save_path: str = None,
        debug: bool = False
    ) -> None:
        end_step = self.step if end_step is None else end_step
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
        for obstacle in self.obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        # Draw Obstacles
        for obstacle in self.obstacles:
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

        for rover in self.rovers:
            # Draw History of estimated_pose
            x, y, theta = rover.history.estimated_poses[start_step + i]
            xn = x + rover.r * np.cos(theta)
            yn = y + rover.r * np.sin(theta)
            elems += ax.plot([x, xn], [y, yn], color=rover.color, alpha=0.5)
            elems += ax.plot(
                [e[0] for e in rover.history.estimated_poses[start_step:start_step + i + 1]],
                [e[1] for e in rover.history.estimated_poses[start_step:start_step + i + 1]],
                linewidth=1.0,
                linestyle=":",
                color=rover.color,
                alpha=0.5
            )
            c = patches.Circle(xy=(x, y), radius=rover.r, fill=False, color=rover.color, alpha=0.5)
            elems.append(ax.add_patch(c))

            # Draw History of real_pose
            x, y, theta = rover.history.real_poses[start_step + i]
            xn = x + rover.r * np.cos(theta)
            yn = y + rover.r * np.sin(theta)
            elems += ax.plot([x, xn], [y, yn], color=rover.color)
            elems += ax.plot(
                [e[0] for e in rover.history.real_poses[start_step:start_step + i + 1]],
                [e[1] for e in rover.history.real_poses[start_step:start_step + i + 1]],
                linewidth=1.0,
                color=rover.color
            )
            c = patches.Circle(xy=(x, y), radius=rover.r, fill=False, color=rover.color)
            elems.append(ax.add_patch(c))

            # Draw History of sensing_result
            sensed_obstacles = rover.history.sensing_results[start_step + i]
            for sensed_obstacle in sensed_obstacles:
                x, y, theta = rover.history.estimated_poses[start_step + i]

                distance = sensed_obstacle['distance']
                angle = sensed_obstacle['angle'] + theta
                radius = sensed_obstacle['radius']

                # ロボットと障害物を結ぶ線を描写
                xn, yn = np.array([x, y]) + np.array([distance * np.cos(angle), distance * np.sin(angle)])
                elems += ax.plot([x, xn], [y, yn], color="mistyrose")

                # Draw Enlarged Obstacle Regions
                enl_obs = patches.Circle(xy=(xn, yn), radius=radius + rover.r, fc='blue', ec='blue', alpha=0.3)
                elems.append(ax.add_patch(enl_obs))

                # Draw Obstacles
                # for obstacle in self.obstacles:
                #     obs = patches.Circle(xy=(xn, yn), radius=obstacle.r, fc='blue', ec='blue', alpha=0.5)
                #     elems.append(ax.add_patch(obs))

            # Draw History of waypoints
            if len(rover.history.waypoints) != 0:
                waypoints = rover.history.waypoints[start_step + i]
                elems += ax.plot(
                    [e[0] for e in waypoints],
                    [e[1] for e in waypoints],
                    linewidth=1.0,
                    linestyle=":",
                    color="blue",
                    alpha=0.5
                )

        pbar.update(1)
