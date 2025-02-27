import re
import sys
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rover_simulator.core import *
from rover_simulator.utils.draw import *
from matplotlib.figure import Figure
from matplotlib.axes import Axes

if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class World():
    rovers: list[Rover]
    obstacles: list[Obstacle]
    fig: Figure
    ax: Axes

    def __init__(self, time_interval: float = 0.1) -> None:
        self.rovers = []
        self.obstacles = []
        self.step = 0
        self.time_interval = time_interval
        self.fig = None
        self.ax = None
        self.ani = None

    def simulate(self, steps: int = 100, tqdm_leave: bool = True, tqdm_desc="Simulation"):
        for _ in tqdm(range(steps), desc=tqdm_desc, leave=tqdm_leave):
            self.one_step()
            self.step += 1
            """
            最後にhistory.appendをする必要がある
            現状，最後のone_stepは保存されない
            """

    def simulate_rover_to_goal(
        self,
        rover_idx: int,
        goal_pos: np.ndarray, goal_range: float = 2.0,
        stuck_step: int = 50,
        stuck_distance: float = 5.0,
        final_step: int = 1000
    ):
        dist_to_goal = float('inf')
        rover = self.rovers[rover_idx]
        rover_poses = [rover.estimated_pose]
        stuck_flag = False
        while dist_to_goal >= goal_range:
            if rover.collision_detector.detect_collision(rover):
                return "Collided"
            rover.one_step(self.time_interval)
            self.step += 1
            dist_to_goal = np.linalg.norm(rover.estimated_pose[0:2] - goal_pos[0:2])
            rover_poses.append(rover.estimated_pose)
            if len(rover_poses) > stuck_step:
                traversed_dist = 0.0
                for i in range(stuck_step):
                    traversed_dist += np.linalg.norm(rover_poses[i - stuck_step - 1][0:2] - rover_poses[i - stuck_step][0:2])
                if traversed_dist < stuck_distance:
                    stuck_flag = True
                    break
            if self.step > final_step:
                stuck_flag = True
                break
        if dist_to_goal < goal_range:
            return "Succeed"
        elif stuck_flag:
            return "Stucked"
        else:
            return "Collided"

    def one_step(self):
        for rover in self.rovers:
            rover.one_step(self.time_interval)

    def read_objects(self, src: str):
        f = open(src)
        f_lines = f.readlines()
        for f_line in f_lines:
            f_line = f_line.split(',')
            if f_line[0] == 'Obstacle':
                if f_line[1] == 'Circle':
                    self.append_obstacle(CircularObstacle(np.array([float(f_line[2]), float(f_line[3])]), float(f_line[4])))
                elif f_line[1] == 'Rectangle':
                    self.append_obstacle(RectangularObstacle(np.array([float(f_line[2]), float(f_line[3])]), float(f_line[4]), float(f_line[5]), float(f_line[6])))

    def append_rover(self, rover: Rover):
        self.rovers.append(rover)

    def append_obstacle(self, obstacle: Obstacle):
        self.obstacles.append(obstacle)

    def set_start_goal_ramdomly(
        self,
        x_range: list[float] = [0, 20], y_range: list[float] = [0, 20],
        min_dist: float = 0.0, max_dist: float = 20,
        expand_dist=0.0
    ) -> list[np.ndarray]:
        if min_dist > max_dist:
            raise ValueError("min_distance must be lower than max_distance")
        distance = -1.0

        obstacle_kdTree = None
        if len(self.obstacles) != 0:
            obstacle_kdTree = cKDTree([obstacle.pos for obstacle in self.obstacles])

        is_collision = True
        while distance < min_dist or distance > max_dist or is_collision:
            start_pos = np.array([np.random.uniform(x_range[0], x_range[1]), np.random.uniform(y_range[0], y_range[1])])
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(min_dist, max_dist)
            goal_pos = start_pos + np.array([distance * np.cos(angle), distance * np.sin(angle)])

            if not (x_range[0] <= goal_pos[0] <= x_range[1] and y_range[0] <= goal_pos[1] <= y_range[1]):
                continue

            if obstacle_kdTree:
                dist_start, idx_start = obstacle_kdTree.query(start_pos, k=1)
                dist_goal, idx_goal = obstacle_kdTree.query(goal_pos, k=1)
                if dist_start < self.obstacles[idx_start].r + expand_dist + 1.0 or dist_goal < self.obstacles[idx_goal].r + expand_dist + 1.0:
                    is_collision = True
                else:
                    is_collision = False
            else:
                is_collision = False

        return start_pos, goal_pos

    def reset(self, reset_step: bool = True, reset_rovers: bool = True, reset_obstacles: bool = True):
        self.step = 0 if reset_step is True else self.step
        self.rovers = [] if reset_rovers is True else self.rovers
        self.obstacles = [] if reset_obstacles is True else self.obstacles

    def draw(
        self,
        xlim: list[float], ylim: list[float],
        figsize: tuple[float, float] = (8, 8),
        start_pos: np.ndarray = None,
        goal_pos: np.ndarray = None,
        expand_dist: float = 0.0,
        expand_color: str = 'gray',
        legend_flag: bool = False,
        draw_waypoints_flag: bool = False,
        draw_sensing_results_flag: bool = False,
        draw_sensing_points_flag: bool = True,
        draw_sensing_area_flag: bool = True,
        not_show: bool = False
    ):
        self.fig, self.ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(self.ax, self.obstacles, expand_dist, 1.0, expand_color)

        for rover in self.rovers:
            if rover.history:
                draw_poses(self.ax, rover.history.real_poses, rover.color)
                draw_poses(self.ax, rover.history.estimated_poses, rover.color, linestyle=':')
                if draw_sensing_results_flag:
                    draw_sensing_results(self.ax, rover.history.real_poses, rover.sensor.range, rover.sensor.fov, rover.sensing_results, draw_sensing_points_flag, draw_sensing_area_flag)
            draw_rover(self.ax, rover.real_pose, rover.r, rover.color)
            draw_waypoints(self.ax, rover.waypoints, rover.waypoint_color) if draw_waypoints_flag and rover.waypoints is not None else None

        draw_start(self.ax, start_pos) if start_pos is not None else None
        draw_goal(self.ax, goal_pos) if goal_pos is not None else None
        self.ax.legend() if legend_flag is True else None
        plt.show() if not_show is False else None

    def animate(
        self,
        figsize: tuple[float, float] = (8, 8),
        xlim: list[float] = None, ylim: list[float] = None,
        start_step: int = 0, end_step: int = None,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        expand_dist: float = 0.0,
        draw_waypoints_flag: bool = False,
        draw_sensing_points_flag: bool = True,
        draw_sensing_results_flag: bool = True
    ) -> None:
        end_step = self.step if end_step is None else end_step
        self.fig, ax = set_fig_params(figsize, xlim, ylim)

        draw_obstacles(ax, self.obstacles, expand_dist)
        draw_start(ax, start_pos) if start_pos is not None else None
        draw_goal(ax, goal_pos) if goal_pos is not None else None
        elems = []

        # Start Animation
        pbar = tqdm(total=end_step - start_step)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, end_step - start_step, interval=int(self.time_interval * 1000), repeat=False,
            fargs=(ax, xlim, ylim, elems, start_step, draw_waypoints_flag, draw_sensing_points_flag, draw_sensing_results_flag, pbar)
        )
        plt.close()

    def animate_one_step(
            self,
            i: int, ax: Axes, xlim: list, ylim: list, elems: list, start_step: int,
            draw_waypoints_flag: bool, draw_sensing_points_flag: bool, draw_sensing_results_flag: bool,
            pbar
    ):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"t = {self.time_interval * (start_step + i):.2f}[s]" % (),
                fontsize=10
            )
        )

        for rover in self.rovers:
            if rover.history:
                draw_history_pose(ax, elems, rover.history.estimated_poses, rover.r, rover.color, i, start_step)
                draw_history_pose(ax, elems, rover.history.real_poses, rover.r, rover.color, i, start_step)
                draw_history_waypoints(ax, elems, rover.history.waypoints, start_step + i) if draw_waypoints_flag and len(rover.history.waypoints) != 0 else None

                if rover.sensor and draw_sensing_results_flag:
                    rover.sensor.draw(ax, elems, rover.history.sensing_results[start_step + i], rover.history.real_poses[start_step + i])

        pbar.update(1) if not pbar is None else None

    def save_animation(self, src: str, writer='ffmpeg'):
        if self.ani:
            self.ani.save(src, writer=writer)
        else:
            raise Exception("Animation is not created.")
