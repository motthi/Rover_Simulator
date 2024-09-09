import sys
import copy
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from rover_simulator.core import Obstacle, History, Sensor
from rover_simulator.utils.draw import *

if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class SimpleHistory(History):
    real_poses: list
    estimated_poses: list
    waypoints: list
    sensing_results: list
    waypoint_color: str

    def __init__(
        self,
        time_interval: float = 0.1,
        rover_r: float = 0.5,
        sensor: Sensor = None,
        rover_color: str = 'black',
        waypoint_color: str = 'blue'
    ) -> None:
        self.steps = []
        self.real_poses = []
        self.estimated_poses = []
        self.sensing_results = []
        self.waypoints = []
        self.time_interval = time_interval
        self.rover_r = rover_r
        self.sensor = sensor
        self.rover_color = rover_color
        self.waypoint_color = waypoint_color

    def append(self, *args, **kwargs) -> None:
        self.steps.append(len(self.steps))
        self.real_poses.append(kwargs['real_pose'])
        self.estimated_poses.append(kwargs['estimated_pose'])
        if 'sensing_result' in kwargs:
            self.sensing_results.append(kwargs['sensing_result'])
        if 'waypoints' in kwargs:
            self.waypoints.append(copy.copy(kwargs['waypoints']))

    def save(self, src) -> None:
        np.savez(
            src,
            steps=np.array(self.steps),
            real_poses=np.array(self.real_poses),
            estimated_poses=np.array(self.estimated_poses),
            # sensing_results = np.array(self.sensing_results),
            waypoints=self.waypoints,
            time_interval=self.time_interval,
            rover_r=self.rover_r,
            rover_color=self.rover_color,
            waypoint_color=self.waypoint_color
        )

    def load(self, src) -> None:
        data = np.load(src, allow_pickle=True)
        self.stesp = list(data['steps'])
        self.real_poses = data['real_poses']
        self.estimated_poses = data['estimated_poses']
        # self.sensing_results = list(data['sensing_results'])
        self.waypoints = data['waypoints']
        self.time_interval = data['time_interval'].item()
        self.rover_r = data['rover_r'].item()
        self.rover_color = data['rover_color'].item()
        self.waypoint_color = data['waypoint_color'].item()

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
        draw_waypoints_flag: bool = False,
        draw_sensing_points_flag: bool = False,
        draw_sensing_area_flag: bool = False
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)

        # draw_sensing_results(ax, self.real_poses, self.sensor.range, self.sensor.fov, self.sensing_results, draw_sensing_points_flag, draw_sensing_area_flag) @todo re-implement
        draw_rover(ax, self.real_poses[-1], self.rover_r, self.rover_color)  # Last rover position and angle
        draw_rover(ax, self.estimated_poses[-1], self.rover_r, self.rover_color)  # Last rover position and angle

        draw_poses(ax, self.real_poses, self.rover_color)
        draw_poses(ax, self.estimated_poses, self.rover_color, linestyle=":")

        # Draw Waypoints if draw_waypoints is True
        draw_pose(ax, self.waypoints[-1], self.waypoint_color) if draw_waypoints_flag and self.waypoints is not None else None

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        start_step: int = 0, end_step: int = None,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
        draw_waypoints_flag: bool = False,
        draw_sensing_results_flag: bool = False,
        draw_sensing_points_flag: bool = True
    ) -> None:
        end_step = len(self.steps) if end_step is None else end_step
        end_step = end_step if end_step > len(self.steps) else len(self.steps)
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)
        draw_start(ax, start_pos) if start_pos is not None else None
        draw_goal(ax, goal_pos) if goal_pos is not None else None

        elems = []

        # Start Animation
        pbar = tqdm(total=end_step - start_step)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, end_step - start_step, interval=int(self.time_interval * 1000),
            fargs=(ax, xlim, ylim, elems, start_step, draw_waypoints_flag, draw_sensing_results_flag, draw_sensing_points_flag, pbar),
            repeat=False
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, xlim: list, ylim: list, elems: list, start_step: int, draw_waypoints_flag: bool, draw_sensing_results_flag: bool, draw_sensing_points_flag: bool, pbar):
        while elems:
            elems.pop().remove()

        time_str = f"t = {self.time_interval * (start_step + i):.2f}[s]"
        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                time_str,
                fontsize=10
            )
        )

        real_pose = self.real_poses[start_step + i]
        if draw_sensing_results_flag:
            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5)
            self.sensor.draw(ax, elems, self.sensing_results[start_step + i], real_pose)
        draw_history_waypoints(ax, elems, self.waypoints, start_step + i) if draw_waypoints_flag and len(self.waypoints) != 0 else None
        draw_history_pose(ax, elems, self.estimated_poses, self.rover_r, self.rover_color, i, start_step)
        draw_history_pose(ax, elems, self.real_poses, self.rover_r, self.rover_color, i, start_step)
        pbar.update(1) if not pbar is None else None


class HistoryWithKalmanFilter(SimpleHistory):
    def __init__(self, time_interval: float = 0.1, rover_r: float = 0.5, sensor: Sensor = None, rover_color: str = 'black', waypoint_color: str = 'blue') -> None:
        super().__init__(time_interval, rover_r, sensor, rover_color, waypoint_color)
        self.estimated_covs = []

    def append(self, *args, **kwargs) -> None:
        super().append(*args, **kwargs)
        self.estimated_covs.append(kwargs['estimated_pose_cov'])

    def save(self, src) -> None:
        np.savez(
            src,
            steps=np.array(self.steps),
            real_poses=np.array(self.real_poses),
            estimated_poses=np.array(self.estimated_poses),
            estimated_covs=np.array(self.estimated_covs),
            # sensing_results = np.array(self.sensing_results),
            waypoints=np.array(self.waypoints),
            sensor_range=self.sensor_range,
            sensor_fov=self.sensor_fov,
            time_interval=self.time_interval,
            rover_r=self.rover_r,
            rover_color=self.rover_color,
            waypoint_color=self.waypoint_color
        )

    def load(self, src) -> None:
        data = np.load(src, allow_pickle=True)
        self.stesp = list(data['steps'])
        self.real_poses = data['real_poses']
        self.estimated_poses = data['estimated_poses']
        self.estimated_covs = data['estimated_covs']
        # self.sensing_results = list(data['sensing_results'])
        self.waypoints = data['waypoints']
        self.sensor_range = data['sensor_range'].item()
        self.sensor_fov = data['sensor_fov'].item()
        self.time_interval = data['time_interval'].item()
        self.rover_r = data['rover_r'].item()
        self.rover_color = data['rover_color'].item()
        self.waypoint_color = data['waypoint_color'].item()

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
        draw_waypoints_flag: bool = False,
        draw_error_ellipse_flag: bool = True,
        draw_sensing_results_flag: bool = False,
        draw_sensing_points_flag: bool = False,
        draw_sensing_area_flag: bool = False,
        plot_step_uncertainty: int = 20
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)

        draw_sensing_results(ax, self.real_poses, self.sensor_range, self.sensor_fov, self.sensing_results, draw_sensing_points_flag, draw_sensing_area_flag) if draw_sensing_results_flag else None
        draw_rover(ax, self.real_poses[-1], self.rover_r, self.rover_color)  # Last rover position and angle
        draw_rover(ax, self.estimated_poses[-1], self.rover_r, self.rover_color)  # Last rover position and angle
        draw_poses(ax, self.real_poses, self.rover_color, linestyle="-")
        draw_poses(ax, self.estimated_poses, self.rover_color, linestyle=":")
        draw_error_ellipses(ax, self.estimated_poses, self.estimated_covs, "blue", plot_step_uncertainty) if draw_error_ellipse_flag else None
        draw_pose(ax, self.waypoints[-1], self.waypoint_color) if draw_waypoints_flag and self.waypoints is not None else None

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        start_step: int = 0, end_step: int = None,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
        draw_waypoints_flag: bool = False,
        draw_error_ellipse_flag: bool = True,
        draw_sensing_results_flag: bool = False,
        draw_sensing_points_flag: bool = True,
        draw_sensing_area_flag: bool = True,
    ) -> None:
        end_step = len(self.steps) if end_step is None else end_step
        end_step = end_step if end_step > len(self.steps) else len(self.steps)
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)
        draw_start(ax, start_pos) if start_pos is not None else None
        draw_goal(ax, goal_pos) if goal_pos is not None else None

        elems = []

        # Start Animation
        pbar = tqdm(total=end_step - start_step)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, end_step - start_step, interval=int(self.time_interval * 1000), repeat=False,
            fargs=(ax, xlim, ylim, elems, start_step, draw_waypoints_flag, draw_error_ellipse_flag, draw_sensing_results_flag, draw_sensing_points_flag, draw_sensing_area_flag, pbar),
        )
        plt.close()

    def animate_one_step(
            self,
            i: int, ax: Axes,
            xlim: list, ylim: list, elems: list,
            start_step: int,
            draw_waypoints_flag: bool, draw_error_ellipse_flag: bool, draw_sensing_results_flag: bool,
            draw_sensing_points_flag: bool, draw_sensing_area_flag: bool,
            pbar
    ):
        while elems:
            elems.pop().remove()

        time_str = f"t = {self.time_interval * (start_step + i):.2f}[s]"
        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                time_str,
                fontsize=10
            )
        )

        real_pose = self.real_poses[start_step + i]
        if draw_sensing_results_flag:
            ax.plot(real_pose[0], real_pose[1], marker="o", c="red", ms=5) if draw_sensing_points_flag is True else None
            self.sensor.draw(ax, elems, self.sensing_results[start_step + i], real_pose)
        draw_history_waypoints(ax, elems, self.waypoints, start_step + i) if draw_waypoints_flag and len(self.waypoints) != 0 else None
        draw_history_pose_with_error_ellipse(ax, elems, self.estimated_poses, self.estimated_covs, self.rover_r, self.rover_color, "blue", i, start_step) if draw_error_ellipse_flag else None
        draw_history_pose(ax, elems, self.real_poses, self.rover_r, self.rover_color, i, start_step)
        pbar.update(1) if not pbar is None else None
