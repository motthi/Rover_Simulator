
import sys
import copy
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from rover_simulator.core import*
from rover_simulator.utils import occupancyToColor, state_transition
from rover_simulator.core import Obstacle, SensingPlanner
from rover_simulator.world import World
from rover_simulator.collision_detector import IgnoreCollision
from rover_simulator.navigation.path_planner import PathPlanner
from rover_simulator.navigation.mapper import drawGrid
from rover_simulator.navigation.controller import DWAController
from rover_simulator.navigation.sensing_planner import SimpleSensingPlanner

if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class History():
    def __init__(self) -> None:
        self.steps = []
        self.real_poses = []
        self.estimated_poses = []
        self.sensing_results = []
        self.waypoints = []

    def append(self, *args, **kwargs) -> None:
        self.steps.append(len(self.steps))
        self.real_poses.append(kwargs['real_pose'])
        self.estimated_poses.append(kwargs['estimated_pose'])
        self.sensing_results.append(kwargs['sensing_result'])
        if 'waypoints' in kwargs:
            self.waypoints.append(copy.copy(kwargs['waypoints']))


class BasicRover(Rover):
    def __init__(
        self,
        pose: np.ndarray, radius: float,
        sensor: Sensor = None, localizer: Localizer = None, path_planner: PathPlanner = None,
        controller: Controller = None, sensing_planner: SensingPlanner = None,
        mapper: Mapper = None, collision_detector: CollisionDetector = IgnoreCollision(),
        history: History = None,
        color: str = "black", waypoint_color: str = 'blue'
    ) -> None:
        if not pose.shape == (3, ):
            raise ValueError("array 'pose' is not of the right shape (3,), given array's shape is {}".format(pose.shape))
        self.real_pose = pose
        self.estimated_pose = pose
        self.r = radius
        self.color = color
        self.waypoint_color = waypoint_color

        self.sensor = sensor
        self.localizer = localizer
        self.path_planner = path_planner
        self.controller = controller
        self.sensing_planner = sensing_planner
        self.mapper = mapper
        self.collision_detector = collision_detector
        self.history = history
        self.waypoints = []

    def one_step(self, time_interval: float) -> None:
        if self.collision_detector.detect_collision(self):
            self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=[])
            return

        # Sensing
        sensed_obstacles = self.sensor.sense(self) if self.sensor is not None else []

        # Mapping
        self.mapper.update(self.estimated_pose, sensed_obstacles) if self.mapper is not None else None

        # Calculate Control Inputs
        control_inputs = self.controller.calculate_control_inputs()

        # Record
        self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=sensed_obstacles)

        # Move
        self.real_pose = state_transition(self.real_pose, control_inputs, time_interval)

        # Localization
        self.estimated_pose = self.localizer.estimate_pose(self.estimated_pose, control_inputs, time_interval)


class DWARover(BasicRover):
    def __init__(
        self,
        pose: np.ndarray, radius: float,
        sensor: Sensor = None, localizer: Localizer = None, path_planner: PathPlanner = None,
        controller: Controller = DWAController(),
        sensing_planner: SensingPlanner = None,
        mapper: Mapper = None, collision_detector: CollisionDetector = None,
        history: History = None,
        color: str = "black", waypoint_color: str = 'blue',
        waypoint_pos: np.ndarray = np.array([18, 18])
    ) -> None:
        super().__init__(
            pose=pose, radius=radius,
            sensor=sensor, localizer=localizer, path_planner=path_planner,
            controller=controller, sensing_planner=sensing_planner,
            mapper=mapper, collision_detector=collision_detector,
            history=history, color=color, waypoint_color=waypoint_color
        )
        self.v = 0.0
        self.w = 0.0
        self.waypoint = waypoint_pos

    def one_step(self, time_interval: float) -> None:
        # Collision Check
        if self.collision_detector.detect_collision(self):
            self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=[], waypoints=self.waypoints)
            return

        # Goal Check
        if np.linalg.norm(self.waypoint[0:2] - self.estimated_pose[0:2]) < 1.0:
            self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=[], waypoints=self.waypoints)
            return

        # Sensing
        sensed_list = self.sensor.sense(self) if self.sensor is not None else []
        sensed_obstacles = []
        for sensed_obj in sensed_list:
            distance = sensed_obj['distance']
            angle = sensed_obj['angle'] + self.estimated_pose[2]
            obs_pos = self.real_pose[0:2] + distance * np.array([np.cos(angle), np.sin(angle)])
            sensed_obstacles.append(Obstacle(obs_pos, sensed_obj['radius']))

        # Calculate Control Inputs
        control_inputs = self.controller.calculate_control_inputs(
            rover_pose=self.estimated_pose, dt=time_interval,
            goal_pose=self.waypoint, obstacles=sensed_obstacles,
            v=self.v, w=self.w,
        )
        self.v, self.w = control_inputs

        # Record
        self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=sensed_list, waypoints=self.waypoints)

        # Move
        self.real_pose = state_transition(self.real_pose, control_inputs, time_interval)

        # Localization
        self.estimated_pose = self.localizer.estimate_pose(self.estimated_pose, control_inputs, time_interval)


class FollowRover(DWARover):
    def __init__(
        self,
        pose: np.ndarray, radius: float,
        sensor: Sensor = None, localizer: Localizer = None, path_planner: PathPlanner = None,
        controller: Controller = DWAController(),
        sensing_planner: SensingPlanner = None,
        mapper: Mapper = None, collision_detector: CollisionDetector = None,
        history: History = None,
        color: str = "black", waypoint_color: str = 'blue',
        goal_pos: np.ndarray = np.array([18, 18]),
        calculate_path_args=None,
        waypoint_dist=3.0
    ) -> None:
        super().__init__(
            pose, radius,
            sensor=sensor, localizer=localizer, path_planner=path_planner, controller=controller,
            sensing_planner=sensing_planner, mapper=mapper, collision_detector=collision_detector,
            history=history, color=color, waypoint_color=waypoint_color, waypoint_pos=None
        )
        self.path_planner.set_start(pose)
        self.path_planner.set_goal(goal_pos)
        self.path_planner.set_map(mapper) if hasattr(self.path_planner, 'set_map') else None
        self.calculate_path_args = calculate_path_args
        self.goal_poe = goal_pos
        self.cnt_waypoint = 0
        self.waypoint_dist = waypoint_dist

    def one_step(self, time_interval: float) -> None:
        self.waypoint = self.waypoints[0]
        while np.linalg.norm(self.estimated_pose[0:2] - self.waypoint[0:2]) < self.waypoint_dist and len(self.waypoints) > 1:
            self.waypoints.pop(0)
            self.waypoint = self.waypoints[0]
        super().one_step(time_interval)


class OnlinePathPlanningRover(DWARover):
    def __init__(
        self,
        pose: np.ndarray, radius: float,
        sensor: Sensor = None, localizer: Localizer = None, path_planner: PathPlanner = None,
        controller: Controller = DWAController(),
        sensing_planner: SensingPlanner = SimpleSensingPlanner(),
        mapper: Mapper = None, collision_detector: CollisionDetector = None,
        history: History = None,
        color: str = "black", waypoint_color: str = 'blue',
        goal_pos: np.ndarray = np.array([18, 18])
    ) -> None:
        super().__init__(
            pose, radius,
            sensor=sensor, localizer=localizer, path_planner=path_planner, controller=controller,
            sensing_planner=sensing_planner, mapper=mapper, collision_detector=collision_detector,
            history=history, color=color, waypoint_color=waypoint_color, waypoint_pos=None
        )
        self.path_planner.set_start(pose)
        self.path_planner.set_goal(goal_pos)
        self.path_planner.set_map(mapper) if hasattr(self.path_planner, 'set_map') else None
        self.goal_pos = goal_pos

    def one_step(self, time_interval: float) -> None:
        sensed_obstacles = []

        # Collision Detection
        if self.collision_detector.detect_collision(self):
            self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=sensed_obstacles, waypoints=self.waypoints)
            return

        # Goal Check
        if np.linalg.norm(self.goal_pos[0:2] - self.estimated_pose[0:2]) < 1.0:
            self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=sensed_obstacles, waypoints=self.waypoints)
            return

        # Sensing Planning
        sense_plan_flag = self.sensing_planner.decide(self.estimated_pose)

        # Sensing
        if sense_plan_flag:
            sensed_obstacles = self.sensor.sense(self) if self.sensor is not None else []

        # Mapping
        if sense_plan_flag:
            self.mapper.update(self.estimated_pose, sensed_obstacles) if self.mapper is not None else None

        # Path Planning
        if not self.mapper.isOutOfBounds(self.mapper.poseToIndex(self.estimated_pose)):
            self.waypoints = self.path_planner.update_path(self.estimated_pose, self.mapper)

        # Set next waypoint
        self.waypoint = self.waypoints[0]
        while np.linalg.norm(self.estimated_pose[0:2] - self.waypoint[0:2]) < 3.0 and len(self.waypoints) > 1:
            self.waypoints.pop(0)
            self.waypoint = self.waypoints[0]

        # Calculate Control Inputs
        control_inputs = self.controller.calculate_control_inputs(
            rover_pose=self.estimated_pose, v=self.v, w=self.w, dt=time_interval,
            goal_pose=self.waypoint, obstacles=self.mapper.obstacles_table
        )

        # Record
        self.history.append(real_pose=self.real_pose, estimated_pose=self.estimated_pose, sensing_result=sensed_obstacles, waypoints=self.waypoints)

        # Move one step
        self.real_pose = state_transition(self.real_pose, control_inputs, time_interval)

        # Localization
        self.estimated_pose = self.localizer.estimate_pose(self.estimated_pose, control_inputs, time_interval)


class RoverAnimation():
    def __init__(self, world: World, rover: Rover, path_planner: PathPlanner) -> None:
        self.world = world
        self.rover = rover
        self.path_planner = path_planner

    def animate(
        self,
        xlim: List[float], ylim: List[float],
        start_step: int = 0, end_step: int = None,
        figsize: Tuple[float, float] = (8, 8),
        map_name='cost',
        enlarge_obstacle: float = 0.0,
        save_path: str = None,
        debug: bool = False
    ) -> None:
        end_step = self.world.step if end_step is None else end_step
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)
        self.xlim = xlim
        self.ylim = ylim

        if map_name == 'cost':
            for obstacle in self.world.obstacles:
                enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
                ax.add_patch(enl_obs)
            for obstacle in self.world.obstacles:
                obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
                ax.add_patch(obs)

        self.rover.mapper.reset()
        if not self.path_planner is None:
            self.rover.path_planner = copy.copy(self.path_planner)
            self.rover.path_planner.set_map(self.rover.mapper)
            self.rover.waypoints = self.rover.path_planner.calculate_path()
        else:
            self.rover.path_planner = None

        elems = []
        pbar = tqdm(total=end_step - start_step + 1)
        if debug is True:
            for i in range(end_step - start_step):
                self.animate_one_step(i, ax, elems, start_step, map_name, pbar)
        else:
            self.ani = anm.FuncAnimation(
                fig, self.animate_one_step, fargs=(ax, elems, start_step, map_name, pbar),
                frames=end_step - start_step, interval=int(self.world.time_interval * 1000),
                repeat=False
            )
            plt.show()
        if save_path is not None:
            self.ani.save(save_path, writer='ffmpeg')

    def animate_one_step(self, i, ax, elems, start_step, map_name, pbar):
        while elems:
            elems.pop().remove()

        time_str = "t = %.2f[s]" % (self.world.time_interval * (start_step + i))
        elems.append(
            ax.text(
                self.xlim[0] * 0.01,
                self.ylim[1] * 1.02,
                time_str,
                fontsize=10
            )
        )

        x, y, theta = self.rover.history.estimated_poses[start_step + i]
        xn = x + self.rover.r * np.cos(theta)
        yn = y + self.rover.r * np.sin(theta)

        # Sensing, Mapping and Path Planning
        sensed_obstacles = self.rover.history.sensing_results[start_step + i]
        if sensed_obstacles is not None:
            self.rover.mapper.update(self.rover.history.estimated_poses[start_step + i], sensed_obstacles)
            if self.rover.path_planner is not None and not self.rover.mapper.isOutOfBounds(self.rover.mapper.poseToIndex(self.rover.history.estimated_poses[start_step + i])):
                self.waypoints = self.rover.path_planner.update_path(self.rover.history.estimated_poses[start_step + i], self.rover.mapper)
            sensing_range = patches.Wedge(
                (x, y), self.rover.sensor.range,
                theta1=np.rad2deg(theta - self.rover.sensor.fov / 2),
                theta2=np.rad2deg(theta + self.rover.sensor.fov / 2),
                alpha=0.5,
                color="mistyrose"
            )
            elems.append(ax.add_patch(sensing_range))

            if self.rover.mapper.retain_range is not None:
                map_range = patches.Circle(xy=(x, y), radius=self.rover.mapper.retain_range, ec='blue', fill=False)
                elems.append(ax.add_patch(map_range))

        if map_name == 'table':
            for obstacle in self.rover.mapper.obstacles_table:
                enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + self.rover.r, fc='gray', ec='gray')
                elems.append(ax.add_patch(enl_obs))

            for obstacle in self.rover.mapper.obstacles_table:
                obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
                elems.append(ax.add_patch(obs))
        else:
            for idx, _ in np.ndenumerate(self.rover.mapper.map):
                if map_name == 'cost':
                    cost_adj = 5
                    if not self.rover.path_planner.g(idx) < 100000:  # LOWER状態のセルを描画
                        continue
                    c_num = int(self.rover.path_planner.g(idx))  # Black→Blue
                    c_num = int(c_num * cost_adj)
                    if c_num > 0xff:  # Blue → Cyan
                        c_num = (c_num - 0xff) * 16 * 16 + 0xff
                        if c_num > 0xffff:  # Cyan → Green
                            c_num = 0xffff - int((c_num - 0x100ff) * 4 / 256)
                            if c_num < 0xff00:  # Green →Yellow
                                c_num = (0xff00 - c_num) * 65536 + 0xff00
                                if c_num > 0xffff00:  # Yellow → Red
                                    c_num = 0xffff00 - int((c_num - 0xffff00) * 0.5 / 65536) * 256
                    fill = True
                    alpha = 0.5
                    c = '#' + format(int(c_num), 'x').zfill(6)
                elif map_name == 'metric':
                    fill = True
                    alpha = 1.0
                    occupancy = self.rover.path_planner.metric_map(idx)
                    if occupancy < 0:
                        c = "gray"
                    else:
                        c = occupancyToColor(occupancy)
                elif map_name == 'local':
                    fill = True
                    alpha = 1.0
                    occupancy = self.rover.path_planner.local_map(idx)
                    c = occupancyToColor(occupancy)
                elif map_name == 'map':
                    fill = True
                    alpha = 1.0
                    occupancy = self.rover.mapper.map[idx[0]][idx[1]]
                    c = occupancyToColor(occupancy)
                drawGrid(np.array(idx), self.rover.mapper.grid_width, c, alpha, ax, elems, fill)

        # Draw rover real pose history
        elems += ax.plot([x, xn], [y, yn], color=self.rover.color)
        elems += ax.plot(
            [e[0] for e in self.rover.history.real_poses[start_step:start_step + i + 1]],
            [e[1] for e in self.rover.history.real_poses[start_step:start_step + i + 1]],
            linewidth=1.0,
            color=self.rover.color
        )

        # Draw rover
        c = patches.Circle(xy=(x, y), radius=self.rover.r, fill=False, color=self.rover.color)
        elems.append(ax.add_patch(c))

        pbar.update(1)
