import math
import sys
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from rover_simulator.core import Controller, Obstacle
from rover_simulator.utils.utils import set_angle_into_range
from rover_simulator.utils.motion import state_transition
from rover_simulator.utils.draw import set_fig_params, draw_history_pose, draw_obstacles, draw_start, draw_goal
from rover_simulator.navigation.mapper import GridMapper


if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


class ConstantSpeedController(Controller):
    def __init__(self, v: float = 1.0, w: float = np.pi / 4) -> None:
        super().__init__()
        self.cont_v = v
        self.cont_w = w

    def calculate_control_inputs(self):
        return self.cont_v, self.cont_w


class PurePursuitController(Controller):
    def __init__(self, v: float = 1.0, L: float = 1.0) -> None:
        self.v = v
        self.L = L

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        goal_pose: np.ndarray
    ):
        theta = np.arctan2(goal_pose[1] - rover_pose[1], goal_pose[0] - rover_pose[0]) - rover_pose[2]
        theta = set_angle_into_range(theta)
        w = 2 * self.v * np.sin(theta) / self.L
        return self.v, w


class DwaController(Controller):
    def __init__(
        self,
        nu_range: list[float] = [-1.0, 2.0],
        omega_range: list[float] = [-np.deg2rad(120), np.deg2rad(120)],
        nu_delta: float = 0.1,
        omega_delta: float = np.deg2rad(30),
        nu_max_acc: float = 0.3,
        omega_max_acc: float = np.deg2rad(60),
        to_goal_cost_gain: float = 0.4,
        speed_gain: float = 1.0,
        obs_cost_gain: float = 1.7,
        predict_time: float = 1.0,
        rover_r: float = 0.0,
        rover_stuck_flag_cons: float = 0.001,
        sensor_type: str = None  # @todo Make it easier to understand
    ) -> None:
        self.nu_min = nu_range[0]
        self.nu_max = nu_range[1]
        self.omega_min = omega_range[0]
        self.omega_max = omega_range[1]
        self.nu_delta = nu_delta
        self.omega_delta = omega_delta
        self.nu_max_acc = nu_max_acc
        self.omega_max_acc = omega_max_acc

        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_gain = speed_gain
        self.obs_cost_gain = obs_cost_gain

        self.predict_time = predict_time
        self.rover_r = rover_r
        self.rover_stuck_flag_cons = rover_stuck_flag_cons

        self.sensor_type = sensor_type

        self.path_primitive_log = []
        self.best_path_log = []

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        v: float, w: float, dt: float,
        goal_pose: np.ndarray,
        sensing_result: np.ndarray  # @todo Only for the sensor? No compatibility with map?
    ):
        if len(sensing_result) != 0:
            if self.sensor_type == 'stereo_camera':
                sensing_result = sensing_result + rover_pose[:2]
            elif self.sensor_type == 'lidar':
                pts = []
                for [r, th] in sensing_result:
                    if r != float('inf'):
                        ang = th + rover_pose[2]
                        pt = rover_pose[:2] + np.array([r * np.cos(ang), r * np.sin(ang)])
                        pts.append(pt)
                sensing_result = np.array(pts)

        min_cost = float("inf")
        best_u = [0.0, 0.0]
        vs = [self.nu_min, self.nu_max, self.omega_min, self.omega_max]
        vd = [v - self.nu_max_acc, v + self.nu_max, w - self.omega_max_acc, w + self.omega_max_acc]
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]
        path_primitives = []
        best_trajectory = []
        for w in np.arange(dw[2], dw[3], self.omega_delta):
            for v in np.arange(dw[0], dw[1], self.nu_delta):
                x = np.append(rover_pose, [v, w])
                trajectory = self.predict_trajectory(x, dt, sensing_result)
                if len(trajectory) == 0:
                    continue

                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal_pose)
                speed_cost = self.speed_gain * (self.nu_max - trajectory[-1, 3])
                ob_cost = self.obs_cost_gain * self.calc_obstacle_cost(trajectory, sensing_result)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    best_trajectory = trajectory
                    if abs(best_u[0]) < self.rover_stuck_flag_cons and abs(x[3]) < self.rover_stuck_flag_cons:
                        # to ensure the rover do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.omega_max_acc
                path_primitives.append([x, dt])
        self.path_primitive_log.append(path_primitives)
        self.best_path_log.append(best_trajectory)
        return best_u

    def predict_trajectory(self, x_init: np.ndarray, dt: float, sensing_result: np.ndarray) -> np.ndarray:
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            pose = state_transition(x[0:3], x[3], x[4], dt)
            x = np.append(pose, [x[3], x[4]])

            if len(sensing_result) != 0:
                distance = np.linalg.norm(x[0:2] - sensing_result, axis=1)
                if np.any(distance < self.rover_r):
                    return []

            trajectory = np.vstack((trajectory, x))
            time += dt
        return trajectory

    def calc_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost

    def calc_obstacle_cost(self, trajectory, s_pts):
        if len(s_pts) == 0:
            return 0.0
        ox = s_pts[:, 0]
        oy = s_pts[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = s_pts[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= self.rover_r / 2 + 1e-2
        right_check = local_ob[:, 1] <= self.rover_r / 2 + 1e-2
        bottom_check = local_ob[:, 0] >= -self.rover_r / 2 - 1e-2
        left_check = local_ob[:, 1] >= -self.rover_r / 2 - 1e-2
        if (np.logical_and(np.logical_and(upper_check, right_check), np.logical_and(bottom_check, left_check))).any():
            return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        obstacles: list[Obstacle] = None,
        start_pos: np.ndarray = None,
        goal_pos: np.ndarray = None,
        rover_poses: list[np.ndarray] = None,
        expand_dist: float = 0.0,
        end_step=None,
        axes_setting: list = [0.09, 0.07, 0.85, 0.9]
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim, axes_setting)
        draw_obstacles(ax, obstacles, expand_dist)
        draw_start(ax, start_pos)
        draw_goal(ax, goal_pos)
        elems = []

        # Start Animation
        animation_len = end_step
        pbar = tqdm(total=animation_len)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, animation_len, interval=100, repeat=False, fargs=(ax, elems, xlim, ylim, rover_poses, pbar),
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, rover_poses, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )

        draw_history_pose(ax, elems, rover_poses, self.rover_r, "black", i)

        if i < len(self.best_path_log):
            self.draw_path_primitives(ax, elems, self.path_primitive_log[i])
            self.draw_best_path(ax, elems, self.best_path_log[i])
        pbar.update(1)

    def draw_path_primitives(self, ax: Axes, elems: list, path_primitives: list[list[np.ndarray]]) -> None:
        for path_primitive in path_primitives:
            x = path_primitive[0]
            dt = path_primitive[1]
            trajectory = self.predict_trajectory(x, dt, [])
            if len(trajectory) == 0:
                continue
            elems += ax.plot(trajectory[:, 0], trajectory[:, 1], color="cyan", alpha=0.5)

    def draw_best_path(self, ax: Axes, elems: list, best_trajectory: list[np.ndarray]) -> None:
        if len(best_trajectory) == 0:
            return
        elems += ax.plot(best_trajectory[:, 0], best_trajectory[:, 1], color="red")


class PathFollower(DwaController):
    def __init__(
        self,
        nu_range: list[float] = [-1.0, 2.0],
        omega_range: list[float] = [-120 * np.pi / 180, 120 * np.pi / 180],
        nu_delta: float = 0.1,
        omega_delta: float = 30 * np.pi / 180,
        nu_max_acc: float = 0.3,
        omega_max_acc: float = 60 * np.pi / 180,
        to_goal_cost_gain: float = 0.15,
        speed_gain: float = 1.0,
        obs_cost_gain: float = 2.0,
        predict_time: float = 1.0,
        rover_r: float = 0.0,
        rover_stuck_flag_cons: float = 0.001,
    ) -> None:
        super().__init__(
            nu_range=nu_range, omega_range=omega_range,
            nu_delta=nu_delta, omega_delta=omega_delta,
            nu_max_acc=nu_max_acc, omega_max_acc=omega_max_acc,
            to_goal_cost_gain=to_goal_cost_gain, speed_gain=speed_gain, obs_cost_gain=obs_cost_gain,
            predict_time=predict_time, rover_r=rover_r,
            rover_stuck_flag_cons=rover_stuck_flag_cons
        )

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        v: float, w: float, dt: float,
        goal_pose: np.ndarray,
        obstacles: list[Obstacle],
        *args, **kwargs
    ):
        angle_to_goal = np.arctan2(goal_pose[1] - rover_pose[1], goal_pose[0] - rover_pose[0]) - rover_pose[2]
        while angle_to_goal > np.pi:
            angle_to_goal -= 2 * np.pi
        while angle_to_goal < - np.pi:
            angle_to_goal += 2 * np.pi
        if angle_to_goal > np.pi / 4:
            return 0.0, self.omega_max
        elif angle_to_goal < -np.pi / 4:
            return 0.0, self.omega_min
        else:
            return super().calculate_control_inputs(rover_pose, v, w, dt, goal_pose, obstacles)


class ArcPathController(Controller):
    def __init__(
        self,
        v_range: list[float] = [0.0, 2.0],
        w_range: list[float] = [-2 * np.pi, 2 * np.pi],
        dv: float = 0.5,
        branch_num: int = 11,
        branch_depth: int = 5,
        to_goal_cost_gain: float = 0.15,
        speed_gain: float = 1.0,
        obs_cost_gain: float = 1.0,
        rover_r: float = 0.0,
        rover_stuck_flag_cons: float = 0.001,
    ) -> None:
        self.v_min = v_range[0]
        self.v_max = v_range[1]
        self.w_min = w_range[0]
        self.w_max = w_range[1]
        self.dv = dv

        self.branch_num = branch_num
        self.branch_depth = branch_depth

        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_gain = speed_gain
        self.obs_cost_gain = obs_cost_gain

        self.rover_stuck_flag_cons = rover_stuck_flag_cons

        self.rover_r = rover_r

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        dt: float,
        goal_pose: np.ndarray,
        mapper: GridMapper,
        *args, **kwargs
    ):
        min_cost = float("inf")
        best_u = [0.0, 0.0]

        # List up Trajectroy
        traj_list = []
        for v in np.arange(self.v_min, self.v_max + 1e-4, self.dv):
            for w in np.linspace(self.w_min, self.w_max, self.branch_num):
                is_collision = False
                if v < self.rover_stuck_flag_cons and w < self.rover_stuck_flag_cons:
                    continue
                x = np.append(rover_pose, [v, w])
                x = np.array(x)
                traj = np.array(x)
                for _ in range(self.branch_depth):
                    pose = state_transition(x[0:3], v, w, dt)
                    x = np.append(pose, [v, w])

                    # Collision Check
                    idx = mapper.poseToIndex(pose)
                    if mapper.map[idx[0], idx[1]] > 0.5:
                        is_collision = True

                    traj = np.vstack((traj, x))
                    if is_collision:
                        break
                traj_list.append(traj) if is_collision is False else None

        for trajectory in traj_list:
            # to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal_pose)
            # speed_cost = self.speed_gain * (self.v_max - trajectory[-1, 3])
            # ob_cost = self.obs_cost_gain * self.calc_obstacle_cost(trajectory, obstacle_list)
            to_goal_cost = self.to_goal_cost_gain * self.calc_dist_to_goal_cost(trajectory, goal_pose)
            # final_cost = to_goal_cost + speed_cost + ob_cost
            final_cost = to_goal_cost  # + speed_cost

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                best_u = [trajectory[-1, 3], trajectory[-1, 4]]
                # if abs(best_u[0]) < self.rover_stuck_flag_cons and abs(trajectory[-1, 3]) < self.rover_stuck_flag_cons:
                # to ensure the rover do not get stuck in
                # best v=0 m/s (in front of an obstacle) and
                # best omega=0 rad/s (heading to the goal with
                # angle difference of 0)
                # best_u[1] = self.w_min
        return best_u

    def calc_dist_to_goal_cost(self, trajectory, goal):
        x, y = trajectory[-1, 0], trajectory[-1, 1]
        cost_dist = np.linalg.norm(trajectory[-1, 0:2] - goal[0:2])
        return cost_dist

    def calc_angle_to_goal_cost(self, trajectory, goal):
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        error_angle = math.atan2(dy, dx)
        cost_angle = error_angle - trajectory[-1, 2]
        cost = abs(math.atan2(math.sin(cost_angle), math.cos(cost_angle)))
        return cost

    def calc_obstacle_cost(self, trajectory, ob):
        if len(ob) == 0:
            return 0.0
        ox = ob[:, 0]
        oy = ob[:, 1]
        dx = trajectory[:, 0] - ox[:, None]
        dy = trajectory[:, 1] - oy[:, None]
        r = np.hypot(dx, dy)

        yaw = trajectory[:, 2]
        rot = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        rot = np.transpose(rot, [2, 0, 1])
        local_ob = ob[:, None] - trajectory[:, 0:2]
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        local_ob = np.array([local_ob @ x for x in rot])
        local_ob = local_ob.reshape(-1, local_ob.shape[-1])
        upper_check = local_ob[:, 0] <= self.rover_r / 2
        right_check = local_ob[:, 1] <= self.rover_r / 2
        bottom_check = local_ob[:, 0] >= -self.rover_r / 2
        left_check = local_ob[:, 1] >= -self.rover_r / 2
        if (np.logical_and(np.logical_and(upper_check, right_check), np.logical_and(bottom_check, left_check))).any():
            return float("Inf")

        min_r = np.min(r)
        return 1.0 / min_r  # OK
