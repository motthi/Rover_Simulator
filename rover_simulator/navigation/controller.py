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
        sensing_result: np.ndarray,  # @todo Only for the sensor? No compatibility with map?
        *args, **kwargs
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
        v_range: list[float] = [-1.0, 2.0],
        w_range: list[float] = [-120 * np.pi / 180, 120 * np.pi / 180],
        v_delta: float = 0.1,
        w_delta: float = 30 * np.pi / 180,
        v_max_acc: float = 0.3,
        w_max_acc: float = 60 * np.pi / 180,
        to_goal_cost_gain: float = 0.15,
        speed_gain: float = 1.0,
        obs_cost_gain: float = 2.0,
        predict_time: float = 1.0,
        rover_r: float = 0.0,
        rover_stuck_flag_cons: float = 0.001,
    ) -> None:
        super().__init__(
            nu_range=v_range, omega_range=w_range,
            nu_delta=v_delta, omega_delta=w_delta,
            nu_max_acc=v_max_acc, omega_max_acc=w_max_acc,
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
        angle_to_goal = set_angle_into_range(angle_to_goal)
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
        branch_dt: float = 3,
        to_goal_cost_gain: float = 0.15,
        speed_gain: float = 1.0,
        obs_cost_gain: float = 1.0,
        rover_r: float = 0.0,
        stuck_flag_cons: float = 0.001,
        stuck_cnt_max: int = 100,
        dt: float = 0.1
    ) -> None:
        self.v_min = v_range[0]
        self.v_max = v_range[1]
        self.w_min = w_range[0]
        self.w_max = w_range[1]
        self.dv = dv
        self.branch_dt = branch_dt
        self.w_stuck = np.pi / 2

        self.branch_num = branch_num
        self.branch_depth = branch_depth

        self.to_goal_cost_gain = to_goal_cost_gain
        self.speed_gain = speed_gain
        self.obs_cost_gain = obs_cost_gain

        self.stuck_flag_cons = stuck_flag_cons
        self.stuck_cnt = 0
        self.stuck_cnt_max = stuck_cnt_max
        self.stuck_flag = False

        self.rover_r = rover_r

        self.path_primitives = self.generate_path_primitives(dt)

        self.is_prev_rot = False

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        dt: float,
        goal_pose: np.ndarray,
        mapper: GridMapper,
        *args, **kwargs
    ):
        from tqdm import tqdm
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        rover_idx = mapper.poseToIndex(rover_pose)
        if mapper.isOutOfBounds(rover_idx) or mapper.map[rover_idx[0], rover_idx[1]] > 0.5:
            mapper.map[rover_idx[0], rover_idx[1]] = 0.1

        traj_list = transform_traj_list(self.path_primitives, rover_pose)
        angle_to_goal = np.arctan2(goal_pose[1] - rover_pose[1], goal_pose[0] - rover_pose[0]) - rover_pose[2]
        angle_to_goal = set_angle_into_range(angle_to_goal)

        curr_dist_to_wpt = np.linalg.norm(rover_pose[:2] - goal_pose[:2])

        # if self.is_prev_rot and angle_to_goal > np.pi / 4:
        #     best_u[1] = self.w_max
        # elif self.is_prev_rot and angle_to_goal < -np.pi / 4:
        #     best_u[1] = self.w_min
        # elif angle_to_goal > 3 * np.pi / 4:
        #     best_u[1] = self.w_max
        #     self.is_prev_rot = True
        # elif angle_to_goal < -3 * np.pi / 4:
        #     best_u[1] = self.w_min
        #     self.is_prev_rot = True
        if True:
            self.is_prev_rot = False
            for traj in traj_list:
                is_collision = False
                for [x, y, _, _, _] in traj:
                    # Collision Check
                    idx = mapper.poseToIndex(np.array([x, y]))
                    if mapper.isOutOfBounds(idx) or mapper.map[idx[0], idx[1]] > 0.5:
                        is_collision = True
                        break

                dist_to_wpt = np.linalg.norm(traj[-1, :2] - goal_pose[:2])
                if curr_dist_to_wpt < dist_to_wpt:
                    continue

                if is_collision:
                    continue

                rw_cost = 5 * np.abs(traj[0, 4]) if traj[0, 3] < 1e-3 else 0.0  # it is better to select the path which does not excute rotation at first.

                # speed_cost = self.speed_gain * (self.v_max - traj[-1, 3])
                # ob_cost = self.obs_cost_gain * self.calc_obstacle_cost(trajectory, obstacle_list)
                to_goal_cost = self.to_goal_cost_gain * self.calc_dist_to_goal_cost(traj, goal_pose)
                # final_cost = to_goal_cost + speed_cost + ob_cost
                final_cost = to_goal_cost + rw_cost  # + speed_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [traj[0, 3], traj[0, 4]]
                    # if abs(best_u[0]) < self.rover_stuck_flag_cons and abs(trajectory[-1, 3]) < self.rover_stuck_flag_cons:
                    # to ensure the rover do not get stuck in
                    # best v=0 m/s (in front of an obstacle) and
                    # best omega=0 rad/s (heading to the goal with
                    # angle difference of 0)
                    # best_u[1] = self.w_min

            if best_u[0] < self.stuck_flag_cons and best_u[1] < self.stuck_flag_cons:
                idx = mapper.poseToIndex(rover_pose)
                mapper.isOutOfBounds(idx)
                # tqdm.write(f"No path found\t{mapper.map[idx[0], idx[1]] > 0.5}")
                best_u[1] = self.w_max
                # if angle_to_goal >= 0:
                #     best_u[1] = self.w_max
                # elif angle_to_goal < 0:
                #     best_u[1] = self.w_min

        if best_u[0] < self.stuck_flag_cons:
            self.stuck_cnt += 1
            if self.stuck_cnt > self.stuck_cnt_max:
                self.stuck_flag = True
        else:
            self.stuck_cnt = 0
            self.stuck_flag = False

        # tqdm.write(f"{self.stuck_cnt}, {self.stuck_cnt_max}")
        # dist_to_goal = np.linalg.norm(rover_pose[:2] - goal_pose[:2])
        # tqdm.write(f"{best_u=}    \t{np.rad2deg(angle_to_goal)=}")
        return best_u

    def calc_dist_to_goal_cost(self, trajectory, goal):
        # x, y = trajectory[-1, 0], trajectory[-1, 1]
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

    def draw_path_primitives(self, dt: float, src: str = None) -> None:
        traj_list = self.generate_path_primitives(dt)
        traj_list = transform_traj_list(traj_list, np.array([0, 0, np.pi / 2]))

        fig, ax = plt.subplots()
        for traj in traj_list:
            ax.plot(traj[:, 0], traj[:, 1])
        ax.set_aspect('equal')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        if src:
            fig.tight_layout()
            fig.savefig(src, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
        else:
            plt.show()

    def generate_path_primitives(self, dt):
        def w_lists(depth, max_depth, dt):
            w_values = np.linspace(self.w_min, self.w_max, self.branch_num)
            scaled_space = np.sign(w_values) * np.abs(w_values)**2
            scaled_space = (scaled_space - scaled_space.min()) / (scaled_space.max() - scaled_space.min())
            w_values = scaled_space * (self.w_max - self.w_min) + self.w_min

            if depth == max_depth:
                return [[]]

            w_list = []
            for w in w_values:
                sub_w_lists = w_lists(depth + 1, max_depth, dt)
                for sub_list in sub_w_lists:
                    w_list.append([w] + sub_list)
            return w_list
        rover_pose = np.array([0, 0, 0])
        traj_list = []

        v_values = np.arange(self.v_min, self.v_max + 1e-4, self.dv)
        w_list = w_lists(0, self.branch_depth, dt)
        for v in v_values:
            for w_sequence in w_list:
                traj = np.array([np.append(rover_pose, [v, w_sequence[0]])])
                current_pose = rover_pose
                for depth in range(self.branch_depth):
                    v_ = v
                    w = w_sequence[depth]
                    if v < 1e-3 and depth > 0:
                        v_ = self.v_max / 2
                    new_pose = current_pose
                    for i in range(int(self.branch_dt / dt)):
                        pose = state_transition(new_pose[:3], v_, w, dt)
                        new_pose = np.append(pose, [v, w])
                        traj = np.vstack((traj, new_pose))
                    current_pose = new_pose
                traj_list.append(traj)
        return np.array(traj_list)


def transform_traj_list(path_primitives, pose):
    traj_list = path_primitives.copy()
    tho = pose[2]
    rot = np.array([[np.cos(tho), -np.sin(tho)], [np.sin(tho), np.cos(tho)]])
    xy_rotated = rot @ traj_list[:, :, 0:2].reshape(-1, 2).T
    traj_list[:, :, :2] = xy_rotated.T.reshape(traj_list.shape[0], traj_list.shape[1], 2)
    traj_list[:, :, :2] += pose[:2]
    traj_list[:, :, 2] = traj_list[:, :, 2] + tho
    return traj_list
