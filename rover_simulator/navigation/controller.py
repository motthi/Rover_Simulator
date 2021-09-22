import math
import numpy as np
from typing import List
from scipy.spatial import cKDTree
from rover_simulator.core import Controller, Obstacle
from rover_simulator.utils import angle_to_range, state_transition


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
        self.L = 1.0

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        goal_pose: np.ndarray
    ):
        theta = np.arctan2(goal_pose[1] - rover_pose[1], goal_pose[0] - rover_pose[0]) - rover_pose[2]
        theta = angle_to_range(theta)
        w = 2 * self.v * np.sin(theta) / self.L
        return self.v, w


class DWAController(Controller):
    def __init__(
        self,
        nu_range: List[float] = [-1.0, 2.0],
        omega_range: List[float] = [-120 * np.pi / 180, 120 * np.pi / 180],
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

    def calculate_control_inputs(
        self,
        rover_pose: np.ndarray,
        v: float, w: float, dt: float,
        goal_pose: np.ndarray,
        obstacles: List[Obstacle]
    ):
        min_cost = float("inf")
        best_u = [0.0, 0.0]
        obstacle_list = np.array([[obstacle.pos[0], obstacle.pos[1]] for obstacle in obstacles])
        vs = [self.nu_min, self.nu_max, self.omega_min, self.omega_max]
        vd = [v - self.nu_max_acc, v + self.nu_max, w - self.omega_max_acc, w + self.omega_max_acc]
        dw = [max(vs[0], vd[0]), min(vs[1], vd[1]), max(vs[2], vd[2]), min(vs[3], vd[3])]
        # self.obstacles = obstacles
        # if len(self.obstacles) == 0:
        #     obstacles_kdTree = None
        # else:
        #     obstacles_kdTree = cKDTree([obstacle.pos[0:2] for obstacle in obstacles])
        for w in np.arange(dw[2], dw[3], self.omega_delta):
            for v in np.arange(dw[0], dw[1], self.nu_delta):
                x = np.append(rover_pose, [v, w])
                trajectory = self.predict_trajectory(x, dt)
                # if len(trajectory) == 0:
                #     continue
                to_goal_cost = self.to_goal_cost_gain * self.calc_to_goal_cost(trajectory, goal_pose)
                speed_cost = self.speed_gain * (self.nu_max - trajectory[-1, 3])
                ob_cost = self.obs_cost_gain * self.calc_obstacle_cost(trajectory, obstacle_list)

                final_cost = to_goal_cost + speed_cost + ob_cost

                # search minimum trajectory
                if min_cost >= final_cost:
                    min_cost = final_cost
                    best_u = [v, w]
                    # best_trajectory = trajectory
                    if abs(best_u[0]) < self.rover_stuck_flag_cons and abs(x[3]) < self.rover_stuck_flag_cons:
                        # to ensure the rover do not get stuck in
                        # best v=0 m/s (in front of an obstacle) and
                        # best omega=0 rad/s (heading to the goal with
                        # angle difference of 0)
                        best_u[1] = -self.omega_max_acc
        return best_u

    def predict_trajectory(self, x_init: np.ndarray, dt: float) -> np.ndarray:
        x = np.array(x_init)
        trajectory = np.array(x)
        time = 0
        while time <= self.predict_time:
            pose = state_transition(x[0:3], [x[3], x[4]], dt)
            x = np.append(pose, [x[3], x[4]])
            # if obstacles_kdTree is not None:
            #     indices = obstacles_kdTree.query_ball_point(x[0:2], r=4.0)
            #     for idx in indices:
            #         pos = self.obstacles[idx].pos
            #         distance = np.linalg.norm(x[0:2] - pos[0:2])
            #         if distance < self.rover_r + self.obstacles[idx].r:
            #             return []
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


class PathFollower(DWAController):
    def __init__(
        self,
        nu_range: List[float] = [-1.0, 2.0],
        omega_range: List[float] = [-120 * np.pi / 180, 120 * np.pi / 180],
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
        obstacles: List[Obstacle],
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
        v_range: List[float] = [0.0, 2.0],
        w_range: List[float] = [-2 * np.pi, 2 * np.pi],
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
        obstacles: List[Obstacle],
        *args, **kwargs
    ):
        min_cost = float("inf")
        best_u = [0.0, 0.0]

        self.obstacles = obstacles
        if len(self.obstacles) == 0:
            obstacles_kdTree = None
        else:
            obstacles_kdTree = cKDTree([obstacle.pos[0:2] for obstacle in obstacles])

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
                    pose = state_transition(x[0:3], [v, w], dt)
                    x = np.append(pose, [v, w])

                    # Collision Check
                    if obstacles_kdTree is not None:
                        idxes = obstacles_kdTree.query_ball_point(x[0:2], 3.0)
                        for idx in idxes:
                            obs_pos = self.obstacles[idx].pos
                            obs_r = self.obstacles[idx].r
                            dist = np.linalg.norm(obs_pos - x[0:2])
                            if dist < obs_r + self.rover_r:
                                is_collision = True
                                break
                    traj = np.vstack((traj, x))
                    if is_collision is True:
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
