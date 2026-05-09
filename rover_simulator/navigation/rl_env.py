import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rover_simulator.world import World
from rover_simulator.sensor import ImaginalLiDAR
from rover_simulator.history import SimpleHistory
from rover_simulator.navigation.localizer import ImaginalLocalizer
from rover_simulator.rover import BasicRover
from rover_simulator.core import Controller, CircularObstacle


class ActionController(Controller):
    def __init__(self, default_action=(0.0, 0.0)):
        super().__init__()
        self.next_action = np.array(default_action, dtype=float)

    def calculate_control_inputs(self):
        return float(self.next_action[0]), float(self.next_action[1])


class RoverGymEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, map_file: str = None, time_interval: float = 0.1, max_steps: int = 200, lidar_range: float = 10.0, d_ang: float = np.pi/360, rover_r: float = 0.5, goal_range: float = 1.0, random_obstacles: bool = True, num_obstacles: int = 10):
        super().__init__()
        self.map_file = map_file
        self.time_interval = time_interval
        self.max_steps = max_steps
        self.rover_r = rover_r
        self.goal_range = goal_range

        # world and placeholders; actual rover is created in reset
        self.world = World(time_interval=self.time_interval)
        if self.map_file:
            self.world.read_objects(self.map_file)

        # create a sensor template (will be re-created in reset to link obstacles)
        self.lidar_range = lidar_range
        self.d_ang = d_ang

        # action: [v, w]
        self.v_limit = [0.0, 2.0]
        self.w_limit = [-math.pi, math.pi]
        self.action_space = spaces.Box(low=np.array([self.v_limit[0], self.w_limit[0]], dtype=np.float32), high=np.array([self.v_limit[1], self.w_limit[1]], dtype=np.float32), dtype=np.float32)

        # observation space: create a temporary sensor to infer observation length
        tmp_sensor = ImaginalLiDAR(range=self.lidar_range, d_ang=self.d_ang, obstacles=[])
        smp_num = tmp_sensor.smp_num
        obs_len = smp_num + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)

        self._step_count = 0
        self._goal = None
        self.rover = None
        # episode result buffer for callbacks
        self._episode_results = []
        # random obstacle settings
        self.random_obstacles = random_obstacles
        self.num_obstacles = num_obstacles
        self.obstacle_bounds = ([0, 20], [0, 20])
        self.obstacle_r_range = (0.3, 1.5)

    def _build_obs_space(self):
        smp_num = self.rover.sensor.smp_num
        obs_len = smp_num + 3
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_len,), dtype=np.float32)

    def _get_obs(self):
        # use rover.sensor to sense (uses rover.real_pose internally)
        lidar = self.rover.sensor.sense(self.rover)
        if lidar.size == 0:
            dists = np.ones(self.rover.sensor.smp_num) * self.rover.sensor.range
        else:
            if lidar.ndim == 1:
                lidar = lidar.reshape(1, -1)
            dists = np.array([p[0] if p[0] != float('inf') else self.rover.sensor.range for p in lidar])
            if len(dists) < self.rover.sensor.smp_num:
                pad = np.ones(self.rover.sensor.smp_num - len(dists)) * self.rover.sensor.range
                dists = np.concatenate([dists, pad])
            elif len(dists) > self.rover.sensor.smp_num:
                dists = dists[: self.rover.sensor.smp_num]

        dists = np.clip(dists, 0.0, self.rover.sensor.range) / float(self.rover.sensor.range)

        # goal relative in robot frame (use estimated_pose)
        est = self.rover.estimated_pose
        dx = self._goal[0] - est[0]
        dy = self._goal[1] - est[1]
        th = -est[2]
        gx = math.cos(th) * dx - math.sin(th) * dy
        gy = math.sin(th) * dx + math.cos(th) * dy
        gx_n = np.clip(gx / 20.0, -1.0, 1.0)
        gy_n = np.clip(gy / 20.0, -1.0, 1.0)
        heading = math.sin(est[2])

        obs = np.concatenate([dists.astype(np.float32), np.array([gx_n, gy_n, heading], dtype=np.float32)])
        return obs

    def reset(self, seed: int | None = None, return_info: bool = False, options: dict | None = None, min_dist: float = 0.0, max_dist: float = 15.0):
        # Accept `options` and `return_info` for compatibility with Gym/Gymnasium/shimmy wrappers
        if seed is not None:
            np.random.seed(seed)
        self._step_count = 0
        # reset world
        self.world.reset()
        # populate obstacles: random per-episode if enabled, else load map_file if provided
        if self.random_obstacles:
            # create random circular obstacles
            xmin, xmax = self.obstacle_bounds[0]
            ymin, ymax = self.obstacle_bounds[1]
            rmin, rmax = self.obstacle_r_range
            for _ in range(self.num_obstacles):
                pos = np.array([np.random.uniform(xmin, xmax), np.random.uniform(ymin, ymax)])
                r = float(np.random.uniform(rmin, rmax))
                self.world.append_obstacle(CircularObstacle(pos, r))
        elif self.map_file:
            self.world.read_objects(self.map_file)

        # sample start/goal
        start, goal = self.world.set_start_goal_ramdomly(min_dist=min_dist, max_dist=max_dist)
        angle = np.random.uniform(-math.pi, math.pi)

        # create sensor linked to world obstacles
        sensor = ImaginalLiDAR(range=self.lidar_range, d_ang=self.d_ang, obstacles=self.world.obstacles)
        history = SimpleHistory(sensor=sensor, rover_r=self.rover_r)
        localizer = ImaginalLocalizer()
        controller = ActionController()

        rover = BasicRover(np.array([start[0], start[1], angle]), self.rover_r, sensor=sensor, localizer=localizer, controller=controller, history=history)
        self.world.append_rover(rover)
        self.rover = rover
        self._goal = np.array([goal[0], goal[1], 0.0])

        # build observation space
        self._build_obs_space()

        obs = self._get_obs()
        # Always return (obs, info) for compatibility with gym/gymnasium wrappers
        return obs, {}

    def step(self, action):
        action = np.array(action, dtype=float)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # set action to rover's controller
        if isinstance(self.rover.controller, ActionController):
            self.rover.controller.next_action = action

        # previous distance (estimated)
        prev_dist = np.linalg.norm(self._goal[:2] - self.rover.estimated_pose[:2])

        # advance world (runs rover.one_step internally)
        self.world.one_step()

        # observation
        obs = self._get_obs()

        # reward: progress toward goal (estimated pose)
        new_dist = np.linalg.norm(self._goal[:2] - self.rover.estimated_pose[:2])
        reward = (prev_dist - new_dist) * 10.0
        reward -= 0.01

        done = False
        info = {}
        self._step_count += 1

        # collision detection via last sensing result
        if len(self.rover.history.sensing_results) > 0:
            last = self.rover.history.sensing_results[-1]
            if hasattr(last, 'size') and last.size != 0:
                dists = np.array([p[0] if p[0] != float('inf') else self.rover.sensor.range for p in last])
                if np.any(dists < self.rover.r + 1e-3):
                    reward -= 50.0
                    done = True
                    info['collision'] = True

        terminated = False
        truncated = False
        if new_dist < self.goal_range:
            reward += 100.0
            terminated = True
            info['success'] = True

        # collision
        if 'collision' in info and info['collision']:
            terminated = True

        # timeout / max steps
        if self._step_count >= self.max_steps:
            truncated = True

        # Return Gymnasium-compatible 5-tuple: obs, reward, terminated, truncated, info
        # record episode result for external callbacks/logging
        if terminated or truncated:
            self._episode_results.append({'success': bool(info.get('success', False)), 'collision': bool(info.get('collision', False))})
        return obs, float(reward), bool(terminated), bool(truncated), info

    def pop_episode_results(self):
        """Return and clear stored episode results for this env instance.
        Used by VecEnv.env_method from callbacks to aggregate success/collision rates.
        """
        res = list(self._episode_results)
        self._episode_results = []
        return res

    def render(self, mode='human'):
        if self.rover:
            print(f"Pose: {self.rover.real_pose}, Goal: {self._goal}")

    def close(self):
        return None

