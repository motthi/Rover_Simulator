import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List
from scipy.spatial import cKDTree
from rover_simulator.core import Mapper, Sensor, Obstacle
from rover_simulator.utils import isInRange, angle_to_range, isInList, round_off


class GridMapper(Mapper):
    def __init__(
        self,
        grid_size: np.ndarray = np.array([20, 20]),
        grid_width: float = 0.1,
        sensor: Sensor = None,
        know_obstacles=[],
        rover_r: float = 0.0,
        expand_rate: float = 1.0
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_num = (self.grid_size / self.grid_width).astype(np.int32)

        self.sensor = sensor
        self.rover_r = rover_r
        self.expand_rate = expand_rate

        self.map = np.full(self.grid_num, 0.5)

        self.obstacles_table = []
        self.obstacle_kdTree = None
        self.observed_grids = []
        self.retain_range = float('inf')

        for obstacle in know_obstacles:
            self.update_circle(obstacle.pos, (obstacle.r + rover_r) * expand_rate, 1.0)

    def reset(self) -> None:
        self.map = np.full(self.grid_num, 0.5)
        self.obstacles_table = []
        self.obstacle_kdTree = None
        self.observed_grids = []

    def update(self, rover_estimated_pose: np.ndarray, sensed_obstacles: List[Dict]) -> None:
        # Initailize occupancy of grid in sensing range to 0
        rover_idx = self.poseToIndex(rover_estimated_pose)
        sensed_grids = []
        ang_range_min = angle_to_range(rover_estimated_pose[2] - self.sensor.fov / 2)
        ang_range_max = angle_to_range(rover_estimated_pose[2] + self.sensor.fov / 2)
        sensing_range = self.sensor.range / self.grid_width
        for i in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
            for j in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                if np.sqrt(i**2 + j**2) > sensing_range + 1e-5:
                    continue
                u = rover_idx + np.array([i, j])
                if self.isOutOfBounds(u):
                    continue
                if isInRange(np.arctan2(j, i), ang_range_min, ang_range_max):
                    sensed_grids.append(u)
                    if self.map[u[0]][u[1]] <= 0.5:
                        self.map[u[0]][u[1]] = 0.01

        # List up new obstacles
        new_obstacles = []
        for sensed_obstacle in sensed_obstacles:
            distance = sensed_obstacle['distance']
            angle = sensed_obstacle['angle'] + rover_estimated_pose[2]
            radius = sensed_obstacle['radius']
            obstacle_pos = rover_estimated_pose[0:2] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            new_obstacles.append([obstacle_pos, radius])

        # List up deleted obstacles in sensing range
        deleted_obstacles = []
        if self.obstacle_kdTree is not None:
            idxes = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.sensor.range)
            for idx in idxes:
                obstacle_pos = self.obstacles_table[idx].pos
                angle = angle_to_range(
                    np.arctan2(
                        obstacle_pos[1] - rover_estimated_pose[1],
                        obstacle_pos[0] - rover_estimated_pose[0]
                    ) - rover_estimated_pose[2]
                )
                if isInRange(angle, -self.sensor.fov / 2, self.sensor.fov / 2):
                    deleted_obstacles.append(idx)   # センシング範囲内の過去の障害物を一旦削除する
        # Delete obstacle list from obstacle_table
        for idx in sorted(deleted_obstacles, reverse=True):
            _ = self.update_circle(self.obstacles_table[idx].pos, (self.obstacles_table[idx].r + self.rover_r) * self.expand_rate, 0.01)
            self.obstacles_table.pop(idx)

        # List up all obstacles
        updated_grids = []
        for pos, radius in new_obstacles:
            updated_grids += self.update_circle(pos, (radius + self.rover_r) * self.expand_rate, 0.99)
            self.obstacles_table.append(Obstacle(pos, radius))

        # List up observed grids
        self.observed_grids = []
        for u in sensed_grids:
            if self.map[u[0]][u[1]] > 0.5:
                self.observed_grids.append([u, 0.99])
            else:
                self.observed_grids.append([u, 0.01])
        for u, occ in updated_grids:
            self.observed_grids.append([u, occ]) if not isInList(u, [v[0] for v in self.observed_grids]) else None

        # Create obstacle's KD Tree
        if not len(self.obstacles_table) == 0:
            obstacle_positions = [obstacle.pos[0:2] for obstacle in self.obstacles_table]
            self.obstacle_kdTree = cKDTree(obstacle_positions)
        else:
            self.obstacle_kdTree = None

    def update_circle(self, pos, r, occupancy):
        updated_grids = []
        range_minus = -np.arange(self.grid_width, r + 1e-5, self.grid_width)
        range_plus = np.arange(0, r + 1e-5, self.grid_width)
        lattice_range = np.append(range_minus, range_plus)
        for lattice_x in lattice_range:
            for lattice_y in lattice_range:
                if np.linalg.norm(np.array([lattice_x, lattice_y])) > r:
                    continue
                lattice_pos = np.array([lattice_x, lattice_y]) + pos
                u = self.poseToIndex(lattice_pos)
                if self.isOutOfBounds(u):
                    continue
                if not isInList(u, [grids[0] for grids in updated_grids]):
                    self.map[u[0]][u[1]] = occupancy
                    updated_grids.append([u, occupancy])
        return updated_grids

    def poseToIndex(self, pose: np.ndarray) -> np.ndarray:
        return round_off(np.array(pose[0:2]) / self.grid_width).astype('int32')

    def indexToPose(self, idx):
        return np.append(idx * self.grid_width, 0.0)

    def isOutOfBounds(self, idx: np.ndarray) -> bool:
        # if np.any(idx >= self.grid_num) or np.any(idx < [0, 0]):
        #     return True
        # else:
        #     return False
        if idx[0] >= self.grid_num[0]:
            return True
        elif idx[0] < 0:
            return True
        if idx[1] >= self.grid_num[1]:
            return True
        elif idx[1] < 0:
            return True
        return False

    def draw_map(
        self,
        xlim: List[float], ylim: List[float],
        figsize=(8, 8),
        obstacles: List[Obstacle] = [],
        enlarge_obstacle: float = 0.0,
        map_name='map'
    ) -> None:
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        if map_name == 'map':
            # Draw Obstacles
            for obstacle in obstacles:
                enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray', zorder=-1.0)
                ax.add_patch(enl_obs)
            for obstacle in obstacles:
                obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black', zorder=-1.0)
                ax.add_patch(obs)

            im = ax.imshow(
                cv2.rotate(self.map, cv2.ROTATE_90_COUNTERCLOCKWISE),
                cmap="Greys",
                alpha=0.5,
                extent=(
                    -self.grid_width / 2,
                    self.grid_width * self.grid_num[0] - self.grid_width / 2,
                    -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
                ),
                zorder=1.0
            )
            plt.colorbar(im)
        elif map_name == 'table':
            for obstacle in obstacles:
                enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray', alpha=0.3)
                ax.add_patch(enl_obs)
            for obstacle in obstacles:
                obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black', alpha=0.3)
                ax.add_patch(obs)
            for obstacle in self.obstacles_table:
                enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray')
                ax.add_patch(enl_obs)
            for obstacle in self.obstacles_table:
                obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
                ax.add_patch(obs)


class TableMapper(Mapper):
    def __init__(
        self,
        sensor: Sensor = None,
        know_obstacles=[],
        rover_r: float = 0.0,
        retain_range: float = 10.0
    ) -> None:
        self.sensor = sensor
        self.rover_r = rover_r
        self.retain_range = retain_range
        self.obstacle_kdTree = None
        self.obstacles_table = []
        for obstacle in know_obstacles:
            self.obstacles_table.append(obstacle)

    def reset(self) -> None:
        self.obstacles_table = []
        self.obstacle_kdTree = None

    def update(self, rover_estimated_pose: np.ndarray, sensed_obstacles: List[Dict]) -> None:
        # List up obstacles in sensing range
        obstacle_in_sense_idxes = []
        if self.obstacle_kdTree is not None:
            idxes = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.sensor.range)
            for idx in idxes:
                obstacle_pos = self.obstacles_table[idx].pos
                angle = angle_to_range(
                    np.arctan2(
                        obstacle_pos[1] - rover_estimated_pose[1],
                        obstacle_pos[0] - rover_estimated_pose[0]
                    ) - rover_estimated_pose[2]
                )
                if isInRange(angle, -self.sensor.fov / 2, self.sensor.fov / 2):
                    obstacle_in_sense_idxes.append(idx)

        # List up obstales which is out of retain_range
        obstacle_outofrange_idxes = []
        if self.obstacle_kdTree is not None:
            idxes = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.retain_range)
            obstacle_outofrange_idxes = [i for i in range(len(self.obstacles_table)) if i not in idxes]

        # Delete Obstacles
        for idx in sorted(set(obstacle_in_sense_idxes + obstacle_outofrange_idxes), reverse=True):
            self.obstacles_table.pop(idx)

        # Append new obstacles
        for sensed_obstacle in sensed_obstacles:
            distance = sensed_obstacle['distance']
            angle = sensed_obstacle['angle'] + rover_estimated_pose[2]
            radius = sensed_obstacle['radius']
            obstacle_pos = rover_estimated_pose[0:2] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            self.obstacles_table.append(Obstacle(obstacle_pos, radius))

        # Create obstacle's KD Tree
        if not len(self.obstacles_table) == 0:
            obstacle_positions = [obstacle.pos[0:2] for obstacle in self.obstacles_table]
            self.obstacle_kdTree = cKDTree(obstacle_positions)
        else:
            self.obstacle_kdTree = None

    def draw_map(self, xlim: List[float], ylim: List[float], figsize=(8, 8)) -> None:
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        for obstacle in self.obstacles_table:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + self.rover_r, fc='gray', ec='gray')
            ax.add_patch(enl_obs)

        for obstacle in self.obstacles_table:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black')
            ax.add_patch(obs)
