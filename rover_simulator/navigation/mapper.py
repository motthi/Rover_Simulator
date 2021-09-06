import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Dict, List
from scipy.spatial import cKDTree
from rover_simulator.core import Mapper, Sensor, Obstacle
from rover_simulator.utils import isInRange, drawGrid, occupancyToColor, isInList


class GridMapper(Mapper):
    def __init__(
        self,
        grid_size: np.ndarray = np.array([20, 20]),
        grid_width: float = 0.1,
        sensor: Sensor = None,
        know_obstacles=[],
        rover_r: float = 0.0
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_num = (self.grid_size / self.grid_width).astype(np.int32)

        self.sensor = sensor
        self.rover_r = rover_r

        self.map = np.full(self.grid_num, 0.5)

        self.obstacles_table = []
        self.obstacle_kdTree = None
        self.observed_grids = []

        for obstacle in know_obstacles:
            obstacle_idx = self.poseToIndex(obstacle.pos)
            self.update_circle(obstacle_idx, obstacle.r + rover_r, 1.0)

    def reset(self) -> None:
        self.map = np.full(self.grid_num, 0.5)
        self.obstacles_table = []
        self.obstacle_kdTree = None
        self.observed_grids = []

    def update(self, rover_estimated_pose: np.ndarray, sensed_obstacles: List[Dict]) -> None:
        # Initailize occupancy of grid in sensing range to 0
        rover_idx = self.poseToIndex(rover_estimated_pose)
        sensed_grids = []
        for i in range(np.ceil(-self.sensor.range / self.grid_width).astype(np.int32), np.floor(self.sensor.range / self.grid_width).astype(np.int32) + 1):
            for j in range(np.ceil(-self.sensor.range / self.grid_width).astype(np.int32), np.floor(self.sensor.range / self.grid_width).astype(np.int32) + 1):
                if np.sqrt(i**2 + j**2) > self.sensor.range / self.grid_width + 1e-5:
                    continue
                u = rover_idx + np.array([i, j])
                if self.isOutOfBounds(u):
                    continue
                if isInRange(np.arctan2(j, i), rover_estimated_pose[2] - self.sensor.fov / 2, rover_estimated_pose[2] + self.sensor.fov / 2):
                    sensed_grids.append(u)
                    if self.map[u[0]][u[1]] <= 0.5:
                        self.map[u[0]][u[1]] = 0.0

        # List up new obstacles
        new_obstacles = []
        for sensed_obstacle in sensed_obstacles:
            distance = sensed_obstacle['distance']
            angle = sensed_obstacle['angle'] + rover_estimated_pose[2]
            radius = sensed_obstacle['radius']
            obstacle_pose = rover_estimated_pose[0:2] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            obstacle_idx = self.poseToIndex(obstacle_pose)
            new_obstacles.append([obstacle_idx, radius])

        # List up deleted obstacles in sensing range
        deleted_obstacles = []
        if self.obstacle_kdTree is not None:
            indices = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.sensor.range)
            for idx in indices:
                radius = self.obstacles_table[idx].r
                obstacle_pos = self.obstacles_table[idx].pos
                obstacle_idx = self.poseToIndex(obstacle_pos)
                angle = np.arctan2(
                    obstacle_pos[1] - rover_estimated_pose[1],
                    obstacle_pos[0] - rover_estimated_pose[0]
                ) - rover_estimated_pose[2]
                if isInRange(angle, -self.sensor.fov / 2, self.sensor.fov / 2):
                    # センシング範囲内の過去の障害物を一旦削除する
                    deleted_obstacles.append(idx)

        # Delete obstacle list from obstacle_table
        for i in sorted(deleted_obstacles, reverse=True):
            obstacle_idx = self.poseToIndex(self.obstacles_table[i].pos)
            radius = self.obstacles_table[i].r
            _ = self.update_circle(obstacle_idx, radius + self.rover_r, 0.01)
            self.obstacles_table.pop(i)

        # List up all obstacles
        updated_grids = []
        for idx, radius in new_obstacles:
            updated_grids += self.update_circle(idx, radius + self.rover_r, 0.99)
            self.obstacles_table.append(Obstacle(self.indexToPose(idx), radius))

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

    def update_circle(self, idx, r, occupancy):
        updated_grids = []
        for i in range(np.ceil(-r / self.grid_width).astype(np.int32), np.floor(r / self.grid_width).astype(np.int32) + 1):
            for j in range(np.ceil(-r / self.grid_width).astype(np.int32), np.floor(r / self.grid_width).astype(np.int32) + 1):
                if np.sqrt(i**2 + j**2) > r / self.grid_width + 1e-5:
                    continue
                u = idx + np.array([i, j])
                if self.isOutOfBounds(u):
                    continue
                self.map[u[0]][u[1]] = occupancy
                updated_grids.append([u, occupancy])
        return updated_grids

    def poseToIndex(self, pose: np.ndarray) -> np.ndarray:
        return (np.array(pose[0: 2]) // self.grid_width + np.array([self.grid_width, self.grid_width]) / 2).astype(np.int32)

    def indexToPose(self, idx):
        return np.append(idx * self.grid_width, 0.0)

    def isOutOfBounds(self, idx: np.ndarray) -> bool:
        if np.any(idx >= self.grid_num) or np.any(idx < [0, 0]):
            return True
        else:
            return False

    def drawMap(self, xlim: List[float], ylim: List[float], figsize=(8, 8)) -> None:
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        for idx, occ in np.ndenumerate(self.map):
            if not occ == 0.0:
                drawGrid(np.array(idx), self.grid_width, occupancyToColor(occ), 1.0, ax)
        # drawGrid(np.array([30, 20]), self.grid_width, "red", 1.0, ax)
        # plt.show()


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
                angle = np.arctan2(
                    obstacle_pos[1] - rover_estimated_pose[1],
                    obstacle_pos[0] - rover_estimated_pose[0]
                ) - rover_estimated_pose[2]
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

    def drawMap(self, xlim: List[float], ylim: List[float], figsize=(8, 8)) -> None:
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
