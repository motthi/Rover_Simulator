import numpy as np
from scipy.spatial import cKDTree
from rover_simulator.core import Mapper, Sensor, Obstacle
from rover_simulator.utils.utils import is_angle_in_range, set_angle_into_range, is_in_list, round_off
from rover_simulator.utils.draw import draw_grid_map, set_fig_params, draw_obstacles, draw_start, draw_goal, draw_grid


class GridMapper(Mapper):
    def __init__(
        self,
        grid_size: np.ndarray = np.array([20, 20]),
        grid_width: float = 0.1,
        sensor: Sensor = None,
        known_obstacles=[],
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

        for obstacle in known_obstacles:
            self.obstacles_table.append(obstacle)
            self.update_circle(obstacle.pos, (obstacle.r + rover_r) * expand_rate, 1.0)

    def reset(self) -> None:
        self.map = np.full(self.grid_num, 0.5)
        self.obstacles_table = []
        self.obstacle_kdTree = None
        self.observed_grids = []

    def update(self, rover_estimated_pose: np.ndarray, sensed_obstacles: list[dict]) -> None:
        # Initailize occupancy of grid in sensing range to 0
        rover_idx = self.poseToIndex(rover_estimated_pose)
        sensed_grids = []
        ang_range_min = set_angle_into_range(rover_estimated_pose[2] - self.sensor.fov / 2)
        ang_range_max = set_angle_into_range(rover_estimated_pose[2] + self.sensor.fov / 2)
        sensing_range = self.sensor.range / self.grid_width
        for i in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
            for j in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                if np.sqrt(i**2 + j**2) > sensing_range + 1e-5:
                    continue
                u = rover_idx + np.array([i, j])
                if self.isOutOfBounds(u):
                    continue
                if is_angle_in_range(np.arctan2(j, i), ang_range_min, ang_range_max):
                    sensed_grids.append(u)
                    if self.map[u[0]][u[1]] <= 0.5:
                        self.map[u[0]][u[1]] = 0.01

        # list up new obstacles
        new_obstacles = []
        for sensed_obstacle in sensed_obstacles:
            distance = sensed_obstacle['distance']
            angle = sensed_obstacle['angle'] + rover_estimated_pose[2]
            radius = sensed_obstacle['radius']
            obstacle_pos = rover_estimated_pose[0:2] + np.array([distance * np.cos(angle), distance * np.sin(angle)])
            new_obstacles.append([obstacle_pos, radius])

        # list up deleted obstacles in sensing range
        deleted_obstacles = []
        if self.obstacle_kdTree is not None:
            idxes = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.sensor.range)
            for idx in idxes:
                obstacle_pos = self.obstacles_table[idx].pos
                angle = set_angle_into_range(
                    np.arctan2(
                        obstacle_pos[1] - rover_estimated_pose[1],
                        obstacle_pos[0] - rover_estimated_pose[0]
                    ) - rover_estimated_pose[2]
                )
                if is_angle_in_range(angle, -self.sensor.fov / 2, self.sensor.fov / 2):
                    deleted_obstacles.append(idx)   # センシング範囲内の過去の障害物を一旦削除する
        # Delete obstacle list from obstacle_table
        for idx in sorted(deleted_obstacles, reverse=True):
            _ = self.update_circle(self.obstacles_table[idx].pos, (self.obstacles_table[idx].r + self.rover_r) * self.expand_rate, 0.01)
            self.obstacles_table.pop(idx)

        # list up all obstacles
        updated_grids = []
        for pos, radius in new_obstacles:
            updated_grids += self.update_circle(pos, (radius + self.rover_r) * self.expand_rate, 0.99)
            self.obstacles_table.append(Obstacle(pos, radius))

        # list up observed grids
        self.observed_grids = []
        for u in sensed_grids:
            if self.map[u[0]][u[1]] > 0.5:
                self.observed_grids.append([u, 0.99])
            else:
                self.observed_grids.append([u, 0.01])
        for u, occ in updated_grids:
            self.observed_grids.append([u, occ]) if not is_in_list(u, [v[0] for v in self.observed_grids]) else None

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
                if not is_in_list(u, [grids[0] for grids in updated_grids]):
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

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize=(8, 8),
        obstacles: list[Obstacle] = [],
        enlarge_range: float = 0.0,
        map_name='map'
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        extent = (
            -self.grid_width / 2,
            self.grid_width * self.grid_num[0] - self.grid_width / 2,
            -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
        )

        if map_name == 'map':
            draw_obstacles(ax, obstacles, enlarge_range)
            draw_grid_map(ax, self.map, "Greys", 0.0, 1.0, 0.5, extent, 1.0)
        elif map_name == 'table':
            draw_obstacles(ax, obstacles, enlarge_range, 0.3)
            draw_obstacles(ax, self.obstacles_table, enlarge_range, 0.3)


class TableMapper(Mapper):
    def __init__(
        self,
        sensor: Sensor = None,
        rover_r: float = 0.0,
        known_obstacles=[],
        retain_range: float = 10.0
    ) -> None:
        self.sensor = sensor
        self.rover_r = rover_r
        self.retain_range = retain_range
        self.obstacle_kdTree = None
        self.obstacles_table = []
        for obstacle in known_obstacles:
            self.obstacles_table.append(obstacle)

    def reset(self) -> None:
        self.obstacles_table = []
        self.obstacle_kdTree = None

    def update(self, rover_estimated_pose: np.ndarray, sensed_obstacles: list[dict]) -> None:
        # list up obstacles in sensing range
        obstacle_in_sense_idxes = []
        if self.obstacle_kdTree is not None:
            idxes = self.obstacle_kdTree.query_ball_point(rover_estimated_pose[0:2], r=self.sensor.range)
            for idx in idxes:
                obstacle_pos = self.obstacles_table[idx].pos
                angle = set_angle_into_range(
                    np.arctan2(
                        obstacle_pos[1] - rover_estimated_pose[1],
                        obstacle_pos[0] - rover_estimated_pose[0]
                    ) - rover_estimated_pose[2]
                )
                if is_angle_in_range(angle, -self.sensor.fov / 2, self.sensor.fov / 2):
                    obstacle_in_sense_idxes.append(idx)

        # list up obstales which is out of retain_range
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

    def draw_map(self, xlim: list[float], ylim: list[float], figsize=(8, 8)) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, self.obstacles_table, self.rover_r, 0.3)
