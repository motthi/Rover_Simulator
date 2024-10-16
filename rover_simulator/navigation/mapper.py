import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree
from rover_simulator.core import Mapper, Sensor, Obstacle
from rover_simulator.utils.utils import is_angle_in_range, set_angle_into_range, is_in_list
from rover_simulator.utils.draw import draw_grid_map, set_fig_params, draw_obstacles
from rover_simulator.sensor import generate_points_in_circle


def bresenham(pt_s, pt_e):
    x1, y1 = pt_s
    x2, y2 = pt_e
    dx = x2 - x1
    dy = y2 - y1
    is_steep = abs(dy) > abs(dx)
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    y_step = 1 if y1 < y2 else -1

    y_temp = y1
    pts = []
    for x in range(x1, x2 + 1):
        coord = [y_temp, x] if is_steep else (x, y_temp)
        pts.append(coord)
        error -= abs(dy)
        if error < 0:
            y_temp += y_step
            error += dx

    if swapped:
        pts.reverse()
    return np.array(pts)


class GridMapper(Mapper):
    def __init__(
        self,
        grid_size: np.ndarray = np.array([20, 20]),
        grid_width: float = 0.1,
        sensor: Sensor = None,
        known_obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
    ) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.grid_width = grid_width
        self.grid_num = (self.grid_size / self.grid_width).astype(np.int32)

        self.sensor = sensor
        self.expand_dist = expand_dist

        self.map = np.full(self.grid_num, 0.5)
        self.observed_grids = []

        for obstacle in known_obstacles:
            if obstacle.type == 'circular':
                pts = generate_points_in_circle(obstacle.r)
                for pt in pts:
                    self.update_circle(pt + obstacle.pos, expand_dist, 1.0)
                # self.update_circle(obstacle.pos, obstacle.r + expand_dist, 1.0)
            elif obstacle.type == 'rectangular':
                self.update_rectangle(obstacle.xy, obstacle.w, obstacle.h, obstacle.angle, 1.0)

        self.sensing_grid_candidates = []
        self.enlarged_sensing_grid_candidates = []
        if sensor:
            sensing_range = self.sensor.range / self.grid_width
            for i in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                for j in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                    if np.sqrt(i**2 + j**2) > sensing_range + 1e-5:
                        continue
                    self.sensing_grid_candidates.append([i, j])

            sensing_range = (self.sensor.range + expand_dist) / self.grid_width
            for i in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                for j in range(np.ceil(-sensing_range).astype(np.int32), np.floor(sensing_range).astype(np.int32) + 1):
                    if np.sqrt(i**2 + j**2) > sensing_range + 1e-5:
                        continue
                    self.enlarged_sensing_grid_candidates.append([i, j])

    def reset(self) -> None:
        self.map = np.full(self.grid_num, 0.5)
        self.obstacles_table = []
        self.observed_grids = []

    def update(self, rover_estimated_pose: np.ndarray, sensing_results) -> None:
        rover_idx = self.poseToIndex(rover_estimated_pose)
        if self.sensor.type == 'stereo_camera':
            map_temp = np.copy(self.map)

            ang_range_min = set_angle_into_range(rover_estimated_pose[2] - self.sensor.fov / 2)
            ang_range_max = set_angle_into_range(rover_estimated_pose[2] + self.sensor.fov / 2)

            # Initailize occupancy of grid in sensing range to 0
            for [i, j] in self.sensing_grid_candidates:
                if is_angle_in_range(np.arctan2(j, i), ang_range_min, ang_range_max):
                    u = rover_idx + np.array([i, j])
                    if self.isOutOfBounds(u):
                        continue
                    if self.map[u[0]][u[1]] <= 0.5:
                        self.map[u[0]][u[1]] = 0.01

            update_idxes = []
            update_idxes = set()
            for pt in sensing_results:
                idx = self.poseToIndex(pt + rover_estimated_pose[:2])
                idx_tuple = tuple(idx)
                if idx_tuple in update_idxes:
                    continue
                update_idxes.add(idx_tuple)
                self.update_circle(pt + rover_estimated_pose[:2], self.expand_dist, 0.95)

            # list up observed grids
            self.observed_grids = []
            for [i, j] in self.enlarged_sensing_grid_candidates:
                # if is_angle_in_range(np.arctan2(j, i), ang_range_min, ang_range_max):
                u = rover_idx + np.array([i, j])
                if self.isOutOfBounds(u):
                    continue
                if map_temp[u[0]][u[1]] != self.map[u[0]][u[1]]:
                    self.observed_grids.append([u, self.map[u[0]][u[1]]])

            # for u in sensed_grids:
            #     if self.map[u[0]][u[1]] > 0.5:
            #         self.observed_grids.append([u, 0.99])
            #     else:
            #         self.observed_grids.append([u, 0.01])
        elif self.sensor.type == 'lidar':
            for [r, th] in sensing_results:
                if r > self.sensor.range:
                    dist = self.sensor.range
                else:
                    dist = r
                dist -= self.expand_dist

                ang = th + rover_estimated_pose[2]
                pt = rover_estimated_pose[:2] + np.array([dist * np.cos(ang), dist * np.sin(ang)])
                hit_idx = self.poseToIndex(pt)
                beam = bresenham(rover_idx, hit_idx)
                for u in beam:
                    if self.isOutOfBounds(u):
                        continue
                    self.map[u[0]][u[1]] = 0.05
                    self.observed_grids.append([u, 0.05])

                if r <= self.sensor.range:
                    if not self.isOutOfBounds(hit_idx):
                        self.map[hit_idx[0]][hit_idx[1]] = 0.95
                        self.observed_grids.append([hit_idx, 0.95])

                    if not self.isOutOfBounds(hit_idx + np.array([1, 0])):
                        self.map[hit_idx[0] + 1][hit_idx[1]] = 0.95
                        self.observed_grids.append([hit_idx + np.array([1, 0]), 0.95])

                    if not self.isOutOfBounds(hit_idx + np.array([0, 1])):
                        self.map[hit_idx[0]][hit_idx[1] + 1] = 0.95
                        self.observed_grids.append([hit_idx + np.array([0, 1]), 0.95])

                    if not self.isOutOfBounds(hit_idx + np.array([1, 1])):
                        self.map[hit_idx[0] + 1][hit_idx[1] + 1] = 0.95
                        self.observed_grids.append([hit_idx + np.array([1, 1]), 0.95])

    def update_circle(self, pos: np.ndarray, r: float, occupancy: float):
        updated_grids = []
        updated_set = set()
        grid_width = self.grid_width
        pos_x, pos_y = pos
        r2 = r * r

        lattice_range = np.linspace(-r, r, int(2 * r / grid_width) + 2)

        for lattice_x in lattice_range:
            for lattice_y in lattice_range:
                dist2 = (np.fabs(lattice_x) - grid_width / 2) ** 2 + (np.fabs(lattice_y) - grid_width / 2) ** 2
                if dist2 > r2:
                    continue  # 円の外なら無視
                lattice_pos_x = lattice_x + pos_x
                lattice_pos_y = lattice_y + pos_y
                u = self.poseToIndex((lattice_pos_x, lattice_pos_y))
                if self.isOutOfBounds(u):
                    continue
                u_tuple = tuple(u)
                if u_tuple not in updated_set:
                    self.map[u[0]][u[1]] = occupancy
                    updated_grids.append([u, occupancy])
                    updated_set.add(u_tuple)
        return updated_grids

    def update_rectangle(self, xy, width, height, angle, occupancy):
        angle = np.deg2rad(angle)
        updated_grids = []
        range_x = np.arange(-self.expand_dist, self.expand_dist + width + 1e-5, self.grid_width * 0.5)
        range_y = np.arange(-self.expand_dist, self.expand_dist + height + 1e-5, self.grid_width * 0.5)
        for lattice_x in range_x:
            for lattice_y in range_y:
                lattice_pos = np.array([lattice_x, lattice_y])
                rot_mat = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                lattice_pos = np.dot(rot_mat, lattice_pos) + xy
                u = self.poseToIndex(lattice_pos)
                if self.isOutOfBounds(u):
                    continue
                if not is_in_list(u, [grids[0] for grids in updated_grids]):
                    self.map[u[0]][u[1]] = occupancy
                    updated_grids.append([u, occupancy])
        return updated_grids

    def poseToIndex(self, pose: np.ndarray) -> np.ndarray:
        return np.round(np.array(pose[0:2]) / self.grid_width).astype(np.int32)
        # return round_off(np.array(pose[0:2]) / self.grid_width).astype('int32')

    def indexToPose(self, idx):
        return np.array([idx[0] * self.grid_width, idx[1] * self.grid_width, 0.0])
        # return np.append(idx * self.grid_width, 0.0)

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
        expand_dist: float = 0.0,
        map_name='map'
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        extent = (
            -self.grid_width / 2,
            self.grid_width * self.grid_num[0] - self.grid_width / 2,
            -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
        )

        if map_name == 'map':
            draw_obstacles(ax, obstacles, expand_dist)
            draw_grid_map(ax, self.map, "Greys", 0.0, 1.0, 0.5, extent, 1.0)
        elif map_name == 'table':
            draw_obstacles(ax, obstacles, expand_dist, 0.3)
            draw_obstacles(ax, self.obstacles_table, expand_dist, 0.3)

        plt.show()


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
