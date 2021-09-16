import math
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Tuple
from rover_simulator.core import Mapper, Obstacle
from rover_simulator.navigation.path_planner import PathPlanner, PathNotFoundError
from rover_simulator.utils import environment_cmap, round_off


neigbor_grids = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])


class GridBasePathPlanning(PathPlanner):
    def __init__(self) -> None:
        pass

    def set_start(self, start_pos: np.ndarray):
        self.start_idx = self.poseToIndex(start_pos[0:2])

    def set_goal(self, goal_pos: np.ndarray):
        self.goal_idx = self.poseToIndex(goal_pos)

    def indexToPose(self, idx):
        return np.append(idx * self.grid_width, 0.0)

    def poseToIndex(self, pose: np.ndarray) -> np.ndarray:
        return round_off(np.array(pose[0:2]) / self.grid_width).astype('int32')

    def neigborGrids(self, idx):
        neigbors = []
        for grid in neigbor_grids:
            neigbor_grid = idx + grid
            if self.isOutOfBounds(neigbor_grid):
                continue
            neigbors.append(neigbor_grid)
        return neigbors

    def isStart(self, idx):
        if np.all(idx == self.start_idx):
            return True
        else:
            return False

    def isGoal(self, idx):
        if np.all(idx == self.goal_idx):
            return True
        else:
            return False

    def isObstacle(self, idx):
        if self.grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False

    def isOutOfBounds(self, idx: np.ndarray) -> bool:
        if idx[0] >= self.grid_num[0]:
            return True
        elif idx[0] < 0:
            return True
        if idx[1] >= self.grid_num[1]:
            return True
        elif idx[1] < 0:
            return True
        return False

    def isObstacleDiagonal(self, idx1, idx2):
        v = idx2 - idx1
        if not np.all(np.abs(v) == [1, 1]):
            return False
        else:
            if self.isObstacle(idx1 + [v[0], 0]):
                return True
            elif self.isObstacle(idx1 + [0, v[1]]):
                return True
            else:
                return False


class Dijkstra(GridBasePathPlanning):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        map: np.ndarray = None, map_grid_width: float = 1.0
    ):
        self.grid_map = map
        self.grid_width = map_grid_width
        self.start_idx = self.poseToIndex(start_pos) if start_pos is not None else None
        self.goal_idx = self.poseToIndex(goal_pos) if goal_pos is not None else None
        self.grid_num = np.array(map.shape) if map is not None else None
        self.grid_map = np.full(self.grid_num, 0.5, dtype=float) if map is not None else None
        self.cost_map = np.full(self.grid_num, 0.0, dtype=float) if map is not None else None
        self.id_map = np.full(self.grid_num, 0, dtype=np.int32) if map is not None else None
        self.parent_id_map = np.full(self.grid_num, 0, dtype=np.int32) if map is not None else None
        self.is_opened_map = np.full(np.append(np.array(map.shape), 3), 0.0) if map is not None else None
        self.is_closed_map = np.full(np.append(np.array(map.shape), 3), 0.0) if map is not None else None

        self.open_list = []
        self.resultPath = []
        self.takenPath = []

        self.name = "Dijkstra"

        if map is not None:
            cnt = 0
            for u, _ in np.ndenumerate(self.cost_map):
                self.id_map[u[0]][u[1]] = cnt
                if self.isStart(u):
                    self.open_list.append([cnt, 0, 0])
                    self.is_opened_map[u[0]][u[1]][0] = 1.0
                    self.is_opened_map[u[0]][u[1]][1] = 0.0
                    self.is_opened_map[u[0]][u[1]][2] = 0.0
                cnt += 1

    def set_map(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)
        self.cost_map = np.full(self.grid_num, 0.0, dtype=float)
        self.id_map = np.full(self.grid_num, 0, dtype=np.int32)
        self.parent_id_map = np.full(self.grid_num, 0, dtype=np.int32)
        self.is_opened_map = np.full(np.append(np.array(mapper.map.shape), 3), 0.0)
        self.is_closed_map = np.full(np.append(np.array(mapper.map.shape), 3), 0.0)
        cnt = 0
        for u, _ in np.ndenumerate(self.cost_map):
            self.id_map[u[0]][u[1]] = cnt
            if self.isStart(u):
                self.open_list.append([cnt, 0, 0])
                self.is_opened_map[u[0]][u[1]][0] = 1.0
                self.is_opened_map[u[0]][u[1]][1] = 0.0
                self.is_opened_map[u[0]][u[1]][2] = 0.0
            cnt += 1

    def calculate_path(self, *args):
        if self.isOutOfBounds(self.start_idx):
            raise PathNotFoundError("Start index is out of bounds")
        if self.isOutOfBounds(self.goal_idx):
            raise PathNotFoundError("Goal index is out of bounds")
        while not self.isClosed(self.goal_idx):
            idx, cost = self.expandGrid()
            self.cost_map[idx[0]][idx[1]] = cost
        path = self.get_path()
        waypoints = [self.indexToPose(self.goal_idx)]
        for grid in path:
            pose = self.indexToPose(grid)
            waypoints.append(pose[0:2])
        waypoints.reverse()
        return waypoints

    def expandGrid(self):
        if len(self.open_list) == 0:
            raise PathNotFoundError("Path was not found")
        val = np.argmin(self.open_list, axis=0)  # 評価マップの中から最も小さいもの抽出
        grid_id, cost_f, cost_g = self.open_list[val[1]]
        idx = np.array([np.where(self.id_map == grid_id)[0][0], np.where(self.id_map == grid_id)[1][0]])

        # Remove from Opened List
        self.open_list.remove([grid_id, cost_f, cost_g])
        self.is_opened_map[idx[0]][idx[1]][0] = 0.0
        self.is_opened_map[idx[0]][idx[1]][1] = 0.0
        self.is_opened_map[idx[0]][idx[1]][2] = 0.0

        # Append Closed List
        self.is_closed_map[idx[0]][idx[1]][0] = 1.0
        self.is_closed_map[idx[0]][idx[1]][1] = cost_f
        self.is_closed_map[idx[0]][idx[1]][2] = cost_g
        self.calculateCost(idx, cost_g)  # コストの計算
        return idx, cost_f

    def calculateCost(self, idx, cost_g):
        for neigbor_idx in self.listFreeNeigbor(idx):
            evaluation_f = cost_g + self.cost(neigbor_idx) + self.c(neigbor_idx, idx)
            if self.isOpened(neigbor_idx):
                neigbor_cost_f = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1]
                neigbor_cost_g = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2]
                if neigbor_cost_f > evaluation_f:
                    self.open_list.remove([self.id(neigbor_idx), neigbor_cost_f, neigbor_cost_g])
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][0] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2] = 0.0
                else:
                    continue
            elif self.isClosed(neigbor_idx):
                neigbor_cost_f = self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][1]
                neigbor_cost_g = self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][2]
                if neigbor_cost_f > evaluation_f:
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][0] = 0.0
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][1] = 0.0
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][2] = 0.0
                else:
                    continue
            self.parent_id_map[neigbor_idx[0]][neigbor_idx[1]] = self.id(idx)
            self.open_list.append([self.id(neigbor_idx), evaluation_f, evaluation_f])
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][0] = 1.0
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1] = evaluation_f
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2] = evaluation_f

    def get_path(self):
        parent_id = self.b(self.goal_idx)
        path = []
        while parent_id != self.id(self.start_idx):
            parent = np.where(self.id_map == parent_id)
            path.append(np.array([parent[0][0], parent[1][0]]))
            parent_id = self.b(parent)
        path.append(self.start_idx)
        return path

    def listFreeNeigbor(self, idx):
        neigbor_indice = []
        for neigbor_grid in neigbor_grids:
            neigbor_idx = idx + neigbor_grid
            if(
                not self.isOutOfBounds(neigbor_idx) and
                not self.isObstacle(neigbor_idx) and
                not self.isObstacleDiagonal(idx, neigbor_idx)  # Diagonal
            ):
                neigbor_indice.append(neigbor_idx)
        return neigbor_indice

    def isOpened(self, u):
        if self.is_opened_map[u[0]][u[1]][0] == 1.0:
            return True
        else:
            return False

    def isClosed(self, u):
        if self.is_closed_map[u[0]][u[1]][0] == 1.0:
            return True
        else:
            return False

    def id(self, u):
        return self.id_map[u[0]][u[1]]

    def cost(self, u):
        return self.cost_map[u[0]][u[1]]

    def b(self, u):
        return self.parent_id_map[int(u[0])][int(u[1])]

    def c(self, u, v):
        return np.linalg.norm(u - v)

    def draw_map(
        self,
        xlim: List[float], ylim: List[float],
        figsize: Tuple[float, float] = (8, 8),
        obstacles: List[Obstacle] = [],
        map_name: str = 'cost',
        enlarge_obstacle: float = 0.0,
    ):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        # Draw Obstacles
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray', zorder=-1.0)
            ax.add_patch(enl_obs)
        for obstacle in obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black', zorder=-1.0)
            ax.add_patch(obs)

        # Draw Map
        if map_name == 'cost':
            draw_map = self.cost_map
            cmap = 'plasma'
            vmin = None
            vmax = None
        elif map_name == 'grid':
            draw_map = self.grid_map
            cmap = 'Greys'
            vmin = 0.0
            vmax = 1.0

        draw_map = cv2.rotate(draw_map, cv2.ROTATE_90_COUNTERCLOCKWISE)
        im = ax.imshow(
            draw_map,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.5,
            extent=(
                -self.grid_width / 2,
                self.grid_width * self.grid_num[0] - self.grid_width / 2,
                -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
            ),
            zorder=1.0
        )
        plt.colorbar(im)


class Astar(Dijkstra):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        map: np.ndarray = None, map_grid_width: float = 1.0, heuristic=0.9
    ):
        super().__init__(start_pos, goal_pos, map, map_grid_width)
        self.heuristic = heuristic
        self.name = "Astar"

    def calculateCost(self, idx, cost_g):  # コストの計算
        for neigbor_idx in self.listFreeNeigbor(idx):
            evaluation_f = cost_g + self.cost(neigbor_idx) + self.c(neigbor_idx, idx) + self.__h(neigbor_idx)  # 評価を計算
            if self.isOpened(neigbor_idx):  # オープンリストに含まれているか
                neigbor_cost_f = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1]
                neigbor_cost_g = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2]
                # neigbor_idx, neigbor_cost_f, neigbor_cost_g = self.open_list[[val[0] for val in self.open_list].index(self.id(neigbor_idx))]
                if neigbor_cost_f > evaluation_f:
                    self.open_list.remove([self.id(neigbor_idx), neigbor_cost_f, neigbor_cost_g])
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][0] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2] = 0.0
                else:
                    continue
            elif self.isClosed(neigbor_idx):  # クローズドリストに含まれているか
                neigbor_cost_f = self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][1]
                neigbor_cost_g = self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][2]
                # its_idx, neigbor_cost_f, neigbor_cost_g = self.closed_list[[val[0] for val in self.closed_list].index(self.id(neigbor_idx))]
                if neigbor_cost_f > evaluation_f:
                    # self.closed_list.remove([neigbor_idx, neigbor_cost_f, neigbor_cost_g])
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][0] = 0.0
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][1] = 0.0
                    self.is_closed_map[neigbor_idx[0]][neigbor_idx[1]][2] = 0.0
                else:
                    continue
            self.parent_id_map[neigbor_idx[0]][neigbor_idx[1]] = self.id(idx)
            self.open_list.append([self.id(neigbor_idx), evaluation_f, evaluation_f - self.__h(neigbor_idx)])
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][0] = 1.0
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1] = evaluation_f
            self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2] = evaluation_f - self.__h(neigbor_idx)

    def __h(self, u):
        return 0.9 * np.linalg.norm(self.goal_idx - u)


class DstarLite(GridBasePathPlanning):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        mapper: np.ndarray = None, map_grid_width: float = 1.0,
        heuristics: float = 0.5,
    ):
        self.grid_width = map_grid_width
        self.start_idx = self.poseToIndex(start_pos) if start_pos is not None else None
        self.current_idx = self.start_idx if start_pos is not None else None
        self.goal_idx = self.poseToIndex(goal_pos) if goal_pos is not None else None
        self.heuristics = heuristics

        if mapper is not None:
            self.local_grid_map = np.full(mapper.map.shape, 0.5, dtype=float)  # Local Map is Obstacle Occupancy Grid Map
            self.metric_grid_map = np.full(self.grid_cost_num, -1.0, dtype=np.float)  # Metric Map shows wheter the grid is observed, -1: Unobserved, 0: Free, 1: Obstacles
            self.is_in_U_map = self.local_grid_map = np.full(mapper.mapshape, 0, dtype=np.int16)

        self.U = []
        self.km = 0.0

        self.pathToTake = []
        self.takenPath = []

        self.newObstacles = []
        self.name = "DstarLite"

    def set_map(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)

        if self.isOutOfBounds(self.start_idx):
            raise ValueError("Start position is out of bounds")
        if self.isOutOfBounds(self.goal_idx):
            raise ValueError("Goal position is out of bounds")

        self.local_grid_map = copy.copy(mapper.map)  # センシングによって構築したマップ
        self.local_grid_map[self.start_idx[0]][self.start_idx[1]] = 0.0
        self.metric_grid_map = np.full(self.grid_num, -1.0, dtype=np.float)  # 経路計画で使用するマップ
        self.metric_grid_map[self.start_idx[0]][self.start_idx[1]] = 0

        self.is_in_U_map = np.full(mapper.map.shape, 0, dtype=np.int16)

        self.g_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map[self.goal_idx[0]][self.goal_idx[1]] = 0
        self.__uAppend(self.goal_idx, [self.__h(self.start_idx, self.goal_idx), 0])
        self.previous_idx = np.array(self.start_idx)

    def initialize(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)

        self.local_grid_map = copy.copy(mapper.map)
        self.local_grid_map[self.start_idx[0]][self.start_idx[1]] = 0.0
        self.metric_grid_map = np.full(self.grid_num, -1.0, dtype=np.float)
        self.metric_grid_map[self.start_idx[0]][self.start_idx[1]] = 0

        self.g_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map[self.goal_idx[0]][self.goal_idx[1]] = 0
        self.__uAppend(self.goal_idx, [self.__h(self.start_idx, self.goal_idx), 0])
        self.previous_idx = np.array(self.start_idx)

        self.U = []
        self.km = 0.0

        self.pathToTake = []
        self.takenPath = []

        self.newObstacles = []

        self.current_idx = self.start_idx

    def calculate_path(self):
        self.computeShortestPath(self.start_idx)
        waypoints = self.get_path(self.current_idx)
        return waypoints

    def update_path(self, pose: np.ndarray, mapper: Mapper):
        self.current_idx = self.poseToIndex(pose)

        self.newObstacles = []
        self.newFrees = []

        for u, c in mapper.observed_grids:
            if self.isOutOfBounds(u):
                continue
            prev_occ = self.local_grid_map[u[0]][u[1]]
            self.local_grid_map[u[0]][u[1]] = mapper.map[u[0]][u[1]]
            if self.local_grid_map[u[0]][u[1]] > 0.5 and prev_occ <= 0.5:
                self.newObstacles.append(u)
            elif self.local_grid_map[u[0]][u[1]] <= 0.5 and prev_occ > 0.5:
                self.newFrees.append(u)

        # 障害物周囲の辺をリストアップ
        update_to_obstacle_list = []
        update_to_free_list = []
        # Free -> Obstacle
        for u in self.newObstacles:
            if self.isOutOfBounds(u):
                continue
            for v in self.neigborGrids(u):
                if self.isOutOfBounds(v):
                    continue
                if not self.__c(u, v) == float('inf'):
                    update_to_obstacle_list.append([u, v])
                    update_to_obstacle_list.append([v, u])
            # Diagonal
            for vertex in [[[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, -1], [-1, 0]]]:
                w, x = u + np.array(vertex[0]), u + np.array(vertex[1])
                if not self.__c(w, x) == float('inf'):
                    update_to_obstacle_list.append([w, x])
                    update_to_obstacle_list.append([x, w])

        # Obstacle -> Free
        for u in self.newFrees:
            if self.isOutOfBounds(u):
                continue
            for v in self.neigborGrids(u):
                if self.isOutOfBounds(v):
                    continue
                if not self.__c(u, v) == 0.0:
                    update_to_free_list.append([u, v])
                    update_to_free_list.append([v, u])
            # Diagonal
            for vertex in [[[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, -1], [-1, 0]]]:
                w, x = u + np.array(vertex[0]), u + np.array(vertex[1])
                if self.isOutOfBounds(w) or self.isOutOfBounds(x):
                    continue
                if not self.__c(w, x) == 0.0:
                    update_to_free_list.append([w, x])
                    update_to_free_list.append([x, w])

        # コストが変わる辺をリストアップ
        updated_vertex = []
        for vertex in update_to_obstacle_list:
            u, v = vertex[0], vertex[1]
            c = self.__c(u, v)
            updated_vertex.append([u, v, c, float('inf')])

        # Metric Mapを更新
        for u, _ in mapper.observed_grids:
            if self.local_grid_map[u[0]][u[1]] > 0.5:
                self.metric_grid_map[u[0]][u[1]] = 1.0
            else:
                self.metric_grid_map[u[0]][u[1]] = 0.0

        for vertex in update_to_free_list:
            u, v = vertex[0], vertex[1]
            c = self.__c(u, v)
            updated_vertex.append([u, v, float('inf'), c])

        if len(updated_vertex) > 0:
            self.km = self.km + self.__h(self.previous_idx, self.current_idx)
            self.previous_idx = self.current_idx

            # Update rhs
            for vertex in updated_vertex:
                u = vertex[0]
                v = vertex[1]
                c_old = vertex[2]
                c_new = vertex[3]
                if c_old > c_new:
                    if not self.isGoal(u):
                        self.rhs_map[u[0]][u[1]] = min(self.rhs(u), c_new + self.g(v))
                elif math.isclose(self.rhs(u), c_old + self.g(v)):
                    if not self.isGoal(u):
                        self.rhs_map[u[0]][u[1]] = self.__getMinRhs(u)
                self.updateVertex(u)

            self.computeShortestPath(self.current_idx)

        waypoints = self.get_path(self.current_idx)
        return waypoints

    def get_path(self, idx):
        self.pathToTake = [idx]

        last_cost = float('inf')
        if self.isObstacle(idx):
            next_idx = idx
            min_cost = float('inf')
            for s_ in self.neigborGrids(idx):
                c = 1.0 if np.linalg.norm(idx - s_) < 1.1 else 1.41
                if min_cost > c + self.g(s_) and last_cost >= self.g(s_):
                    min_cost = c + self.g(s_)
                    last_cost = self.g(s_)
                    next_idx = s_
            idx = next_idx
        # while not self.isGoal(idx):
        for i in range(30):
            if self.isGoal(idx):
                break
            next_idx = idx
            min_cost = float('inf')
            for s_ in self.neigborGrids(idx):
                if min_cost > self.__c(idx, s_) + self.g(s_) and last_cost >= self.g(s_):
                    min_cost = self.__c(idx, s_) + self.g(s_)
                    last_cost = self.g(s_)
                    next_idx = s_
            idx = next_idx
            if np.any(np.all(self.pathToTake == idx, axis=1)):
                break
            self.pathToTake.append(idx)
        self.pathToTake.append(self.goal_idx)
        waypoints = []
        for grid in self.pathToTake:
            waypoints.append(self.indexToPose(grid)[0:2])
        return waypoints

    def rhs(self, s):
        return self.rhs_map[s[0]][s[1]]

    def g(self, s):
        return self.g_map[s[0]][s[1]]

    def metric_map(self, s):
        return self.metric_grid_map[s[0]][s[1]]

    def local_map(self, s):
        return self.local_grid_map[s[0]][s[1]]

    def draw_map(
        self,
        xlim: List[float], ylim: List[float],
        figsize: Tuple[float, float] = (8, 8),
        map_name: str = 'cost',
        obstacles: List[Obstacle] = None,
        enlarge_obstacle: float = 0.0,
    ):
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlim(xlim[0], xlim[1])
        ax.set_ylim(ylim[0], ylim[1])
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        if map_name == 'cost':
            draw_map = self.g_map
        elif map_name == 'metric':
            draw_map = self.metric_grid_map
        elif map_name == 'local':
            draw_map = self.local_grid_map

        # Draw Obstacles
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_obstacle, fc='gray', ec='gray', zorder=-1.0)
            ax.add_patch(enl_obs)
        for obstacle in obstacles:
            obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r, fc='black', ec='black', zorder=-1.0)
            ax.add_patch(obs)

        # Draw Map
        if map_name == 'cost':
            cmap = 'plasma'
            vmin = None
            vmax = None
        elif map_name == 'metric':
            cmap = environment_cmap
            vmin = -1.0
            vmax = 1.0
        elif map_name == 'local':
            cmap = 'Greys'
            vmin = 0.0
            vmax = 1.0
        im = ax.imshow(
            cv2.rotate(draw_map, cv2.ROTATE_90_COUNTERCLOCKWISE),
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            alpha=0.5,
            extent=(
                -self.grid_width / 2,
                self.grid_width * self.grid_num[0] - self.grid_width / 2,
                -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
            ),
            zorder=1.0
        )
        plt.colorbar(im)

    def computeShortestPath(self, index):
        U_row = [row[1] for row in self.U]
        if len(U_row) == 0:
            return
        u_data = min(U_row)
        idx = U_row.index(u_data)
        u, k_old = np.array(self.U[idx][0]), self.U[idx][1]
        k_new = self.__calculateKey(u)
        g_u = self.g(u)
        rhs_u = self.rhs(u)

        while not self.__kCompare(k_old, self.__calculateKey(index)) or self.rhs(index) > self.g(index):
            if self.__kCompare(k_old, k_new):
                self.__uUpdate(u, k_new)
            elif g_u > rhs_u:
                self.g_map[u[0]][u[1]] = rhs_u
                self.__uRemove(u)
                for s in self.neigborGrids(u):
                    if not self.isGoal(s):
                        self.rhs_map[s[0]][s[1]] = min(self.rhs(s), self.__c(s, u) + self.g(u))
                    self.updateVertex(s)
            else:
                g_old = self.g(u)
                self.g_map[u[0]][u[1]] = float('inf')
                for s in self.neigborGrids(u) + [u]:
                    if math.isclose(self.rhs(s), self.__c(s, u) + g_old):
                        if not self.isGoal(s):
                            self.rhs_map[s[0]][s[1]] = self.__getMinRhs(s)
                    self.updateVertex(s)
            U_row = [row[1] for row in self.U]
            if len(U_row) == 0:
                break
            u_data = min(U_row)
            idx = U_row.index(u_data)
            u, k_old = np.array(self.U[idx][0]), self.U[idx][1]
            k_new = self.__calculateKey(u)
            g_u = self.g(u)
            rhs_u = self.rhs(u)

    def __calculateKey(self, s):
        key1 = min(self.g(s), self.rhs(s)) + self.__h(self.current_idx, s) + self.km
        key2 = min(self.g(s), self.rhs(s))
        return [key1, key2]

    def __kCompare(self, k1, k2):
        if k1[0] > k2[0]:
            return True
        elif math.isclose(k1[0], k2[0]):
            if k1[1] > k2[1] and not math.isclose(k1[1], k2[1]):
                return True
        return False

    def updateVertex(self, u):
        if self.is_in_U_map[u[0]][u[1]] == 1:
            u_flag = True
        else:
            u_flag = False
        # u_flag = list(u) in [row[0] for row in self.U]
        g_u = self.g(u)
        rhs_u = self.rhs(u)
        if not math.isclose(g_u, rhs_u) and u_flag:
            self.__uUpdate(u, self.__calculateKey(u))
        elif not math.isclose(g_u, rhs_u) and not u_flag:
            self.__uAppend(u, self.__calculateKey(u))
        elif math.isclose(g_u, rhs_u) and u_flag:
            self.__uRemove(u)

    def __uAppend(self, u, u_num):
        self.U.append([list(u), u_num])
        self.is_in_U_map[u[0]][u[1]] = 1

    def __uRemove(self, u):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U.remove([list(u), self.U[idx][1]])
        self.is_in_U_map[u[0]][u[1]] = 0

    def __uUpdate(self, u, u_num_new):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U[idx][1] = u_num_new
        self.is_in_U_map[u[0]][u[1]] = 1

    def __getMinRhs(self, u):
        min_rhs = float('inf')
        for v in self.neigborGrids(u):
            if min_rhs > self.__c(u, v) + self.g(v):
                min_rhs = self.__c(u, v) + self.g(v)
        return min_rhs

    def __c(self, u, v):
        if self.isOutOfBounds(u) or self.__isObservedObstacle(u) or self.isOutOfBounds(v) or self.__isObservedObstacle(v):
            return float('inf')
        else:
            if np.all(np.abs(u - v) == [1, 1]):
                c_ = 1.41
                if np.all(v - u == [1, 1]):
                    if self.metric_grid_map[u[0] + 1][u[1]] == 1 or self.metric_grid_map[u[0]][u[1] + 1] == 1:
                        c_ = float('inf')
                elif np.all(v - u == [1, -1]):
                    if self.metric_grid_map[u[0] + 1][u[1]] == 1 or self.metric_grid_map[u[0]][u[1] - 1] == 1:
                        c_ = float('inf')
                elif np.all(v - u == [-1, 1]):
                    if self.metric_grid_map[u[0] - 1][u[1]] == 1 or self.metric_grid_map[u[0]][u[1] + 1] == 1:
                        c_ = float('inf')
                elif np.all(v - u == [-1, -1]):
                    if self.metric_grid_map[u[0] - 1][u[1]] == 1 or self.metric_grid_map[u[0]][u[1] - 1] == 1:
                        c_ = float('inf')
            else:
                c_ = 1.0
            return c_

    def __h(self, s1, s2):
        return np.round(self.heuristics * np.linalg.norm(s1 - s2), decimals=2)

    def __isObservedObstacle(self, idx):
        if self.metric_grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False

    def isObstacle(self, idx):
        if self.metric_grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False
