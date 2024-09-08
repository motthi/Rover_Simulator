import copy
import heapq
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rover_simulator.core import Mapper, Obstacle
from rover_simulator.navigation.mapper import GridMapper
from rover_simulator.navigation.path_planner import PathPlanner, PathNotFoundError
from rover_simulator.utils.utils import round_off
from rover_simulator.utils.draw import set_fig_params, draw_grid_map, draw_grid_map_contour, draw_obstacles, draw_start, draw_goal, environment_cmap


neigbor_grids = np.array([[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]])


class GridBasePathPlanning(PathPlanner):
    grid_width: float

    def __init__(self) -> None:
        pass

    def set_start(self, start_pos: np.ndarray):
        self.start_idx = self.pose2index(start_pos[0:2])

    def set_goal(self, goal_pos: np.ndarray):
        self.goal_idx = self.pose2index(goal_pos)

    def index2pose(self, idx):
        return np.append(idx * self.grid_width, 0.0)

    def pose2index(self, pose: np.ndarray) -> np.ndarray:
        return round_off(np.array(pose[0:2]) / self.grid_width).astype('int32')

    def get_neigbors(self, idx):
        neigbors = []
        for grid in neigbor_grids:
            neigbor_grid = idx + grid
            if self.is_ob(neigbor_grid):
                continue
            neigbors.append(neigbor_grid)
        return neigbors

    def is_start(self, idx):
        if np.all(idx == self.start_idx):
            return True
        else:
            return False

    def is_goal(self, idx):
        if np.all(idx == self.goal_idx):
            return True
        else:
            return False

    def is_obstacle(self, idx):
        if self.grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False

    def is_ob(self, idx: np.ndarray) -> bool:
        if idx[0] >= self.grid_num[0]:
            return True
        elif idx[0] < 0:
            return True
        if idx[1] >= self.grid_num[1]:
            return True
        elif idx[1] < 0:
            return True
        return False

    def has_obstacle_diag(self, idx1, idx2):
        v = idx2 - idx1
        if not np.all(np.abs(v) == [1, 1]):
            return False
        else:
            if self.is_obstacle(idx1 + [v[0], 0]):
                return True
            elif self.is_obstacle(idx1 + [0, v[1]]):
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
        self.start_idx = self.pose2index(start_pos) if start_pos is not None else None
        self.goal_idx = self.pose2index(goal_pos) if goal_pos is not None else None
        self.grid_num = np.array(map.shape) if map is not None else None
        self.grid_map = np.full(self.grid_num, 0.5, dtype=float) if map is not None else None
        self.cost_map = np.full(self.grid_num, float('inf'), dtype=float) if map is not None else None
        self.id_map = np.full(self.grid_num, 0, dtype=np.int32) if map is not None else None
        self.parent_id_map = np.full(self.grid_num, 0, dtype=np.int32) if map is not None else None
        self.is_opened_map = np.full(np.append(np.array(map.shape), 3), 0.0) if map is not None else None
        self.is_closed_map = np.full(np.append(np.array(map.shape), 3), 0.0) if map is not None else None

        self.open_list = []
        self.resultPath = []
        self.takenPath = []

        self.name = "Dijkstra"

        if map is not None:
            for cnt, (u, _) in enumerate(np.ndenumerate(self.cost_map)):
                self.id_map[u[0]][u[1]] = cnt
                if self.is_start(u):
                    self.open_list.append([cnt, 0, 0])
                    self.is_opened_map[u[0]][u[1]][0] = 1.0
                    self.is_opened_map[u[0]][u[1]][1] = 0.0
                    self.is_opened_map[u[0]][u[1]][2] = 0.0
                    self.cost_map[u[0]][u[1]] = 0.0

    def set_map(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)
        self.cost_map = np.full(self.grid_num, float('inf'), dtype=float)
        self.id_map = np.full(self.grid_num, 0, dtype=np.int32)
        self.parent_id_map = np.full(self.grid_num, 0, dtype=np.int32)
        self.is_opened_map = np.full(np.append(np.array(mapper.map.shape), 3), 0.0)
        self.is_closed_map = np.full(np.append(np.array(mapper.map.shape), 3), 0.0)
        for cnt, (u, _) in enumerate(np.ndenumerate(self.cost_map)):
            self.id_map[u[0]][u[1]] = cnt
            if self.is_start(u):
                self.open_list.append([cnt, 0, 0])
                self.is_opened_map[u[0]][u[1]][0] = 1.0
                self.is_opened_map[u[0]][u[1]][1] = 0.0
                self.is_opened_map[u[0]][u[1]][2] = 0.0
                self.cost_map[u[0]][u[1]] = 0.0

    def calculate_path(self, *args):
        if self.is_ob(self.start_idx):
            raise PathNotFoundError("Start index is out of bounds")
        if self.is_ob(self.goal_idx):
            raise PathNotFoundError("Goal index is out of bounds")
        while not self.is_closed(self.goal_idx):
            idx, cost = self.expand_grid()
            self.cost_map[idx[0]][idx[1]] = cost
        path = self.get_path()
        waypoints = [list(self.index2pose(self.goal_idx)[:2])]
        for grid in path:
            pose = self.index2pose(grid)
            waypoints.append(list(pose[0:2]))
        waypoints.reverse()
        return np.array(waypoints)

    def expand_grid(self):
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
        self.calc_cost(idx, cost_g)  # コストの計算
        return idx, cost_f

    def calc_cost(self, idx, cost_g):
        for neigbor_idx in self.get_free_neigbors(idx):
            # evaluation_f = cost_g + self.cost(neigbor_idx) + self.c(neigbor_idx, idx)
            evaluation_f = cost_g + self.c(neigbor_idx, idx)
            if self.is_opened(neigbor_idx):
                neigbor_cost_f = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1]
                neigbor_cost_g = self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2]
                if neigbor_cost_f > evaluation_f:
                    self.open_list.remove([self.id(neigbor_idx), neigbor_cost_f, neigbor_cost_g])
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][0] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][1] = 0.0
                    self.is_opened_map[neigbor_idx[0]][neigbor_idx[1]][2] = 0.0
                else:
                    continue
            elif self.is_closed(neigbor_idx):
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

    def get_free_neigbors(self, idx):
        neigbor_indice = []
        for neigbor_grid in neigbor_grids:
            neigbor_idx = idx + neigbor_grid
            if (
                not self.is_ob(neigbor_idx) and
                not self.is_obstacle(neigbor_idx) and
                not self.has_obstacle_diag(idx, neigbor_idx)  # Diagonal
            ):
                neigbor_indice.append(neigbor_idx)
        return neigbor_indice

    def is_opened(self, u):
        if self.is_opened_map[u[0]][u[1]][0] == 1.0:
            return True
        else:
            return False

    def is_closed(self, u):
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

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        obstacles: list[Obstacle] = [],
        map_name: str = 'cost',
        enlarge_range: float = 0.0,
        draw_map=True,
        draw_contour=True
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, enlarge_range)

        # Draw Map
        if map_name == 'cost':
            map = self.cost_map
            cmap = 'plasma'
            vmin = None
            vmax = None
        elif map_name == 'grid':
            map = self.grid_map
            cmap = 'Greys'
            vmin = 0.0
            vmax = 1.0
        extent = (
            -self.grid_width / 2,
            self.grid_width * self.grid_num[0] - self.grid_width / 2,
            -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
        )

        if draw_map:
            draw_grid_map(ax, map, cmap, vmin, vmax, 0.5, extent, 1.0)
        if draw_contour:
            draw_grid_map_contour(ax, map, self.grid_num, self.grid_width, 20)
        draw_start(ax, self.index2pose(self.start_idx))
        draw_goal(ax, self.index2pose(self.goal_idx))
        plt.show()


class Astar(Dijkstra):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        map: np.ndarray = None, map_grid_width: float = 1.0, heuristic=0.9
    ):
        super().__init__(start_pos, goal_pos, map, map_grid_width)
        self.heuristic = heuristic
        self.name = "Astar"

    def calc_cost(self, idx, cost_g):  # コストの計算
        for neigbor_idx in self.get_free_neigbors(idx):
            evaluation_f = cost_g + self.c(neigbor_idx, idx) + self.__h(neigbor_idx)  # 評価を計算
            if self.is_opened(neigbor_idx):  # オープンリストに含まれているか
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
            elif self.is_closed(neigbor_idx):  # クローズドリストに含まれているか
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
        mapper: GridMapper = None, map_grid_width: float = 1.0,
        heuristics: float = 0.5,
    ):
        self.grid_width = map_grid_width
        self.start_idx = self.pose2index(start_pos) if start_pos is not None else None
        self.current_idx = self.start_idx if start_pos is not None else None
        self.goal_idx = self.pose2index(goal_pos) if goal_pos is not None else None
        self.heuristics = heuristics

        if mapper:
            self.local_grid_map = np.full(mapper.map.shape, 0.5, dtype=float)  # Local Map is Obstacle Occupancy Grid Map
            self.metric_grid_map = np.full(self.grid_cost_num, -1.0, dtype=np.float)  # Metric Map shows wheter the grid is observed, -1: Unobserved, 0: Free, 1: Obstacles
            self.is_in_U_map = np.full(mapper.map.shape, 0, dtype=np.int16)

        self.U = []
        self.km = 0.0

        self.pathToTake = []
        self.takenPath = []

        self.new_obstacles = []
        self.name = "DstarLite"

    def set_map(self, mapper: GridMapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)

        if self.is_ob(self.start_idx):
            raise ValueError("Start position is out of bounds")
        if self.is_ob(self.goal_idx):
            raise ValueError("Goal position is out of bounds")

        self.local_grid_map = copy.copy(mapper.map)  # センシングによって構築したマップ
        self.metric_grid_map = np.full(self.grid_num, -1.0, dtype=float)  # 経路計画で使用するマップ
        self.g_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map = np.full(self.local_grid_map.shape, float('inf'))
        self.is_in_U_map = np.full(self.local_grid_map.shape, 0, dtype=np.int16)

        self.local_grid_map[self.start_idx[0]][self.start_idx[1]] = 0.0
        self.rhs_map[self.goal_idx[0]][self.goal_idx[1]] = 0

        self.U = []
        self.__u_append(self.goal_idx, [self.__h(self.start_idx, self.goal_idx), 0])
        self.previous_idx = np.array(self.start_idx)

        for idx, occ in np.ndenumerate(self.local_grid_map):
            if occ > 0.5:
                self.metric_grid_map[idx[0]][idx[1]] = 1.0
        self.metric_grid_map[self.start_idx[0]][self.start_idx[1]] = 0

    def initialize(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)

        self.local_grid_map = copy.copy(mapper.map)
        self.metric_grid_map = np.full(self.grid_num, -1.0, dtype=np.float)
        self.g_map = np.full(self.local_grid_map.shape, float('inf'))
        self.rhs_map = np.full(self.local_grid_map.shape, float('inf'))
        self.is_in_U_map = np.full(mapper.map.shape, 0, dtype=np.int16)

        self.local_grid_map[self.start_idx[0]][self.start_idx[1]] = 0.0
        self.metric_grid_map[self.start_idx[0]][self.start_idx[1]] = 0
        self.rhs_map[self.goal_idx[0]][self.goal_idx[1]] = 0

        self.km = 0.0
        self.U = []
        self.__u_append(self.goal_idx, [self.__h(self.start_idx, self.goal_idx), 0])

        self.pathToTake = []
        self.takenPath = []
        self.new_obstacles = []

        self.previous_idx = np.array(self.start_idx)
        self.current_idx = self.start_idx

        for idx, occ in np.ndenumerate(self.local_grid_map):
            if occ > 0.5:
                self.metric_grid_map[idx[0]][idx[1]] = 1.0

    def calculate_path(self) -> np.ndarray:
        self.compute_shortest_path(self.start_idx)
        waypoints = self.get_path(self.current_idx)
        return waypoints

    def update_path(self, pose: np.ndarray, mapper: GridMapper):
        self.current_idx = self.pose2index(pose)

        self.new_obstacles = []
        self.new_frees = []

        for u, c in mapper.observed_grids:
            if self.is_ob(u):
                continue
            prev_occ = self.local_grid_map[u[0]][u[1]]
            self.local_grid_map[u[0]][u[1]] = mapper.map[u[0]][u[1]]
            if self.local_grid_map[u[0]][u[1]] > 0.5 and prev_occ <= 0.5:
                self.new_obstacles.append(u)
            elif self.local_grid_map[u[0]][u[1]] <= 0.5 and prev_occ > 0.5:
                self.new_frees.append(u)
        # self.local_grid_map[self.current_idx[0]][self.current_idx[1]] = 0.01

        # Listup edge around obstacles
        update_to_obstacle_list = []
        update_to_free_list = []

        # Free -> Obstacle
        for u in self.new_obstacles:
            if self.is_ob(u):
                continue
            for v in self.get_neigbors(u):
                if self.is_ob(v):
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
        for u in self.new_frees:
            if self.is_ob(u):
                continue
            for v in self.get_neigbors(u):
                if self.is_ob(v):
                    continue
                if not self.__c(u, v) == 0.0:
                    update_to_free_list.append([u, v])
                    update_to_free_list.append([v, u])
            # Diagonal
            for vertex in [[[-1, 0], [0, 1]], [[0, 1], [1, 0]], [[1, 0], [0, -1]], [[0, -1], [-1, 0]]]:
                w, x = u + np.array(vertex[0]), u + np.array(vertex[1])
                if self.is_ob(w) or self.is_ob(x):
                    continue
                if not self.__c(w, x) == 0.0:
                    update_to_free_list.append([w, x])
                    update_to_free_list.append([x, w])

        # Listup edge which cost is changed
        updated_vertex = []
        for vertex in update_to_obstacle_list:
            u, v = vertex[0], vertex[1]
            c = self.__c(u, v)
            updated_vertex.append([u, v, c, float('inf')])

        # Update Metric Map
        for u, _ in mapper.observed_grids:
            if self.local_grid_map[u[0]][u[1]] > 0.5:
                self.metric_grid_map[u[0]][u[1]] = 1.0
            else:
                self.metric_grid_map[u[0]][u[1]] = 0.0

        for vertex in update_to_free_list:
            u, v = vertex[0], vertex[1]
            c = self.__c(u, v)
            updated_vertex.append([u, v, float('inf'), c])

        # Update Path
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
                    if not self.is_goal(u):
                        self.rhs_map[u[0]][u[1]] = min(self.rhs(u), c_new + self.g(v))
                elif math.isclose(self.rhs(u), c_old + self.g(v)):
                    if not self.is_goal(u):
                        self.rhs_map[u[0]][u[1]] = self.__get_min_rhs(u)
                self.update_vertex(u)

            self.compute_shortest_path(self.current_idx)

        waypoints = self.get_path(self.current_idx)
        return waypoints

    def get_path(self, idx) -> np.ndarray:
        self.pathToTake = [idx]

        last_cost = float('inf')
        if self.is_obstacle(idx):
            next_idx = idx
            min_cost = float('inf')
            for s_ in self.get_neigbors(idx):
                c = 1.0 if np.linalg.norm(idx - s_) < 1.1 else 1.41
                if min_cost > c + self.g(s_) and last_cost >= self.g(s_):
                    min_cost = c + self.g(s_)
                    last_cost = self.g(s_)
                    next_idx = s_
            idx = next_idx
        # while not self.isGoal(idx):
        for i in range(200):
            if self.is_goal(idx):
                break
            next_idx = idx
            min_cost = float('inf')
            for s_ in self.get_neigbors(idx):
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
            waypoints.append(list(self.index2pose(grid)[0:2]))
        return np.array(waypoints)

    def rhs(self, s):
        return self.rhs_map[s[0]][s[1]]

    def g(self, s):
        return self.g_map[s[0]][s[1]]

    def metric_map(self, s):
        return self.metric_grid_map[s[0]][s[1]]

    def local_map(self, s):
        return self.local_grid_map[s[0]][s[1]]

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        map_name: str = 'cost',
        obstacles: list[Obstacle] = [],
        enlarge_range: float = 0.0,
        draw_map: bool = True,
        draw_contour: bool = True
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)

        if map_name == 'cost':
            map = self.g_map
        elif map_name == 'metric':
            map = self.metric_grid_map
        elif map_name == 'local':
            map = self.local_grid_map

        # Draw Obstacles
        draw_obstacles(ax, obstacles, enlarge_range)

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
        extent = (
            -self.grid_width / 2,
            self.grid_width * self.grid_num[0] - self.grid_width / 2,
            -self.grid_width / 2, self.grid_width * self.grid_num[1] - self.grid_width / 2
        )

        if draw_map:
            draw_grid_map(ax, map, cmap, vmin, vmax, 0.5, extent, 1.0)
        if draw_contour:
            draw_grid_map_contour(ax, map, self.grid_num, self.grid_width, 30)
        draw_start(ax, self.index2pose(self.start_idx))
        draw_goal(ax, self.index2pose(self.goal_idx))
        plt.show()

    def compute_shortest_path(self, index):
        U_row = [row[1] for row in self.U]
        if len(U_row) == 0:
            return
        u_data = min(U_row)
        idx = U_row.index(u_data)
        u, k_old = np.array(self.U[idx][0]), self.U[idx][1]
        k_new = self.__calculate_key(u)
        g_u = self.g(u)
        rhs_u = self.rhs(u)

        while not self.__k_compare(k_old, self.__calculate_key(index)) or self.rhs(index) > self.g(index):
            if self.__k_compare(k_old, k_new):
                self.__u_update(u, k_new)
            elif g_u > rhs_u:
                self.g_map[u[0]][u[1]] = rhs_u
                self.__u_remove(u)
                for s in self.get_neigbors(u):
                    if not self.is_goal(s):
                        self.rhs_map[s[0]][s[1]] = min(self.rhs(s), self.__c(s, u) + self.g(u))
                    self.update_vertex(s)
            else:
                g_old = self.g(u)
                self.g_map[u[0]][u[1]] = float('inf')
                for s in self.get_neigbors(u) + [u]:
                    if math.isclose(self.rhs(s), self.__c(s, u) + g_old):
                        if not self.is_goal(s):
                            self.rhs_map[s[0]][s[1]] = self.__get_min_rhs(s)
                    self.update_vertex(s)

            U_row = [row[1] for row in self.U]
            if len(U_row) == 0:
                break
            u_data = min(U_row)
            idx = U_row.index(u_data)
            u, k_old = np.array(self.U[idx][0]), self.U[idx][1]
            k_new = self.__calculate_key(u)
            g_u = self.g(u)
            rhs_u = self.rhs(u)

    def __calculate_key(self, s):
        key1 = min(self.g(s), self.rhs(s)) + self.__h(self.current_idx, s) + self.km
        key2 = min(self.g(s), self.rhs(s))
        return [key1, key2]

    def __k_compare(self, k1, k2):
        if k1[0] > k2[0]:
            return True
        elif math.isclose(k1[0], k2[0]):
            if k1[1] > k2[1] and not math.isclose(k1[1], k2[1]):
                return True
        return False

    def update_vertex(self, u):
        if self.is_in_U_map[u[0]][u[1]] == 1:
            u_flag = True
        else:
            u_flag = False
        # u_flag = list(u) in [row[0] for row in self.U]
        g_u = self.g(u)
        rhs_u = self.rhs(u)
        if not math.isclose(g_u, rhs_u) and u_flag:
            self.__u_update(u, self.__calculate_key(u))
        elif not math.isclose(g_u, rhs_u) and not u_flag:
            self.__u_append(u, self.__calculate_key(u))
        elif math.isclose(g_u, rhs_u) and u_flag:
            self.__u_remove(u)

    def __u_append(self, u, u_num):
        self.U.append([list(u), u_num])
        self.is_in_U_map[u[0]][u[1]] = 1

    def __u_remove(self, u):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U.remove([list(u), self.U[idx][1]])
        self.is_in_U_map[u[0]][u[1]] = 0

    def __u_update(self, u, u_num_new):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U[idx][1] = u_num_new
        self.is_in_U_map[u[0]][u[1]] = 1

    def __get_min_rhs(self, u):
        min_rhs = float('inf')
        for v in self.get_neigbors(u):
            if min_rhs > self.__c(u, v) + self.g(v):
                min_rhs = self.__c(u, v) + self.g(v)
        return min_rhs

    def __c(self, u, v):
        if self.is_ob(u) or self.__isoObserved_obstacle(u) or self.is_ob(v) or self.__isoObserved_obstacle(v):
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

    def __isoObserved_obstacle(self, idx):
        if self.metric_grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False

    def is_obstacle(self, idx):
        if self.metric_grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False


class FieldDstar(GridBasePathPlanning):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        mapper: GridMapper = None,
        heristics: float = 0.5
    ) -> None:
        super().__init__()
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.start_idx = self.pose2index(start_pos) if start_pos is not None else None
        self.current_idx = self.start_idx if start_pos is not None else None
        self.goal_idx = self.pose2index(goal_pos) if goal_pos is not None else None
        self.heuristics = heristics

        self.opened_map = np.full(mapper.map.shape, 0, dtype=int)   # Map express wheter it is OPEN ot not
        self.key_map = np.full(np.append(np.array(mapper.map.shape), 2), np.array([float('inf'), float('inf')]))
        self.g_map = np.full(mapper.map.shape, float('inf'), dtype=float)
        self.rhs_map = np.full(mapper.map.shape, float('inf'))
        self.bptr_map = np.full(np.append(np.array(mapper.map.shape), 2), np.array([0, 0]))
        self.visted_map = np.full(mapper.map.shape, 0.0, dtype=int)
        self.metric_grid_map = np.full(mapper.map.shape, -1.0, dtype=np.float)  # Metric Map shows wheter the grid is observed, -1: Unobserved, 0: Free, 1: Obstacles

        self.g_map[self.start_idx[0]][self.start_idx[1]] = float('inf')
        self.rhs_map[self.start_idx[0]][self.start_idx[1]] = float('inf')
        self.g_map[self.goal_idx[0]][self.goal_idx[1]] = float('inf')
        self.rhs_map[self.goal_idx[0]][self.goal_idx[1]] = 0.0
        self.set_open(self.goal_idx)
        self.visted_map[self.goal_idx[0]][self.goal_idx[1]] = 1

        # if mapper is not None:
        #     self.local_grid_map = np.full(mapper.map.shape, 0.5, dtype=float)  # Local Map is Obstacle Occupancy Grid Map
        #     self.metric_grid_map = np.full(self.grid_cost_num, -1.0, dtype=np.float)  # Metric Map shows wheter the grid is observed, -1: Unobserved, 0: Free, 1: Obstacles
        #     self.is_in_U_map = np.full(mapper.mapshape, 0, dtype=np.int16)

        self.pathToTake = []
        self.takenPath = []

        self.newObstacles = []
        self.name = "FieldDstar"

    def calculate_path(self):
        self.compute_shortest_path()
        waypoints = self.get_path(self.current_idx)
        return waypoints

    def get_path(self, idx):
        # I do not know how to extract path
        self.pathToTake = [idx]

        last_cost = float('inf')
        # if self.isObstacle(idx):
        #     next_idx = idx
        #     min_cost = float('inf')
        #     for s_ in self.neigborGrids(idx):
        #         c = 1.0 if np.linalg.norm(idx - s_) < 1.1 else 1.41
        #         if min_cost > c + self.g(s_) and last_cost >= self.g(s_):
        #             min_cost = c + self.g(s_)
        #             last_cost = self.g(s_)
        #             next_idx = s_
        #     idx = next_idx

        # while not self.isGoal(idx):
        for i in range(200):
            if self.is_goal(idx):
                break
            next_idx = idx
            min_cost = float('inf')
            for s_ in self.get_neigbors(idx):
                if last_cost >= self.g(s_):
                    last_cost = self.g(s_)
                    next_idx = s_
            idx = next_idx
            if np.any(np.all(self.pathToTake == idx, axis=1)):
                break
            self.pathToTake.append(idx)
        self.pathToTake.append(self.goal_idx)
        waypoints = []
        for grid in self.pathToTake:
            waypoints.append(self.index2pose(grid)[0:2])
        return waypoints

    def compute_shortest_path(self):
        while not self.k_compare(self.min_val_of_opened(), self.key(self.start_idx)) or self.rhs(self.start_idx) != self.g(self.start_idx):
            s = self.min_idx_of_opened()
            if s is None:
                print("Cannot find the path")
                break
            self.visted_map[s[0]][s[1]] = 1
            if self.g(s) > self.rhs(s):
                self.g_map[s[0]][s[1]] = self.rhs(s)
                self.unset_open(s)
                for s_ in self.get_neigbors(s):
                    if not self.has_visited(s_):
                        self.g_map[s_[0]][s_[1]] = float('inf')
                        self.rhs_map[s_[0]][s_[1]] = float('inf')
                        # self.visted_map[s_[0]][s_[1]] = 1
                    rhs_old = self.rhs(s_)
                    if self.rhs(s_) > self.compute_cost(s_, s, self.ccknbr(s_, s)):
                        self.rhs_map[s_[0]][s_[1]] = self.compute_cost(s_, s, self.ccknbr(s_, s))
                        self.bptr_map[s_[0]][s_[1]] = s
                    if self.rhs(s_) > self.compute_cost(s_, s, self.cknbr(s_, s)):
                        self.rhs_map[s_[0]][s_[1]] = self.compute_cost(s_, self.cknbr(s_, s), s)
                        self.bptr_map[s_[0]][s_[1]] = self.cknbr(s_, s)
                    if not math.isclose(self.rhs(s), rhs_old):
                        self.update_state(s_)
            else:
                self.rhs_map[s[0]][s[1]] = self.min_cost_in_neigbor(s)
                self.bptr_map[s[0]][s[1]] = self.min_idx_in_neigbor(s)
                if self.g(s) < self.rhs(s):
                    self.g_map[s[0]][s[1]] = float('inf')
                    for s_ in self.get_neigbors(s):
                        if np.all(self.bptr(s_) == s) or np.all(self.bptr(s_) == self.cknbr(s_, s)):
                            if not math.isclose(self.rhs(s_), self.compute_cost(s_, self.bptr(s_), self.ccknbr(s_, self.bptr(s_)))):
                                if self.g(s_) < self.rhs(s_) or not self.is_open(s_):
                                    self.rhs_map[s_[0]][s_[1]] = float('inf')
                                    self.update_state(s_)
                                else:
                                    self.rhs_map[s_[0]][s_[1]] = self.min_cost_in_neigbor(s_)
                                    self.bptr_map[s_[0]][s_[1]] = self.min_idx_in_neigbor(s_)
                                    self.update_state(s_)
                        self.visted_map[s_[0]][s_[1]] = 1
                self.update_state(s)

    def update_cell_cost(self, x, c):
        if c > x:
            for s in self.corners(x):
                if self.is_corner(self.bptr(s)) or self.is_corner(self.ccknbr(s)):
                    if not math.isclose(self.rhs(s), self.compute_cost(s, self.bptr(s), self.ccknbr(s, self.bptr(s)))):
                        if self.g(s) < self.rhs(s) or self.is_open(s):
                            self.rhs_map[s[0]][s[1]] = float('inf')
                            self.update_state(s)
                        else:
                            self.rhs_map[s[0]][s[1]] = self.min_cost_in_neigbor(s)
                            self.bptr_map[s[0]][s[1]] = self.min_idx_in_neigbor(s)
                            self.update_state(s)
        else:
            rhs_min = float('inf')
            for s in self.corners(x):
                if not self.has_visited(s):
                    self.g_map[s[0]][s[1]] = float('inf')
                    self.rhs_map[s[0]][s[1]] = float('inf')
                    self.visted_map[s[0]][s[1]] = 1
                elif self.rhs(s) < rhs_min:
                    rhs_min = self.rhs(s)
                    s_star = s
            if not math.isclose(rhs_min, float('inf')):
                self.set_open(s_star)

    def update_state(self, s):
        if self.g(s) != self.rhs(s):
            self.set_open(s)
        elif self.is_open(s):
            self.unset_open(s)

    def compute_cost(self, s, sa, sb):
        if self.is_diagonal_neigbor(s, sa):
            s1 = sb
            s2 = sa
        else:
            s1 = sa
            s2 = sb
        if self.is_ob(s1) or self.is_ob(s2):
            return float('inf')
        c_ = self.c(s, s2)
        b_ = self.c(s, s1)
        if min(c_, b_) == float('inf'):
            vs = float('inf')
        elif self.g(s1) <= self.g(s2):
            vs = min(c_, b_) + self.g(s1)
        else:
            f = self.g(s1) - self.g(s2)
            if f <= b_:
                if c_ <= f:
                    vs = c_ * np.sqrt(2.0) + self.g(s2)
                else:
                    y = min(f / np.sqrt(c_**2.0 - f**2.0), 1.0)
                    vs = c_ * np.sqrt(1.0 + y**2.0) + f * (1.0 - y) + self.g(s2)
            else:
                if c_ <= b_:
                    vs = c_ * np.sqrt(2.0) + self.g(s2)
                else:
                    x = 1.0 - min(b_ / np.sqrt(c_**2 - b_**2), 1.0)
                    vs = c_ * np.sqrt(1.0 + (1.0 - x)**2) + b_ * x + self.g(s2)
        return vs

    def is_diagonal_neigbor(self, s, s_):
        ds = s_ - s
        if np.all(np.abs(ds) == np.array([1, 1])):
            return True
        else:
            False

    def cknbr(self, s: np.ndarray, s_: np.ndarray) -> np.ndarray:
        ds = s - s_
        if np.all(ds == np.array([1, 0])):  # 右
            return s + np.array([1, -1])
        elif np.all(ds == np.array([1, 1])):  # 右上
            return s + np.array([1, 0])
        elif np.all(ds == np.array([0, 1])):  # 上
            return s + np.array([1, 1])
        elif np.all(ds == np.array([-1, 1])):  # 左上
            return s + np.array([0, 1])
        elif np.all(ds == np.array([-1, 0])):  # 左
            return s + np.array([-1, 1])
        elif np.all(ds == np.array([-1, -1])):  # 左下
            return s + np.array([-1, 0])
        elif np.all(ds == np.array([0, -1])):  # 下
            return s + np.array([-1, -1])
        elif np.all(ds == np.array([1, -1])):  # 右下
            return s + np.array([0, -1])

    def ccknbr(self, s: np.ndarray, s_: np.ndarray) -> np.ndarray:
        ds = s - s_
        if np.all(ds == np.array([1, 0])):  # 右
            return s + np.array([1, 1])
        elif np.all(ds == np.array([1, 1])):  # 右上
            return s + np.array([0, 1])
        elif np.all(ds == np.array([0, 1])):  # 上
            return s + np.array([-1, 1])
        elif np.all(ds == np.array([-1, 1])):  # 左上
            return s + np.array([-1, 0])
        elif np.all(ds == np.array([-1, 0])):  # 左
            return s + np.array([-1, -1])
        elif np.all(ds == np.array([-1, -1])):  # 左下
            return s + np.array([0, -1])
        elif np.all(ds == np.array([0, -1])):  # 下
            return s + np.array([1, -1])
        elif np.all(ds == np.array([1, -1])):  # 右下
            return s + np.array([1, 0])

    def key(self, s: np.ndarray) -> list:
        return [min(self.g(s), self.rhs(s)) + self.h(self.start_idx, s), min(self.g(s), self.rhs(s))]

    def k_compare(self, k1, k2):
        if k1[0] > k2[0]:
            return True
        elif math.isclose(k1[0], k2[0]):
            if k1[1] > k2[1] and not math.isclose(k1[1], k2[1]):
                return True
        return False

    def rhs(self, s):
        return self.rhs_map[s[0]][s[1]]

    def g(self, s):
        return self.g_map[s[0]][s[1]]

    def h(self, s1, s2):
        return np.round(self.heuristics * np.linalg.norm(s1 - s2), decimals=2)

    def c(self, u, v):
        if self.is_ob(u) or self.is_observed_obstacle(u) or self.is_ob(v) or self.is_observed_obstacle(v):
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

    def bptr(self, s: np.ndarray) -> np.ndarray:
        return self.bptr_map[s[0]][s[1]]

    def min_idx_in_neigbor(self, s: np.ndarray) -> np.ndarray:
        min_cost = float('inf')
        min_idx = np.array([0, 0])
        for s_ in self.get_neigbors(s):
            cost = self.compute_cost(s, s_, self.ccknbr(s, s_))
            if min_cost > cost:
                min_cost = cost
                min_idx = s_
        return min_idx

    def min_cost_in_neigbor(self, s: np.ndarray) -> float:
        min_cost = float('inf')
        for s_ in self.get_neigbors():
            cost = self.compute_cost(s, s_, self.ccknbr(s, s_))
            if min_cost > cost:
                min_cost = cost
        return cost

    def min_idx_of_opened(self):
        x_idxes, y_idxes = np.where(self.opened_map == 1)
        min_idx = None
        key_val = [float('inf'), float('inf')]
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            val = self.key(np.array([x_idx, y_idx]))
            if key_val[0] > val[0]:
                key_val = val
                min_idx = np.array([x_idx, y_idx])
        return min_idx

    def min_val_of_opened(self):
        x_idxes, y_idxes = np.where(self.opened_map == 1)
        key_val = [float('inf'), float('inf')]
        for x_idx, y_idx in zip(x_idxes, y_idxes):
            val = self.key_map[x_idx][y_idx]
            if key_val[0] > val[0]:
                key_val = val
        return key_val

    def set_open(self, s):
        self.opened_map[s[0]][s[1]] = 1
        k = self.key(s)
        self.key_map[s[0]][s[1]][0] = k[0]
        self.key_map[s[0]][s[1]][1] = k[1]

    def unset_open(self, s):
        self.opened_map[s[0]][s[1]] = 0
        self.key_map[s[0]][s[1]][0] = float('inf')
        self.key_map[s[0]][s[1]][1] = float('inf')

    def is_open(self, s):
        if self.opened_map[s[0]][s[1]] == 1:
            return True
        else:
            return False

    def has_visited(self, s):
        if self.visted_map[s[0]][s[1]] == 0.0:
            return False
        else:
            return True

    def corners(self, x):
        return [x, x + np.array([1, 0]), x + np.array([1, 1]), x + np.array([0, 1])]

    def is_corner(self, s, x):
        if np.all(s == x) or np.all(s == np.array([1, 0])) or np.all(s == np.array([1, 1])) or np.all(s == np.array([0, 1])):
            return True
        else:
            return False

    def is_observed_obstacle(self, idx):
        if self.metric_grid_map[idx[0]][idx[1]] > 0.5:
            return True
        else:
            return False

    def draw(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        map_name: str = 'cost',
        obstacles: list[Obstacle] = [],
        enlarge_range: float = 0.0,
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)

        if map_name == 'cost':
            draw_map = self.g_map
        elif map_name == 'metric':
            draw_map = self.metric_grid_map
        elif map_name == 'local':
            draw_map = self.local_grid_map

        # Draw Obstacles
        for obstacle in obstacles:
            enl_obs = patches.Circle(xy=(obstacle.pos[0], obstacle.pos[1]), radius=obstacle.r + enlarge_range, fc='gray', ec='gray', zorder=-1.0)
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
