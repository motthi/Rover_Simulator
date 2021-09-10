import math
import numpy as np
import copy
from rover_simulator.core import Mapper
from rover_simulator.navigation.path_planner import PathPlanner, PathNotFoundError


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
        return (np.array(pose[0:2]) // self.grid_width + np.array([self.grid_width, self.grid_width]) / 2).astype(np.int32)

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

    def isOutOfBounds(self, idx):
        if np.any(idx >= self.grid_num) or np.any(idx < [0, 0]):
            return True
        else:
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

        self.open_list = []
        self.closed_list = []
        self.resultPath = []
        self.takenPath = []

        self.name = "Dijkstra"

        if map is not None:
            cnt = 0
            for u, _ in np.ndenumerate(self.cost_map):
                self.id_map[u[0]][u[1]] = cnt
                if self.isStart(u):
                    self.open_list.append([cnt, 0, 0])
                cnt += 1

    def set_map(self, mapper: Mapper):
        self.grid_width = mapper.grid_width
        self.grid_num = np.array(mapper.map.shape)
        self.grid_map = copy.copy(mapper.map)
        self.cost_map = np.full(self.grid_num, 0.0, dtype=float)
        self.id_map = np.full(self.grid_num, 0, dtype=np.int32)
        self.parent_id_map = np.full(self.grid_num, 0, dtype=np.int32)
        cnt = 0
        for u, _ in np.ndenumerate(self.cost_map):
            self.id_map[u[0]][u[1]] = cnt
            if self.isStart(u):
                self.open_list.append([cnt, 0, 0])
            cnt += 1

    def calculate_path(self, *args):
        while not self.isClosed(self.goal_idx):
            _, _ = self.expandGrid()
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
        self.open_list.remove([grid_id, cost_f, cost_g])  # オープンリストから削除
        self.closed_list.append([grid_id, cost_f, cost_g])  # クローズドリストに追加
        self.calculateCost(idx, cost_g)  # コストの計算
        return idx, cost_f

    def calculateCost(self, idx, cost_g):
        for n in self.listFreeNeigbor(idx):
            evaluation_f = cost_g + self.cost(n) + self.c(n, idx)
            if self.isOpened(n):
                its_idx, its_cost_f, its_cost_g = self.open_list[[val[0] for val in self.open_list].index(self.id(n))]
                if its_cost_f > evaluation_f:
                    self.open_list.remove([its_idx, its_cost_f, its_cost_g])
                else:
                    continue
            elif self.isClosed(n):
                its_idx, its_cost_f, its_cost_g = self.closed_list[[val[0] for val in self.closed_list].index(self.id(n))]
                if its_cost_f > evaluation_f:
                    self.closed_list.remove([its_idx, its_cost_f, its_cost_g])
                else:
                    continue
            self.parent_id_map[n[0]][n[1]] = self.id(idx)
            self.open_list.append([self.id(n), evaluation_f, evaluation_f])

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
            if not self.isOutOfBounds(neigbor_idx) and not self.isObstacle(neigbor_idx) and not self.isObstacleDiagonal(idx, neigbor_idx):
                neigbor_indice.append(neigbor_idx)
        return neigbor_indice

    def isOpened(self, u):
        return self.id(u) in [val[0] for val in self.open_list]

    def isClosed(self, u):
        return self.id(u) in [val[0] for val in self.closed_list]

    def id(self, u):
        return self.id_map[u[0]][u[1]]

    def cost(self, u):
        return self.cost_map[u[0]][u[1]]

    def b(self, u):
        return self.parent_id_map[int(u[0])][int(u[1])]

    def c(self, u, v):
        return np.linalg.norm(u - v)


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
        for n in self.listFreeNeigbor(idx):
            evaluation_f = cost_g + self.cost(n) + self.c(n, idx) + self.__h(n)  # 評価を計算
            if self.isOpened(n):  # オープンリストに含まれているか
                its_index, its_cost_f, its_cost_g = self.open_list[[val[0] for val in self.open_list].index(self.id(n))]
                if its_cost_f > evaluation_f:  # 評価が更新されなければ繰り返しを戻す
                    self.open_list.remove([its_index, its_cost_f, its_cost_g])
                else:
                    continue
            elif self.isClosed(n):  # クローズドリストに含まれているか
                its_index, its_cost_f, its_cost_g = self.closed_list[[val[0] for val in self.closed_list].index(self.id(n))]
                if its_cost_f > evaluation_f:
                    self.closed_list.remove([its_index, its_cost_f, its_cost_g])
                else:
                    continue
            self.parent_id_map[n[0]][n[1]] = self.id(idx)
            self.open_list.append([self.id(n), evaluation_f, evaluation_f - self.__h(n)])

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
            self.local_grid_map = np.full(mapper.shape, 0.5, dtype=float)  # Local Map is Obstacle Occupancy Grid Map
            self.metric_grid_map = np.full(self.grid_cost_num, -1.0, dtype=np.float)  # Metric Map shows wheter the grid is observed, -1: Unobserved, 0: Free, 1: Obstacles

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

        self.local_grid_map = copy.copy(mapper.map)  # センシングによって構築したマップ
        self.local_grid_map[self.start_idx[0]][self.start_idx[1]] = 0.0
        self.metric_grid_map = np.full(self.grid_num, -1.0, dtype=np.float)  # 経路計画で使用するマップ
        self.metric_grid_map[self.start_idx[0]][self.start_idx[1]] = 0

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
        self.__computeShortestPath(self.start_idx)
        waypoints = self.get_path(self.current_idx)
        return waypoints

    def update_path(self, pose: np.ndarray, mapper: Mapper):
        self.current_idx = self.poseToIndex(pose)

        self.newObstacles = []
        self.newFrees = []

        for u, _ in mapper.observed_grids:
            if self.isOutOfBounds(u):
                continue
            prev_occ = copy.copy(self.local_grid_map[u[0]][u[1]])
            self.local_grid_map[u[0]][u[1]] = copy.copy(mapper.map[u[0]][u[1]])
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
        self.local_grid_map[self.current_idx[0]][self.current_idx[1]] = 0.0
        self.metric_grid_map[self.current_idx[0]][self.current_idx[1]] = 0.0

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
                self.__updateVertex(u)

            self.__computeShortestPath(self.current_idx)

        waypoints = self.get_path(self.current_idx)
        return waypoints

    def get_path(self, idx):
        self.pathToTake = [idx]
        last_cost = float('inf')
        while not self.isGoal(idx):
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

    def __computeShortestPath(self, index):
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
                    self.__updateVertex(s)
            else:
                g_old = self.g(u)
                self.g_map[u[0]][u[1]] = float('inf')
                for s in self.neigborGrids(u) + [u]:
                    if math.isclose(self.rhs(s), self.__c(s, u) + g_old):
                        if not self.isGoal(s):
                            self.rhs_map[s[0]][s[1]] = self.__getMinRhs(s)
                    self.__updateVertex(s)
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

    def __updateVertex(self, u):
        u_flag = list(u) in [row[0] for row in self.U]
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

    def __uRemove(self, u):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U.remove([list(u), self.U[idx][1]])

    def __uUpdate(self, u, u_num_new):
        U_row = [row[0] for row in self.U]
        idx = U_row.index(list(u))
        self.U[idx][1] = u_num_new

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
