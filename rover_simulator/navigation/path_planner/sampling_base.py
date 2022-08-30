from __future__ import annotations
from platform import node
import random
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from rover_simulator.navigation.path_planner import PathPlanner
from rover_simulator.utils.utils import angle_to_range, GeoEllipse, cov_to_ellipse, ellipse_collision
from rover_simulator.utils.motion import covariance_transition, state_transition
from rover_simulator.utils.draw import set_fig_params, draw_obstacles, draw_start, draw_goal, sigma_ellipse


class RRT(PathPlanner):
    class Node():
        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y
            self.parent = None
            self.cost = 0.0
            self.path_x = []
            self.path_y = []

    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        explore_region: list = [[0, 20], [0, 20]],
        known_obstacles: list = [],
        enlarge_range: float = 0.0,
        expand_distance: float = 3.0,
        goal_sample_rate: float = 0.9,
        path_resolution: float = 0.5,
        cost_func: function = None
    ):
        super().__init__()
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.start_node = self.Node(start_pos[0], start_pos[1]) if start_pos is not None else None
        self.goal_node = self.Node(goal_pos[0], goal_pos[1]) if goal_pos is not None else None
        self.explore_x_min = explore_region[0][0]
        self.explore_x_max = explore_region[0][1]
        self.explore_y_min = explore_region[1][0]
        self.explore_y_max = explore_region[1][1]
        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_distance
        self.path_resolution = path_resolution
        if cost_func is None:
            self.cost = self.dist_nodes
        else:
            self.cost = cost_func
        self.node_list = []
        self.planned_path = []

        # self.known_obstacles = known_obstacles
        self.obstacle_list = [[obstacle.pos[0], obstacle.pos[1], obstacle.r + enlarge_range] for obstacle in known_obstacles]
        # obstacle_positions = [obstacle.pos for obstacle in known_obstacles] if not known_obstacles is None else None
        # self.obstacle_kdTree = cKDTree(obstacle_positions)
        self.name = "RRT"

    def dist_nodes(self, n1: Node, n2: Node) -> float:
        return math.hypot(n1.x - n2.x, n1.y - n2.y)

    def set_start(self, start_pos):
        self.start_node = self.Node(start_pos[0], start_pos[1])

    def set_goal(self, goal_pos):
        self.goal_node = self.Node(goal_pos[0], goal_pos[1])

    def calculate_path(self, max_iter=200, *kargs):
        self.planned_path = []
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        for _ in range(max_iter):
            rnd_node = self.sample_new_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal_node, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    self.planned_path = self.generate_final_course(len(self.node_list) - 1)
                    return [np.array([n.x, n.y]) for n in self.planned_path]
        return []

    def steer(self, from_node, to_node, extend_length=float("inf")):
        new_node = self.Node(from_node.x, from_node.y)
        d, theta = self.calc_distance_and_angle(new_node, to_node)
        new_node.path_x = [new_node.x]
        new_node.path_y = [new_node.y]
        if extend_length > d:
            extend_length = d

        n_expand = math.floor(extend_length / self.path_resolution)
        for _ in range(n_expand):
            new_node.x += self.path_resolution * math.cos(theta)
            new_node.y += self.path_resolution * math.sin(theta)
            new_node.path_x.append(new_node.x)
            new_node.path_y.append(new_node.y)

        d, _ = self.calc_distance_and_angle(new_node, to_node)
        if d <= self.path_resolution:
            new_node.path_x.append(to_node.x)
            new_node.path_y.append(to_node.y)
            new_node.x = to_node.x
            new_node.y = to_node.y
        new_node.parent = from_node
        new_node.cost = from_node.cost + self.cost(from_node, new_node)
        return new_node

    def sample_new_node(self) -> Node:
        if random.randint(0, 1) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.explore_x_min, self.explore_x_max),
                random.uniform(self.explore_y_min, self.explore_y_max)
            )
        else:  # goal point sampling
            rnd = self.Node(self.goal_node.x, self.goal_node.y)
        return rnd

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, node, obstacle_list):
        if node is None:
            return False
        for (ox, oy, size) in obstacle_list:
            dx_list = [ox - x for x in node.path_x]
            dy_list = [oy - y for y in node.path_y]
            d_list = [dx * dx + dy * dy for (dx, dy) in zip(dx_list, dy_list)]
            if min(d_list) <= size**2:
                return False  # collision
        return True  # safe

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.goal_node.x, y - self.goal_node.y)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        waypoints = [self.goal_node]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            waypoints.append(node)
            node = node.parent
        waypoints.append(node)
        waypoints.append(self.start_node)
        waypoints.reverse()
        return waypoints

    def save_log(self, src: str) -> None:
        num_node = len(self.node_list)
        node_pos = np.array([np.array([n.x, n.y]) for n in self.node_list])
        node_costs = np.array([n.cost for n in self.node_list])
        parents_idx = []
        node_path_x = []
        node_path_y = []
        for n in self.node_list:
            if n.parent is None:
                parents_idx.append(-1)
            else:
                parents_idx.append(self.node_list.index(n.parent))
            if len(n.path_x) == 0:
                node_path_x.append(np.array([np.inf, np.inf]))
            else:
                node_path_x.append(np.array([n.path_x[0], n.path_x[-1]]))
            if len(n.path_y) == 0:
                node_path_y.append(np.array([np.inf, np.inf]))
            else:
                node_path_y.append(np.array([n.path_y[0], n.path_y[-1]]))
        np.savez(src, num_node=num_node, node_pos=node_pos, node_path_x=node_path_x, node_path_y=node_path_y, node_costs=node_costs, parents_idx=parents_idx)

    def load_log(self, src: str) -> None:
        log = np.load(src)
        num_node = log['num_node']
        node_pos = log['node_pos']
        node_path_x = log['node_path_x']
        node_path_y = log['node_path_y']
        node_costs = log['node_costs']
        node_parents_idx = log['parents_idx']
        for i in range(num_node):
            self.node_list.append(self.Node(node_pos[i][0], node_pos[i][1]))
        for i in range(num_node):
            self.node_list[i].path_x = node_path_x[i] if np.sum(np.isinf(node_path_x[i])) == 0 else []
            self.node_list[i].path_y = node_path_y[i] if np.sum(np.isinf(node_path_y[i])) == 0 else []
            self.node_list[i].cost = node_costs[i]
            if node_parents_idx[i] == -1:
                self.node_list[i].parent = None
            else:
                self.node_list[i].parent = self.node_list[node_parents_idx[i]]
        self.planned_path = self.generate_final_course(len(self.node_list) - 1)

    def draw_path(self, ax):
        if self.planned_path is None:
            return
        for node in self.planned_path:
            if node.parent:
                ax.plot(node.path_x, node.path_y, c="red")

    def draw_nodes(self, ax):
        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, c="cyan")

    def draw(
            self,
            xlim: list[float], ylim: list[float],
            figsize: tuple[float, float] = (8, 8),
            obstacles: list = [],
            enlarge_range: float = 0.0,
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, enlarge_range)
        self.draw_nodes(ax)
        self.draw_path(ax)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        plt.show()


class RRTstar(RRT):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        explore_region: list = [[0, 20], [0, 20]],
        known_obstacles: list = [],
        enlarge_range: float = 0.0,
        expand_distance: float = 3.0,
        goal_sample_rate: float = 0.9,
        path_resolution: float = 0.5,
        connect_circle_dist: float = 50.0,
        search_until_max_iter: bool = False,
        cost_func: function = None
    ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(start_pos, goal_pos, explore_region, known_obstacles, enlarge_range, expand_distance, path_resolution, goal_sample_rate, cost_func)
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        self.name = "RRTstar"

    def calculate_path(self, max_iter=500, animation=False, *kargs):
        """
        rrt star path planning
        animation: flag for animation on or off .
        """
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        for i in range(max_iter):
            rnd = self.sample_new_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + self.cost(new_node, near_node)

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if animation:
                self.draw()

            # if not self.search_until_max_iter and new_node:  # if reaches goal
            #     last_idx = self.search_best_goal_node()
            #     if last_idx is not None:
            #         return self.generate_final_course(last_idx)

        last_idx = self.search_best_goal_node()
        if last_idx is not None:
            self.planned_path = self.generate_final_course(last_idx)
            return [np.array([n.x, n.y]) for n in self.planned_path]
        return []

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node
            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            # print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost
        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i
        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [
            (node.x - new_node.x)**2 + (node.y - new_node.y)**2
            for node in self.node_list
        ]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree
                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def load_log(self, src: str) -> None:
        super().load_log(src)
        last_idx = self.search_best_goal_node()
        if last_idx is not None:
            self.planned_path = self.generate_final_course(last_idx)


class ChanceConstrainedRRT(RRT):
    class Node(RRT.Node):
        def __init__(self, x: float, y: float, head: float = 0.0, cov: np.ndarray = np.diag([1e-10, 1e-10, 1e-10])) -> None:
            super().__init__(x, y)
            self.head = head
            self.cov = cov
            self.cost_lb = float('inf')         # Lower bound cost
            self.cost_ub = float('inf')  # Upper bound cost
            self.cost_fs = 0.0          # Cost from start
            self.childs = []

    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None, start_cov: np.ndarray = None,
        start_head: float = None, motion_noise_stds: dict = {"nn": 0.1, "no": 0.0001, "on": 0.013, "oo": 0.02},
        explore_region: list = [[0, 20], [0, 20]], known_obstacles: list = [], enlarge_range: float = 0, expand_distance: float = 3.0,
        cost_func: function = None, steer_func: function = None,
        goal_sample_rate: float = 0.3, goal_region=2.0,
        num_nearest_node: int = 6, p_safe: float = 0.99, k: float = 1.0
    ) -> None:
        super().__init__(start_pos, goal_pos, explore_region, known_obstacles, enlarge_range, expand_distance, goal_sample_rate, 0, cost_func)
        if start_pos is not None and goal_pos is not None and start_cov is not None:
            if start_head is None:
                start_head = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
            self.start_node = self.Node(start_pos[0], start_pos[1], start_head, start_cov)
            self.goal_node = self.Node(goal_pos[0], goal_pos[1], None, None)
        self.num_nearest_node = num_nearest_node
        self.p_safe = p_safe
        self.motion_noise_stds = motion_noise_stds
        self.sampled_pts = []
        self.control_inputs = self.select_inputs if steer_func is None else steer_func
        self.k = k
        self.goal_region = goal_region
        self.name = "CC-RRT"

    def calculate_path(self, max_time: float = 5.0, max_iter: int = 200, *kargs) -> list:
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        self.first_sample = True
        for _ in range(max_iter):
            self.expand_tree()

        self.planned_path = self.generate_final_course()
        return [np.array([n.x, n.y]) for n in self.planned_path]

    def generate_final_course(self):
        node_poses = [np.array([n.x, n.y]) for n in self.node_list]
        nodes_kdtree = cKDTree(node_poses)
        idxes = nodes_kdtree.query_ball_point(np.array([self.goal_node.x, self.goal_node.y]), r=self.goal_region)
        min_cost = float('inf')
        min_idx = None
        for idx in idxes:
            if min_cost > self.node_list[idx].cost_fs:
                min_cost = self.node_list[idx].cost_fs
                min_idx = idx
        if min_idx is None:
            planned_path = []
        else:
            node = self.node_list[min_idx]
            planned_path = [node]
            while node.parent is not None:
                planned_path.append(node.parent)
                node = node.parent
            planned_path.reverse()
        return planned_path

    def expand_tree(self) -> None:
        trg_node = self.sample_new_node()
        near_idxes = self.get_m_nearest_nodes_indexes(self.node_list, trg_node)
        pruned_idxes = []
        node_list = copy.copy(self.node_list)
        for idx in near_idxes:
            if idx in pruned_idxes:
                continue
            near_node = node_list[idx]
            new_nodes, _ = self.connect_to_target(near_node, trg_node)
            for new_node in new_nodes:
                new_node.cost_fs = new_node.parent.cost_fs + self.cost(new_node.parent, new_node)
                self.node_list.append(new_node)
                nodes_to_goal, reached_goal = self.connect_to_target(new_node, self.goal_node)
                if reached_goal is True:
                    ub_cost = 0.0
                    node = nodes_to_goal[-1]
                    while not node.parent is None:
                        p_node = node.parent
                        ub_cost += self.cost(p_node, node)
                        lb_cost = self.cost(self.goal_node, node)
                        if ub_cost < lb_cost:
                            pruned_idxes += self.prune_childs(node, node_list)
                        node = p_node

    def prune_childs(self, node: Node, node_list: list) -> None:
        pruned_idxes = []
        for child in node.childs:
            if child in self.node_list:
                idx = node_list.index(child)
                self.node_list.remove(child)
                child.parent.childs.remove(child)
                pruned_idxes.append(idx)
                pruned_idxes += self.prune_childs(child, node_list)
        return pruned_idxes

    def sample_new_node(self) -> Node:
        # if random.random() > self.goal_sample_rate and self.first_sample is False:
        rnd = self.Node(
            random.uniform(self.explore_x_min, self.explore_x_max),
            random.uniform(self.explore_y_min, self.explore_y_max)
        )
        # else:  # goal point sampling
        #     rnd = self.Node(self.goal_node.x, self.goal_node.y)
        # rnd = self.Node(self.goal_node.x, self.goal_node.y)
        # self.first_sample = False
        self.sampled_pts.append(rnd)
        return rnd

    def connect_to_target(self, node: Node, trg_node: Node) -> list[list[Node], bool]:
        new_nodes = []
        x_tk = np.array([node.x, node.y])
        p_tk = node.cov
        dist_to_trg = float('inf')
        prev_node = node
        while True:
            control_inputs = self.control_inputs(np.array([node.x, node.y, node.head]), np.array([trg_node.x, trg_node.y, trg_node.head]))
            x_tk, p_tk = self.steer(node, control_inputs)
            dist_to_trg = math.hypot(x_tk[0] - trg_node.x, x_tk[1] - trg_node.y)
            new_node = self.Node(x_tk[0], x_tk[1], x_tk[2], p_tk)
            if not self.is_safe(x_tk, p_tk):
                break
            if dist_to_trg <= 0.5 * self.expand_dis:
                # new_node.cost_fs = prev_node.cost_fs + self.cost(new_node, prev_node)
                self.connect_nodes(prev_node, new_node)
                new_node.path_x = [prev_node.x, new_node.x]
                new_node.path_y = [prev_node.y, new_node.y]
                new_node.cost_lb = self.cost(new_node, self.goal_node)
                new_nodes.append(new_node)
                break
            if math.hypot(new_node.x - prev_node.x, new_node.y - prev_node.y) > 1.0:
                # new_node.cost_fs = prev_node.cost_fs + self.cost(new_node, prev_node)
                self.connect_nodes(prev_node, new_node)
                new_node.path_x = [prev_node.x, new_node.x]
                new_node.path_y = [prev_node.y, new_node.y]
                new_node.cost_lb = self.cost(new_node, self.goal_node)
                new_nodes.append(new_node)
                prev_node = new_node
            node = new_node
        return new_nodes, (dist_to_trg < self.expand_dis and len(new_nodes) > 0)

    def connect_nodes(self, pn: Node, cn: Node) -> None:
        cn.parent = pn
        pn.childs.append(cn)

    def get_m_nearest_nodes_indexes(self, node_list: list[Node], rnd_node: Node) -> list:
        dist_costs = [self.dist_nodes(node, rnd_node) for node in node_list]    # rnd_nodeから各nodeへの距離
        fs_costs = [node.cost_fs for node in node_list]  # スタート地点から各nodeまでの距離

        # コスト重視
        # print(dist_costs)
        # print(fs_costs)
        costs = [2.0 * dist + self.k * fs for (dist, fs) in zip(dist_costs, fs_costs)]
        costs_sorted = sorted(costs)
        mininds = []
        for i, d in enumerate(costs_sorted):
            if i >= self.num_nearest_node:
                break
            mininds.append(costs.index(d))

        # コストと距離の中間
        # costs = [1.5 * dist + self.k * fs for (dist, fs) in zip(dist_costs, fs_costs)]
        # costs_sorted = sorted(costs)
        # mininds = []
        # for i, d in enumerate(costs_sorted):
        #     if i >= self.num_nearest_node / 2:
        #         break
        #     mininds.append(costs.index(d))

        # 距離重視
        # costs = [dist + 0.0 * fs for (dist, fs) in zip(dist_costs, fs_costs)]
        # costs_sorted = sorted(costs)
        # mininds = []
        # for i, d in enumerate(costs_sorted):
            # if i >= self.num_nearest_node / 3:
            #     break
            # mininds.append(costs.index(d))
        return mininds

    def select_inputs(self, cur_pose, trg_pose) -> list:
        L = 1.0
        v = 1.0
        theta = np.arctan2(trg_pose[1] - cur_pose[1], trg_pose[0] - cur_pose[0]) - cur_pose[2]
        theta = angle_to_range(theta)
        w = 2 * v * np.sin(theta) / L
        return v, w

    def steer(self, from_node: Node, control_inputs: list[float, float]) -> Node:
        dt = 0.5
        prev_pose = np.array([from_node.x, from_node.y, from_node.head])
        nu, omega = control_inputs
        if abs(omega) < 1e-5:
            omega = 1e-5  # 値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる
        control_inputs = [nu, omega]
        new_pose = state_transition(prev_pose, control_inputs, dt)
        new_cov = covariance_transition(prev_pose, from_node.cov, self.motion_noise_stds, control_inputs, dt)
        return new_pose, new_cov

    def is_safe(self, x: np.ndarray, cov: np.ndarray) -> bool:
        if x[0] < 0 or x[0] > 20 or x[1] < 0 or x[1] > 20:
            return False
        e1 = cov_to_ellipse(x[0:2], cov[0:2, 0:2], 3)
        for obs in self.obstacle_list:
            e2 = GeoEllipse(obs[0], obs[1], 0.0, obs[2], obs[2])
            if ellipse_collision(e1, e2):
                return False
        return True

    def save_log(self, src: str) -> None:
        num_node = len(self.node_list)
        node_pos = np.array([np.array([n.x, n.y]) for n in self.node_list])
        node_head = np.array([n.head for n in self.node_list])
        node_cov = np.array([n.cov for n in self.node_list])
        node_lb_costs = np.array([n.cost_lb for n in self.node_list])
        node_ub_costs = np.array([n.cost_ub for n in self.node_list])
        node_fs_costs = np.array([n.cost_fs for n in self.node_list])
        parents_idx = []
        node_path_x = []
        node_path_y = []
        for n in self.node_list:
            if n.parent is None:
                parents_idx.append(-1)
            else:
                parents_idx.append(self.node_list.index(n.parent))
            if len(n.path_x) == 0:
                node_path_x.append(np.array([np.inf, np.inf]))
            else:
                node_path_x.append(np.array([n.path_x[0], n.path_x[-1]]))
            if len(n.path_y) == 0:
                node_path_y.append(np.array([np.inf, np.inf]))
            else:
                node_path_y.append(np.array([n.path_y[0], n.path_y[-1]]))
        np.savez(src, num_node=num_node, node_pos=node_pos, node_head=node_head, node_cov=node_cov, node_lb_costs=node_lb_costs, node_ub_costs=node_ub_costs, node_fs_costs=node_fs_costs, parents_idx=parents_idx, node_path_x=node_path_x, node_path_y=node_path_y)

    def load_log(self, src: str) -> None:
        log = np.load(src)
        num_node = log['num_node']
        node_pos = log['node_pos']
        node_path_x = log['node_path_x']
        node_path_y = log['node_path_y']
        node_head = log['node_head']
        node_cov = log['node_cov']
        node_lb_costs = log['node_lb_costs']
        node_ub_costs = log['node_ub_costs']
        node_fs_costs = log['node_fs_costs']
        node_parents_idx = log['parents_idx']
        for i in range(num_node):
            self.node_list.append(self.Node(node_pos[i][0], node_pos[i][1], node_head[i], node_cov[i]))
        for i in range(num_node):
            self.node_list[i].path_x = node_path_x[i] if np.sum(np.isinf(node_path_x[i])) == 0 else []
            self.node_list[i].path_y = node_path_y[i] if np.sum(np.isinf(node_path_y[i])) == 0 else []
            self.node_list[i].cost_lb = node_lb_costs[i]
            self.node_list[i].cost_ub = node_ub_costs[i]
            self.node_list[i].cost_fs = node_fs_costs[i]
            if node_parents_idx[i] == -1:
                self.node_list[i].parent = None
            else:
                self.node_list[i].parent = self.node_list[node_parents_idx[i]]
        self.planned_path = self.generate_final_course()

    def draw(
            self,
            xlim: list[float], ylim: list[float],
            figsize: tuple[float, float] = (8, 8),
            obstacles: list = [],
            enlarge_range: float = 0.0,
            draw_ellipse: bool = True,
            draw_result_only: bool = False
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, enlarge_range)

        # Draw all nodes
        if draw_result_only is False:
            # self.draw_node_childs(ax, self.start_node, draw_ellipse)
            for n in self.node_list:
                if n.parent:
                    ax.plot(n.path_x, n.path_y, color="cyan")
                    if draw_ellipse is True:
                        p = np.array([n.x, n.y, n.head])
                        e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3, "blue")
                        ax.add_patch(e)

        # Draw path from start to goal
        for n in self.planned_path:
            if n.parent:
                ax.plot(n.path_x, n.path_y, color="red", zorder=10)
                if draw_ellipse is True and draw_result_only is True:
                    p = np.array([n.x, n.y, n.head])
                    e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3)
                    ax.add_patch(e)

        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)

        plt.show()

    def draw_node_childs(self, ax, n, draw_ellipse, i=0):
        for c_n in n.childs:
            ax.plot(c_n.path_x, c_n.path_y, color="cyan")
            if draw_ellipse is True:
                p = np.array([c_n.x, c_n.y, c_n.head])
                e = sigma_ellipse(p[0:2], c_n.cov[0:2, 0:2], 3, "blue")
                ax.add_patch(e)
            self.draw_node_childs(ax, c_n, draw_ellipse, i)
