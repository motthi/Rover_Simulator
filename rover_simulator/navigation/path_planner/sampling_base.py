from __future__ import annotations
import random
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from rover_simulator.navigation.localizer import KalmanFilter
from rover_simulator.navigation.path_planner import PathPlanner
from rover_simulator.utils import angle_to_range, sigma_ellipse, GeoEllipse, cov_to_ellipse, ellipse_collision


class RRT(PathPlanner):
    class Node():
        def __init__(self, x, y) -> None:
            self.x = x
            self.y = y
            self.parent = None

    def __init__(
        self,
        start_pos=None, goal_pos=None,
        explore_region=[[0, 20], [0, 20]],
        known_obstacles=[],
        enlarge_range=0.0,
        expand_distance=3.0,
        goal_sample_rate=0.9,
        path_resolution=0.5
    ):
        super().__init__()
        self.start_node = self.Node(start_pos[0], start_pos[1]) if start_pos is not None else None
        self.goal_node = self.Node(goal_pos[0], goal_pos[1]) if goal_pos is not None else None
        self.explore_x_min = explore_region[0][0]
        self.explore_x_max = explore_region[0][1]
        self.explore_y_min = explore_region[1][0]
        self.explore_y_max = explore_region[1][1]
        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_distance
        self.path_resolution = path_resolution

        # self.known_obstacles = known_obstacles
        self.obstacle_list = [[obstacle.pos[0], obstacle.pos[1], obstacle.r + enlarge_range] for obstacle in known_obstacles]
        # obstacle_positions = [obstacle.pos for obstacle in known_obstacles] if not known_obstacles is None else None
        # self.obstacle_kdTree = cKDTree(obstacle_positions)

        self.name = "RRT"

    def set_start(self, start_pos):
        self.start_node = self.Node(start_pos[0], start_pos[1])

    def set_goal(self, goal_pos):
        self.goal_node = self.Node(goal_pos[0], goal_pos[1])

    def calculate_path(self, max_iter=200, *kargs):
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
                    return self.generate_final_course(len(self.node_list) - 1)
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
        dx = x - self.goal_node.x
        dy = y - self.goal_node.y
        return math.hypot(dx, dy)

    def calc_distance_and_angle(self, from_node, to_node):
        dx = to_node.x - from_node.x
        dy = to_node.y - from_node.y
        d = math.hypot(dx, dy)
        theta = math.atan2(dy, dx)
        return d, theta

    def generate_final_course(self, goal_ind):
        waypoints = [np.array([self.goal_node.x, self.goal_node.y])]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            waypoints.append([node.x, node.y])
            node = node.parent
        waypoints.append(np.array([node.x, node.y]))
        waypoints.append(np.array([self.start_node.x, self.start_node.y]))
        waypoints.reverse()
        return waypoints

    def draw_graph(self, rnd=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ox, oy, size in self.obstacle_list:
            self.plot_circle(ox, oy, size)

        plt.plot(self.start_node.x, self.start_node.y, "xr")
        plt.plot(self.goal_node.x, self.goal_node.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 22, -2, 22])
        plt.grid(True)
        plt.pause(0.01)

    def plot_circle(self, x, y, size, color="-b"):  # pragma: no cover
        deg = list(range(0, 360, 5))
        deg.append(0)
        xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
        yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
        plt.plot(xl, yl, color)


class RRTstar(RRT):
    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(
        self,
        start_pos=None, goal_pos=None,
        explore_region=[[0, 20], [0, 20]],
        known_obstacles=[],
        enlarge_range=0.0,
        expand_distance=3.0,
        goal_sample_rate=0.9,
        path_resolution=0.5,
        connect_circle_dist=50.0,
        search_until_max_iter=False
    ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(start_pos, goal_pos, explore_region, known_obstacles, enlarge_range, expand_distance, path_resolution, goal_sample_rate)
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
            # print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.sample_new_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd, self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + math.hypot(new_node.x - near_node.x, new_node.y - near_node.y)

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
                self.draw_graph(rnd)

            # if not self.search_until_max_iter and new_node:  # if reaches goal
            #     last_index = self.search_best_goal_node()
            #     if last_index is not None:
            #         print(i)
            #         return self.generate_final_course(last_index)

        # print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index is not None:
            return self.generate_final_course(last_index)

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


class ChanceConstrainedRRT(RRT):
    class Node():
        def __init__(self, x: float, y: float, head: float, cov: np.ndarray) -> None:
            self.x = x
            self.y = y
            self.head = head
            self.cov = cov
            self.parent = None
            self.cost_lb = None         # Lower bound cost
            self.cost_ub = float('inf')  # Upper bound cost
            self.cost_fs = 0.0          # Cost from start

        def set_lower_bound_cost(self, goal_pos) -> None:
            self.cost_lb = np.linalg.norm(np.array(self.x, self.y) - goal_pos)

        def set_upper_bound_cost(self) -> None:
            self.cost_ub = 0.0

    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None, start_cov: np.ndarray = None, start_head: float = None,
        explore_region: list = [[0, 20], [0, 20]], known_obstacles: list = [], enlarge_range: float = 0, expand_distance: float = 3.0,
        goal_sample_rate: float = 0.3, path_resolution: float = 0.5,
        num_nearest_node: int = 10, p_safe: float = 0.99
    ) -> None:
        if start_pos is not None and goal_pos is not None and start_cov is not None:
            if start_head is None:
                start_head = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
            self.start_node = self.Node(start_pos[0], start_pos[1], start_head, start_cov)
            self.goal_node = self.Node(goal_pos[0], goal_pos[1], None, None)
        else:
            self.start_node = None
            self.goal_node = None
        self.explore_x_min = explore_region[0][0]
        self.explore_x_max = explore_region[0][1]
        self.explore_y_min = explore_region[1][0]
        self.explore_y_max = explore_region[1][1]
        self.goal_sample_rate = goal_sample_rate
        self.expand_dis = expand_distance
        self.path_resolution = path_resolution
        self.num_nearest_node = num_nearest_node
        self.p_safe = p_safe

        self.known_obstacles = known_obstacles
        self.obstacle_list = [[obstacle.pos[0], obstacle.pos[1], obstacle.r + enlarge_range] for obstacle in known_obstacles]
        # obstacle_positions = [obstacle.pos for obstacle in known_obstacles] if not known_obstacles is None else None
        # self.obstacle_kdTree = cKDTree(obstacle_positions)
        self.name = "CC-RRT"

    def calculate_path(self, max_iter: int = 200, *kargs) -> list:
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        import time
        s_time = time.time()
        while time.time() - s_time < 5.0:
            self.expand_tree(max_iter)

        nearest_idxes = self.get_m_nearest_nodes_indexes(self.node_list, self.goal_node)
        costs = []
        for idx in nearest_idxes:
            cost = 0.0
            node = self.node_list[idx]
            if np.linalg.norm([node.x - self.goal_node.x, node.y - self.goal_node.y]) > self.expand_dis * 3:
                # ゴールからある程度離れた場合は除外する
                continue
            while node.parent is not None:
                pnode = node.parent
                cost += np.linalg.norm([node.x - pnode.x, node.y - pnode.y])
                node = pnode
            if pnode.x != self.start_node.x or pnode.y != self.start_node.y:
                cost = float('inf')
            costs.append(cost)
        idx = costs.index(min(costs))
        node = self.node_list[nearest_idxes[idx]]
        planned_path = [node]
        while node.parent is not None:
            planned_path.append(node.parent)
            node = node.parent
        planned_path.reverse()
        self.planned_path = planned_path
        return self.planned_path

    def expand_tree(self, max_iter: int) -> None:
        for _ in range(max_iter):
            trg_node = self.sample_new_node()
            near_idxes = self.get_m_nearest_nodes_indexes(self.node_list, trg_node)
            for idx in near_idxes:
                near_node = self.node_list[idx]
                new_nodes, _ = self.connect_to_target(near_node, trg_node)
                for new_node in new_nodes:
                    self.node_list.append(new_node)
                    nodes_to_goal, reached_goal = self.connect_to_target(new_node, self.goal_node)
                    if reached_goal is True:
                        ub_cost = 0.0
                        node = nodes_to_goal.reverse()[-1]
                        while not node.parent is None:
                            parent_node = node.parent
                            ub_cost += np.linalg.norm([parent_node.x - node.x, parent_node.y - node.y])
                            if ub_cost > node.cost_ub:
                                self.node_list.pop(-1)  # Must prune portions of the tree
                                break
                            node.cost_ub = ub_cost
                            node.cost_fs = parent_node.cost_fs + np.linalg.norm([node.x - parent_node.x, node.y - parent_node.y])
                            node = parent_node

    def sample_new_node(self) -> Node:
        if random.random() > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.explore_x_min, self.explore_x_max),
                random.uniform(self.explore_y_min, self.explore_y_max),
                None, None
            )
        else:  # goal point sampling
            rnd = self.Node(self.goal_node.x, self.goal_node.y, None, None)
        return rnd

    def connect_to_target(self, node: Node, trg_node: Node) -> list[list[Node], bool]:
        new_nodes = []
        x_tk = np.array([node.x, node.y])
        p_tk = node.cov
        dist_to_trg = float('inf')
        prev_node = node
        while True:
            control_inputs = self.select_inputs(node, trg_node)
            new_node = self.simulate_one_step(node, control_inputs)
            x_tk = np.array([new_node.x, new_node.y])
            p_tk = new_node.cov
            dist_to_trg = np.linalg.norm([new_node.x - trg_node.x, new_node.y - trg_node.y])
            if not self.is_safe(x_tk, p_tk) or dist_to_trg <= self.expand_dis:
                break
            if np.linalg.norm([new_node.x - prev_node.x, new_node.y - prev_node.y]) > 1.0:
                new_nodes.append(new_node)
                new_node.parent = prev_node
                prev_node = new_node
            node = new_node
        return new_nodes, dist_to_trg < self.expand_dis

    def get_m_nearest_nodes_indexes(self, node_list: list[Node], rnd_node: Node) -> list:
        dist_costs = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]    # rnd_nodeから各nodeへの距離
        fs_costs = [node.cost_fs for node in node_list]  # スタート地点から各nodeまでの距離
        costs = [d + fs for (d, fs) in zip(dist_costs, fs_costs)]
        costs_sorted = sorted(costs)
        mininds = []
        for i, d in enumerate(costs_sorted):
            if i >= self.num_nearest_node:
                break
            mininds.append(costs.index(d))
        return mininds

    def select_inputs(self, node: Node, trg_node: Node) -> list:
        L = 1.0
        v = 1.0
        theta = np.arctan2(trg_node.y - node.y, trg_node.x - node.x) - node.head
        theta = angle_to_range(theta)
        w = 2 * v * np.sin(theta) / L
        return v, w

    def simulate_one_step(self, from_node: Node, control_inputs: list[float, float]) -> Node:
        prev_pose = np.array([from_node.x, from_node.y, from_node.head])
        kf = KalmanFilter(prev_pose, from_node.cov)
        new_pose = kf.estimate_pose(prev_pose, control_inputs, 0.5)
        new_node = self.Node(new_pose[0], new_pose[1], new_pose[2], kf.belief.cov)
        new_node.cost_fs = from_node.cost_fs + np.linalg.norm([new_node.x - from_node.x, new_node.y - from_node.y])
        new_node.parent = from_node
        return new_node

    def is_safe(self, x: np.ndarray, cov: np.ndarray) -> bool:
        e1 = cov_to_ellipse(x[0:2], cov[0:2, 0:2], 3)
        for obs in self.obstacle_list:
            e2 = GeoEllipse(obs[0], obs[1], 0.0, obs[2], obs[2])
            if ellipse_collision(e1, e2):
                return False
        return True

    def draw(self, figsize=(8, 8), draw_ellipse=True, draw_result_only=False) -> None:
        self.fig = plt.figure(figsize=figsize)
        ax = self.fig.add_subplot(111)
        ax.set_aspect('equal')
        ax.set_xlabel("X [m]", fontsize=10)
        ax.set_ylabel("Y [m]", fontsize=10)

        # Draw Obstacle Regions
        for obstacle in self.obstacle_list:
            obs = patches.Circle(xy=(obstacle[0], obstacle[1]), radius=obstacle[2], fc='black', ec='black')
            ax.add_patch(obs)

        if draw_result_only is False:
            # すべての経路候補を表示
            for n in self.node_list:
                n_ = n.parent
                if n_ is not None:
                    ax.plot([n.x, n_.x], [n.y, n_.y], color="cyan")
                ax.scatter([n.x], [n.y], color="b", s=3)
                if draw_ellipse is True:
                    p = np.array([n.x, n.y, n.head])
                    if not self.is_safe(p, n.cov):
                        c = "red"
                    else:
                        c = "blue"
                    e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3, c)
                    ax.add_patch(e)
            for i in range(len(self.planned_path) - 1):
                n = self.planned_path[i]
                n_ = self.planned_path[i + 1]
                ax.plot([n.x, n_.x], [n.y, n_.y], color="red")
        else:
            # 最終的に得られた経路のみを表示
            for i in range(len(self.planned_path) - 1):
                n = self.planned_path[i]
                n_ = self.planned_path[i + 1]
                ax.plot([n.x, n_.x], [n.y, n_.y], color="red")
                if draw_ellipse is True:
                    p = np.array([n.x, n.y, n.head])
                    e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3)
                    ax.add_patch(e)
        plt.show()
