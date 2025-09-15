import sys
import copy
import random
import math
import numpy as np
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.axes import Axes
from scipy.spatial import cKDTree
from rover_simulator.core import Obstacle, PathPlanner, Mapper
from rover_simulator.utils.utils import set_angle_into_range
from rover_simulator.utils.cmotion.cmotion import state_transition, covariance_transition, prob_collision
from rover_simulator.utils.draw import set_fig_params, draw_obstacles, draw_start, draw_goal, sigma_ellipse


if 'google.colab' in sys.modules:
    from tqdm.notebook import tqdm  # Google Colaboratory
elif 'ipykernel' in sys.modules:
    from tqdm.notebook import tqdm    # Jupyter Notebook
else:
    from tqdm import tqdm    # ipython, python script, ...


def rotation_matrix(t):
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


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
        known_obstacles: list[Obstacle] = [],
        expand_dist: float = 0.0,
        expand_distance: float = 3.0,
        goal_sample_rate: float = 0.9,
        path_resolution: float = 0.5,
        cost_func=None
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
        self.expand_dist = expand_dist
        if cost_func is None:
            self.cost = self.dist_nodes
        else:
            self.cost = cost_func
        self.node_list = []
        self.planned_path = []

        self.known_obstacles = known_obstacles
        self.obstacle_list: list[Obstacle] = known_obstacles
        # obstacle_positions = [obstacle.pos for obstacle in known_obstacles] if not known_obstacles is None else None
        # self.obstacle_kdTree = cKDTree(obstacle_positions)
        self.name = "RRT"

    def dist_nodes(self, pose1: np.ndarray, pose2: np.ndarray) -> float:
        return math.hypot(pose1[1] - pose2[1], pose1[0] - pose2[0])

    def set_start(self, start_pos):
        self.start_node = self.Node(start_pos[0], start_pos[1])

    def set_goal(self, goal_pos):
        self.goal_node = self.Node(goal_pos[0], goal_pos[1])

    def set_map(self, mapper: Mapper = None):
        return None

    def calculate_path(self, max_iter=200, *kargs):
        self.planned_path = []
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        for _ in range(max_iter):
            rnd_node = self.sample_new_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd_node)

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)
            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.calc_dist_to_goal(self.node_list[-1].x, self.node_list[-1].y) <= self.expand_dis:
                final_node = self.steer(self.node_list[-1], self.goal_node, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    self.planned_path = self.generate_final_course(len(self.node_list) - 1)
                    return np.array([[n.x, n.y] for n in self.planned_path])
        return []

    def steer(self, from_node: Node, to_node: Node, extend_length: float = float("inf")):
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
        new_node.cost = from_node.cost + self.cost(self.to_pose(from_node), self.to_pose(new_node))
        return new_node

    def sample_new_node(self) -> Node:
        if random.random() > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.explore_x_min, self.explore_x_max),
                random.uniform(self.explore_y_min, self.explore_y_max)
            )
        else:  # goal point sampling
            rnd = self.Node(self.goal_node.x, self.goal_node.y)
        return rnd

    def get_nearest_node(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        idx = dlist.index(min(dlist))
        return node_list[idx]

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [(node.x - rnd_node.x)**2 + (node.y - rnd_node.y)**2 for node in node_list]
        minind = dlist.index(min(dlist))
        return minind

    def check_collision(self, node: Node, obstacle_list: list[Obstacle]):
        if node is None:
            return False
        for obs in obstacle_list:
            for x1, y1, x2, y2 in zip(node.path_x[:-1], node.path_y[:-1], node.path_x[1:], node.path_y[1:]):
                if obs.check_collision_line(np.array([x1, y1]), np.array([x2, y2]), self.expand_dist):
                    return False  # collision
        return True  # safe

    def calc_dist_to_goal(self, x, y):
        return math.hypot(x - self.goal_node.x, y - self.goal_node.y)

    def calc_distance_and_angle(self, from_node, to_node):
        return self.distance(from_node, to_node), self.node_heading_angle(from_node, to_node)

    def distance(self, n1: Node, n2: Node):
        return math.hypot(n2.x - n1.x, n2.y - n1.y)

    def node_heading_angle(self, from_node: Node, to_node: Node):
        return math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)

    def generate_final_course(self, goal_ind):
        node = self.node_list[goal_ind]
        self.goal_node.path_x = [self.goal_node.x, node.x]
        self.goal_node.path_y = [self.goal_node.y, node.y]
        self.goal_node.parent = node
        waypoints = [self.goal_node]
        while node.parent is not None:
            waypoints.append(node)
            node = node.parent
        waypoints.append(node)
        waypoints.append(self.start_node)
        waypoints.reverse()
        return waypoints

    def to_pose(self, n: Node):
        return np.array([n.x, n.y])

    def save_log(self, src: str) -> None:
        nodes_npzf = self.to_npz_format(self.node_list)
        np.savez(
            src,
            num_node=nodes_npzf['num_node'],
            node_pos=nodes_npzf['node_pos'],
            num_path=nodes_npzf['num_path'],
            node_path_x=nodes_npzf['node_path_x'],
            node_path_y=nodes_npzf['node_path_y'],
            node_costs=nodes_npzf['node_costs'],
            parents_idx=nodes_npzf['parents_idx']
        )

    def to_npz_format(self, node_list: list) -> dict:
        num_node = len(node_list)
        node_pos = np.array([np.array([n.x, n.y]) for n in self.node_list])
        node_costs = np.array([n.cost for n in self.node_list])
        parents_idx = []
        num_path = []
        node_path_x = []
        node_path_y = []
        for n in self.node_list:
            if n.parent is None:
                parents_idx.append(-1)
            else:
                parents_idx.append(self.node_list.index(n.parent))
            num_path.append(len(n.path_x))
            node_path_x += n.path_x
            node_path_y += n.path_y
        return {
            'num_node': num_node,
            'node_pos': node_pos,
            'node_path_x': node_path_x,
            'node_path_y': node_path_y,
            'node_costs': node_costs,
            'parents_idx': parents_idx,
            'num_path': num_path
        }

    def load_log(self, src: str) -> None:
        log = np.load(src)
        num_node = log['num_node']
        num_path = log['num_path']
        node_pos = log['node_pos']
        node_path_x = log['node_path_x']
        node_path_y = log['node_path_y']
        node_costs = log['node_costs']
        node_parents_idx = log['parents_idx']
        path_cnt = 0
        for i in range(num_node):
            n = self.Node(node_pos[i][0], node_pos[i][1])
            n.path_x = node_path_x[path_cnt:path_cnt + num_path[i]]
            n.path_y = node_path_y[path_cnt:path_cnt + num_path[i]]
            n.cost = node_costs[i]
            self.node_list.append(n)
            path_cnt += num_path[i]

        # Set parent
        for i in range(num_node):
            if node_parents_idx[i] == -1:
                self.node_list[i].parent = None
            else:
                self.node_list[i].parent = self.node_list[node_parents_idx[i]]
        self.planned_path = self.generate_final_course(len(self.node_list) - 1)

    def draw_path(self, ax, c="red"):
        if self.planned_path is None:
            return
        for node in self.planned_path:
            if node.parent:
                ax.plot(node.path_x, node.path_y, c=c)

    def draw_nodes(self, ax):
        for node in self.node_list:
            if node.parent:
                ax.plot(node.path_x, node.path_y, c="cyan")

    def draw(
            self,
            xlim: list[float] = None, ylim: list[float] = None,
            figsize: tuple[float, float] = (8, 8),
            obstacles: list = [],
            expand_dist: float = 0.0,
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)
        self.draw_nodes(ax)
        self.draw_path(ax)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        plt.show()

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        expand_dist: float = 0.0,
        end_step=None,
        axes_setting: list = [0.09, 0.07, 0.85, 0.9]
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim, axes_setting)
        draw_obstacles(ax, self.known_obstacles, expand_dist)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        elems = []

        # Start Animation
        if end_step:
            animation_len = end_step
        else:
            animation_len = len(self.node_list) + 20
        pbar = tqdm(total=animation_len)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, animation_len, interval=100, repeat=False, fargs=(ax, elems, xlim, ylim, pbar),
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )
        if i <= len(self.node_list) - 1:
            node = self.node_list[i]
            ax.plot(node.path_x, node.path_y, c="cyan")
        elif i == len(self.node_list):
            self.draw_path(ax)
        pbar.update(1)


class RRTstar(RRT):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None,
        explore_region: list = [[0, 20], [0, 20]],
        known_obstacles: list = [],
        expand_dist: float = 0.0,   # Obstacle expand distance
        expand_distance: float = 3.0,   # Node expand distance @todo rename
        goal_sample_rate: float = 0.9,
        path_resolution: float = 0.5,
        connect_circle_dist: float = 50.0,
        search_until_max_iter: bool = False,
        cost_func=None
    ):
        """
        Setting Parameter
        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]
        """
        super().__init__(start_pos, goal_pos, explore_region, known_obstacles, expand_dist, expand_distance, path_resolution, goal_sample_rate, cost_func)
        self.connect_circle_dist = connect_circle_dist
        self.search_until_max_iter = search_until_max_iter
        self.nodes_history = []
        self.path_history = []
        self.best_path_history = []
        self.name = "RRTstar"

    def calculate_path(self, max_iter=500, log_history=False, *kargs):
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.reached_goal = False
        self.node_list = [self.start_node]
        if log_history:
            self.nodes_history.append(copy.deepcopy(self.node_list))
            self.best_path_history.append([])
        for _ in range(max_iter):
            rnd = self.sample_new_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            new_node.cost = nearest_node.cost + self.cost(self.to_pose(new_node), self.to_pose(nearest_node))

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if log_history:
                self.nodes_history.append(copy.deepcopy(self.node_list))
                if not self.reached_goal and self.calc_dist_to_goal(new_node.x, new_node.y) < 1e-5:
                    self.reached_goal = True
                if self.reached_goal:
                    last_idx = self.search_best_goal_node()
                    path = self.generate_final_course(last_idx)
                    self.best_path_history.append(copy.deepcopy(path))
                else:
                    self.best_path_history.append([])

            # if not self.search_until_max_iter and new_node:  # if reaches goal
            #     last_idx = self.search_best_goal_node()
            #     if last_idx is not None:
            #         return self.generate_final_course(last_idx)

        last_idx = self.search_best_goal_node()
        if last_idx is not None:
            self.planned_path = self.generate_final_course(last_idx)
            return np.array([[n.x, n.y] for n in self.planned_path])
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
        d = self.cost(self.to_pose(from_node), self.to_pose(to_node))
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

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        expand_dist: float = 0.0,
        end_step=None,
        axes_setting: list = [0.09, 0.07, 0.85, 0.9]
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim, axes_setting)
        draw_obstacles(ax, self.known_obstacles, expand_dist)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)

        elems = []

        # Start Animation
        if end_step:
            animation_len = end_step
        else:
            animation_len = len(self.nodes_history) + 20
        pbar = tqdm(total=animation_len)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, animation_len, interval=100, repeat=False,
            fargs=(ax, elems, xlim, ylim, pbar),
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )

        # draw nodes
        if i < len(self.nodes_history):
            nodes = self.nodes_history[i]
        else:
            nodes = self.nodes_history[-1]
        for node in nodes:
            if node.parent:
                x = [node.path_x[0], node.path_x[-1]]
                y = [node.path_y[0], node.path_y[-1]]
                elems += ax.plot(x, y, c="cyan")

        # draw path
        for node in self.best_path_history[i]:
            if node.parent:
                elems += ax.plot(node.path_x, node.path_y, c="red")

        pbar.update(1)


class InformedRRTstar(RRTstar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reached_goal = False
        self.a = float('inf')
        self.b = float('inf')
        self.c_min = np.linalg.norm(self.goal_pos[:2] - self.start_pos[:2])
        self.e_theta = np.arctan2(self.goal_pos[1] - self.start_pos[1], self.goal_pos[0] - self.start_pos[0])
        self.sample_area_history = []

    def calculate_path(self, max_iter=500, log_history=False, *kargs):
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.reached_goal = False
        self.node_list = [self.start_node]
        if log_history:
            self.nodes_history.append(copy.deepcopy(self.node_list))
            self.sample_area_history.append([self.a, self.b, self.e_theta])
            self.best_path_history.append([])
        for _ in range(max_iter):
            rnd = self.sample_new_node()
            nearest_node = self.get_nearest_node(self.node_list, rnd)
            new_node = self.steer(nearest_node, rnd, self.expand_dis)
            new_node.cost = nearest_node.cost + self.cost(self.to_pose(new_node), self.to_pose(nearest_node))

            if self.check_collision(new_node, self.obstacle_list):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)
                if self.calc_dist_to_goal(new_node.x, new_node.y) < 1e-5:
                    self.reached_goal = True

            if self.reached_goal:
                last_idx = self.search_best_goal_node()
                path = self.generate_final_course(last_idx)
                dist = 0
                for i in range(len(path) - 1):
                    dist += self.cost(self.to_pose(path[i]), self.to_pose(path[i + 1]))
                self.a = dist / 2
                self.b = np.sqrt(dist ** 2 - self.c_min ** 2) / 2

            if log_history:
                self.nodes_history.append(copy.deepcopy(self.node_list))
                self.sample_area_history.append([self.a, self.b, self.e_theta])
                if self.reached_goal:
                    self.best_path_history.append(copy.deepcopy(path))
                else:
                    self.best_path_history.append([])

            # if not self.search_until_max_iter and new_node:  # if reaches goal
            #     last_idx = self.search_best_goal_node()
            #     if last_idx is not None:
            #         return self.generate_final_course(last_idx)

        last_idx = self.search_best_goal_node()
        if last_idx is not None:
            self.planned_path = self.generate_final_course(last_idx)
            return np.array([[n.x, n.y] for n in self.planned_path])
        return []

    def sample_new_node(self) -> RRT.Node:
        if self.reached_goal:
            # if(np.isnan(self.a) or np.isnan(self.b)):
            #     return super().sample_new_node()
            theta = 2 * random.random() * np.pi - np.pi
            x = np.sqrt(random.random()) * np.array([np.cos(theta) * self.a, np.sin(theta) * self.b])
            xp = np.dot(rotation_matrix(self.e_theta), x) + (self.start_pos[:2] + self.goal_pos[:2]) / 2
            return self.Node(xp[0], xp[1])
        else:
            return super().sample_new_node()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )

        # draw nodes
        if i < len(self.nodes_history):
            nodes = self.nodes_history[i]
        else:
            nodes = self.nodes_history[-1]
        for node in nodes:
            if node.parent:
                elems += ax.plot(node.path_x, node.path_y, c="cyan")

        a, b, e_th = self.sample_area_history[i]
        self.drawEllipse(ax, elems, a, b, e_th)

        # draw path
        for node in self.best_path_history[i]:
            if node.parent:
                elems += ax.plot(node.path_x, node.path_y, c="red")

        pbar.update(1)

    def drawEllipse(self, ax: Axes, elems: list, a: float, b: float, e_th: float):
        if a == float('inf') or b == float('inf'):
            return
        xy = (self.start_pos[:2] + self.goal_pos[:2]) / 2
        e = patches.Ellipse(
            xy=(xy[0], xy[1]),
            width=a * 2,
            height=b * 2,
            angle=np.rad2deg(e_th),
            ec='black',
            fill=False
        )
        elems.append(ax.add_patch(e))


class ChanceConstrainedRRT(RRT):
    class Node(RRT.Node):
        def __init__(self, x: float, y: float, head: float = 0.0, cov: np.ndarray = np.diag([1e-10, 1e-10, 1e-10])) -> None:
            self.x = x
            self.y = y
            self.head = head
            self.cov = cov
            self.parent: ChanceConstrainedRRT.Node = None
            self.cost = float('inf')    # Not used, for compatibility with RRT
            self.cost_lb = float('inf')         # Lower bound cost
            self.cost_ub = float('inf')  # Upper bound cost
            self.cost_fs = 0.0          # Cost from start
            self.childs = []
            self.path_x = [self.x]
            self.path_y = [self.y]
            self.path_head = [self.head]
            self.path_cov = [self.cov]

    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None, start_cov: np.ndarray = None,
        start_head: float = None, motion_noise_stds: dict = {"nn": 0.1, "no": 0.0001, "on": 0.013, "oo": 0.02},
        explore_region: list = [[0, 20], [0, 20]], known_obstacles: list = [], expand_dist: float = 0,
        expand_distance: float = 3.0, path_resolution: float = 1.0,
        cost_func=None, steer_func=None,
        goal_sample_rate: float = 0.01, goal_region=2.0,
        num_nearest_node: int = 6, p_safe: float = 0.99, k: float = 1.0
    ) -> None:
        super().__init__(start_pos, goal_pos, explore_region, known_obstacles, expand_dist, expand_distance, goal_sample_rate, path_resolution, cost_func)
        if start_pos is not None and goal_pos is not None and start_cov is not None:
            if start_head is None:
                start_head = np.arctan2(goal_pos[1] - start_pos[1], goal_pos[0] - start_pos[0])
                start_head = set_angle_into_range(start_head)
            self.start_node = self.Node(start_pos[0], start_pos[1], start_head, start_cov)
            self.goal_node = self.Node(goal_pos[0], goal_pos[1], None, None)
        self.num_nearest_node = num_nearest_node
        self.p_safe = p_safe
        self.motion_noise_stds = np.array([motion_noise_stds["nn"], motion_noise_stds["no"], motion_noise_stds["on"], motion_noise_stds["oo"]])
        self.sampled_pts = []
        self.control_inputs = self.select_inputs if steer_func is None else steer_func
        self.k = k
        self.goal_region = goal_region
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.e_theta = np.arctan2(start_pos[1] - goal_pos[1], start_pos[0] - goal_pos[0])
        self.cnt = 0
        self.dt = 0.5
        self.start_cov = start_cov
        self.start_head = start_head
        self.nodes_history = []
        self.name = "CC-RRT"

    def calculate_path(self, max_iter: int = 200, log_history=False, *kargs) -> list:
        self.cnt = 0
        self.sampled_pts = []
        if self.start_node is None:
            raise ValueError("start_node is None")
        if self.goal_node is None:
            raise ValueError("goal_node is None")

        self.node_list = [self.start_node]
        self.first_sample = True
        self.nodes_history.append(copy.deepcopy(self.node_list)) if log_history is True else None
        for _ in range(max_iter):
            self.expand_tree()
            self.nodes_history.append(copy.copy(self.node_list)) if log_history is True else None

        self.planned_path = self.generate_final_course()
        path_x = []
        path_y = []
        for n in self.planned_path:
            path_x += n.path_x
            path_y += n.path_y
        path_x = np.array(path_x)
        path_y = np.array(path_y)
        path = np.stack([path_x, path_y], axis=1)
        return path

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
        near_nodes = self.get_m_nearest_nodes(self.node_list, trg_node, self.num_nearest_node)
        for near_node in near_nodes:
            if not near_node in self.node_list:
                continue
            new_nodes, _ = self.connect_to_target(near_node, trg_node)
            if len(new_nodes) == 1:
                if math.hypot(new_nodes[0].x - near_node.x, new_nodes[0].y - near_node.y) < 0.01:
                    continue
            for new_node in new_nodes:
                new_node.cost_fs = new_node.parent.cost_fs + self.cost(self.to_pose(new_node.parent), self.to_pose(new_node))
                self.node_list.append(new_node)
                nodes_to_goal, reached_goal = self.connect_to_target(new_node, self.goal_node)
                if reached_goal is True:
                    ub_cost = 0.0
                    node = nodes_to_goal[-1]
                    while not node.parent is None:
                        p_node = node.parent
                        ub_cost += self.cost(self.to_pose(p_node), self.to_pose(node))
                        lb_cost = self.cost(self.to_pose(self.goal_node), self.to_pose(node))
                        if ub_cost < lb_cost:
                            self.prune_childs(node, self.node_list)
                        node = p_node

    def prune_childs(self, node: Node, node_list: list) -> None:
        pruned_idxes = []
        for child in node.childs:
            if child in self.node_list:
                if child in node_list:
                    idx = node_list.index(child)
                    pruned_idxes.append(idx)
                self.node_list.remove(child)
                child.parent.childs.remove(child)
                pruned_idxes += self.prune_childs(child, node_list)
        return pruned_idxes

    def sample_new_node(self) -> Node:
        self.cnt += 1
        not_safe = True
        if random.random() > self.goal_sample_rate and not self.first_sample:
            # theta = 2 * random.random() * np.pi - np.pi
            # dist = math.hypot(self.start_pos[1] - self.goal_pos[1], self.start_pos[0] - self.goal_pos[0])
            # a = dist / 2 + self.cnt * 0.02
            # b = np.sqrt((a * 2) ** 2 - dist**2) / 2
            # x = np.sqrt(random.random()) * np.array([math.cos(theta) * a, math.sin(theta) * b])
            # xp = np.dot(self.rotation_matrix(self.e_theta), x) + (self.start_pos + self.goal_pos) / 2
            # if xp[0] > self.explore_x_max or xp[0] < self.explore_x_min or xp[1] > self.explore_y_max or xp[1] < self.explore_y_min:
            xp = np.array([0.0, 0.0])
            while not_safe:
                xp[0] = random.uniform(self.explore_x_min, self.explore_x_max)
                xp[1] = random.uniform(self.explore_y_min, self.explore_y_max)
                if len(self.known_obstacles) > 0:
                    dist, idx = self.obstacle_kdTree.query(xp)
                    if dist > self.known_obstacles[idx].r + 0.5:
                        not_safe = False
                else:
                    not_safe = False
            rnd = self.Node(xp[0], xp[1])
        else:
            rnd = self.Node(self.goal_node.x, self.goal_node.y)
        self.first_sample = False
        self.sampled_pts.append(rnd)
        return rnd

    def rotation_matrix(self, t):
        return np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])

    def connect_to_target(self, prev_node: Node, trg_node: Node, expand_dis: float = float('inf'), connect_to_end=False) -> list[list[Node], bool]:
        dist_to_trg = float('inf')
        trg_pose = np.array([trg_node.x, trg_node.y, trg_node.head])
        node = self.Node(prev_node.x, prev_node.y, prev_node.head, prev_node.cov)
        node.cost_fs = prev_node.cost_fs
        self.connect_nodes(prev_node, node)
        pose = np.array([node.x, node.y, node.head])
        cov = node.cov
        new_nodes = []

        dist = 0.0
        flag_safe = True
        while dist < expand_dis:
            nu, omega = self.control_inputs(pose, trg_pose)
            new_pose, cov = self.steer(pose, cov, nu, omega)
            dist += math.hypot(new_pose[0] - pose[0], new_pose[1] - pose[1])

            if not self.is_safe(new_pose, cov):
                flag_safe = False
                break

            node = self.update_node(node, pose, new_pose, cov)
            pose = new_pose
            dist_to_trg = math.hypot(pose[1] - trg_pose[1], pose[0] - trg_pose[0])
            if connect_to_end:
                is_near_trg = dist_to_trg <= self.path_resolution * 0.5
            else:
                is_near_trg = dist_to_trg <= self.path_resolution
            is_far_last_node = math.hypot(prev_node.x - node.x, prev_node.y - node.y) > self.path_resolution
            if is_near_trg or is_far_last_node:
                node.cost_lb = self.cost(self.to_pose(node), self.to_pose(self.goal_node))
                self.connect_nodes(prev_node, node)
                new_nodes.append(node)
                prev_node = node
                node = self.Node(prev_node.x, prev_node.y, prev_node.head, prev_node.cov)
                node.cost_fs = prev_node.cost_fs
                if is_near_trg:
                    break
        if flag_safe is False:
            if not node.x == prev_node.x and node.y == prev_node.y:
                node.cost_lb = self.cost(self.to_pose(node), self.to_pose(self.goal_node))
                self.connect_nodes(prev_node, node)
                new_nodes.append(node)
        return new_nodes, (dist_to_trg < self.expand_dis and len(new_nodes) > 0) and flag_safe

    def update_node(self, node: Node, prev_pose: np.ndarray, new_pose: np.ndarray, cov: np.ndarray) -> Node:
        node.x = new_pose[0]
        node.y = new_pose[1]
        node.head = new_pose[2]
        node.cov = cov
        node.path_x.append(new_pose[0])
        node.path_y.append(new_pose[1])
        node.path_head.append(new_pose[2])
        node.path_cov.append(cov)
        node.cost_fs += self.cost(prev_pose, new_pose)
        return node

    def connect_nodes(self, pn: Node, cn: Node) -> None:
        if pn.parent == cn:
            raise Exception("pn's parent is cn")
        # if pn in cn.childs:
        #     raise Exception("pn is cn's child")
        cn.parent = pn
        # pn.childs.append(cn)

    def disconnect_nodes(self, pn: Node, cn: Node) -> None:
        cn.parent = None
        # pn.childs.remove(cn)

    def get_m_nearest_nodes(self, node_list: list[Node], rnd_node: Node, m: int) -> list[Node]:
        dist_costs = [self.dist_nodes(self.to_pose(node), self.to_pose(rnd_node)) for node in node_list]    # rnd_nodeから各nodeへの距離
        fs_costs = [node.cost_fs for node in node_list]  # スタート地点から各nodeまでの距離

        # コスト重視
        # print(dist_costs)
        # print(fs_costs)
        costs = [2.0 * dist + self.k * fs for (dist, fs) in zip(dist_costs, fs_costs)]
        costs_sorted = sorted(costs)
        mininds = []
        for i, d in enumerate(costs_sorted):
            if i >= m:
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

        near_nodes = []
        for idx in mininds:
            near_nodes.append(node_list[idx])
        return near_nodes

    def select_inputs(self, cur_pose: np.ndarray, trg_pose: np.ndarray) -> list:
        theta = math.atan2(trg_pose[1] - cur_pose[1], trg_pose[0] - cur_pose[0]) - cur_pose[2]
        if abs(theta) > math.pi / 6:
            v = 0.0
            w = theta / self.dt
        else:
            v = 1.0
            w = 3 * v * math.sin(theta) / 1.0
        return v, w

    def steer(self, prev_pose: np.ndarray, prev_cov: np.ndarray, nu: float, omega: float) -> Node:
        if abs(omega) < 1e-5:
            omega = 1e-5  # 値が0になるとゼロ割りになって計算ができないのでわずかに値を持たせる
        new_pose = state_transition(prev_pose, nu, omega, self.dt)
        new_cov = covariance_transition(prev_pose, prev_cov, self.motion_noise_stds, nu, omega, self.dt)
        return new_pose, new_cov

    def is_safe(self, x: np.ndarray, cov: np.ndarray) -> bool:
        for obs in self.obstacle_list:
            if obs.type == 'circular':
                if math.hypot(x[0] - obs.pos[0], x[1] - obs.pos[1]) > 5.0:
                    continue
                obs_pos = obs.pos[0:2]
                obs_r = obs.r
                prob_col = prob_collision(x, cov, obs_pos, obs_r)
                if prob_col > 1 - self.p_safe:
                    return False
            elif obs.type == 'rectangular':
                # @todo
                pass
        return True

    def to_pose(self, n: Node):
        return np.array([n.x, n.y, n.head])

    def to_npz_format(self, node_list: list) -> dict:
        nodes_npzf = super().to_npz_format(node_list)
        nodes_npzf['node_head'] = np.array([n.head for n in self.node_list])
        nodes_npzf['node_cov'] = np.array([n.cov for n in self.node_list])
        nodes_npzf['node_lb_costs'] = np.array([n.cost_lb for n in self.node_list])
        nodes_npzf['node_ub_costs'] = np.array([n.cost_ub for n in self.node_list])
        nodes_npzf['node_fs_costs'] = np.array([n.cost_fs for n in self.node_list])
        return nodes_npzf

    def save_log(self, src: str) -> None:
        nodes_npzf = self.to_npz_format(self.node_list)
        np.savez(
            src,
            num_node=nodes_npzf['num_node'],
            node_pos=nodes_npzf['node_pos'],
            node_head=nodes_npzf['node_head'],
            node_cov=nodes_npzf['node_cov'],
            node_lb_costs=nodes_npzf['node_lb_costs'],
            node_ub_costs=nodes_npzf['node_ub_costs'],
            node_fs_costs=nodes_npzf['node_fs_costs'],
            num_path=nodes_npzf['num_path'],
            node_path_x=nodes_npzf['node_path_x'],
            node_path_y=nodes_npzf['node_path_y'],
            node_costs=nodes_npzf['node_costs'],
            parents_idx=nodes_npzf['parents_idx'],

        )

    def load_log(self, src: str) -> None:
        log = np.load(src)
        num_node = log['num_node']
        num_path = log['num_path']
        node_pos = log['node_pos']
        node_path_x = log['node_path_x']
        node_path_y = log['node_path_y']
        node_head = log['node_head']
        node_cov = log['node_cov']
        node_costs = log['node_costs']
        node_lb_costs = log['node_lb_costs']
        node_ub_costs = log['node_ub_costs']
        node_fs_costs = log['node_fs_costs']
        node_parents_idx = log['parents_idx']
        path_cnt = 0
        for i in range(num_node):
            n = self.Node(node_pos[i][0], node_pos[i][1], node_head[i], node_cov[i])
            n.path_x = node_path_x[path_cnt:path_cnt + num_path[i]]
            n.path_y = node_path_y[path_cnt:path_cnt + num_path[i]]
            n.cost_lb = node_lb_costs[i]
            n.cost_ub = node_ub_costs[i]
            n.cost_fs = node_fs_costs[i]
            n.cost = node_costs[i]
            self.node_list.append(n)
            path_cnt += num_path[i]
        for i in range(num_node):
            if node_parents_idx[i] == -1:
                self.node_list[i].parent = None
            else:
                self.node_list[i].parent = self.node_list[node_parents_idx[i]]
        self.planned_path = self.generate_final_course()

    def draw(
            self,
            xlim: list[float] = None, ylim: list[float] = None,
            figsize: tuple[float, float] = (8, 8),
            obstacles: list = [],
            expand_dist: float = 0.0,
            draw_ellipse_flag: bool = True,
            draw_result_only: bool = False
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        draw_obstacles(ax, obstacles, expand_dist)
        self.draw_nodes(ax, draw_ellipse_flag) if draw_result_only is False else None
        self.draw_path(ax, "red", draw_ellipse_flag)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        # self.draw_sampled_pts(ax)

        plt.show()

    def draw_nodes(self, ax, draw_ellipse=False):
        for n in self.node_list:
            if n.parent:
                ax.plot(n.path_x, n.path_y, c="cyan")
                if draw_ellipse:
                    p = np.array([n.x, n.y, n.head])
                    e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3, "blue")
                    ax.add_patch(e)

    def draw_path(self, ax, c="red", draw_ellipse=False):
        if self.planned_path is None:
            return
        for n in self.planned_path:
            if n.parent:
                ax.plot(n.path_x, n.path_y, c=c)
                if draw_ellipse:
                    p = np.array([n.x, n.y, n.head])
                    e = sigma_ellipse(p[0:2], n.cov[0:2, 0:2], 3)
                    ax.add_patch(e)

    def draw_sampled_pts(self, ax):
        for pt in self.sampled_pts:
            ax.scatter(pt.x, pt.y, color="red", s=5)

    def animate(
        self,
        xlim: list[float], ylim: list[float],
        figsize: tuple[float, float] = (8, 8),
        expand_dist: float = 0.0,
        end_step=None,
        draw_ellipse_flag: bool = True,
        axes_setting: list = [0.09, 0.07, 0.85, 0.9],
        interval=100,
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim, axes_setting)
        draw_obstacles(ax, self.known_obstacles, expand_dist)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        elems = []

        # Start Animation
        if end_step:
            animation_len = end_step
        else:
            animation_len = len(self.node_list) + 20
        pbar = tqdm(total=animation_len)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, animation_len, interval=interval, repeat=False,
            fargs=(ax, elems, xlim, ylim, draw_ellipse_flag, pbar),
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, draw_ellipse: bool, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )
        if i <= len(self.node_list) - 1:
            node = self.node_list[i]
            ax.plot([node.path_x[0], node.path_x[-1]], [node.path_y[0], node.path_y[-1]], c="cyan")
            if draw_ellipse:
                p = np.array([node.x, node.y, node.head])
                e = sigma_ellipse(p[0:2], node.cov[0:2, 0:2], 3, "blue")
                ax.add_patch(e)
        elif i == len(self.node_list):
            self.draw_path(ax)
        pbar.update(1)


class ChanceConstrainedRRTstar(ChanceConstrainedRRT):
    def __init__(
        self,
        start_pos: np.ndarray = None, goal_pos: np.ndarray = None, start_cov: np.ndarray = None, start_head: float = None,
        motion_noise_stds: dict = {"nn": 0.1, "no": 0.0001, "on": 0.013, "oo": 0.02},
        explore_region: list = [[0, 20], [0, 20]],
        known_obstacles: list = [], expand_dist: float = 0, expand_distance: float = 3, path_resolution: float = 3.0,
        cost_func=None, steer_func=None,
        goal_sample_rate: float = 0.3, goal_region=2, num_nearest_node: int = 6,
        p_safe: float = 0.99, k: float = 1, mu: float = 10.0
    ) -> None:
        super().__init__(start_pos, goal_pos, start_cov, start_head, motion_noise_stds, explore_region, known_obstacles, expand_dist, expand_distance, path_resolution, cost_func, steer_func, goal_sample_rate, goal_region, num_nearest_node, p_safe, k)
        self.mu = mu
        self.mu_xfree = (explore_region[0][1] - explore_region[0][0]) * (explore_region[1][1] - explore_region[1][0])
        for obs in known_obstacles:
            self.mu_xfree -= obs.r**2 * math.pi

    def calculate_path(self, max_iter: int = 200, log_history=False, *kargs) -> list:
        return super().calculate_path(max_iter, log_history=log_history, *kargs)

    def expand_tree(self) -> None:
        trg_node = self.sample_new_node()
        # nearest_node = self.get_nearest_node(self.node_list, trg_node)

        # 一番近い3つのノードの近い順から検証する
        near_m_nodes = self.get_m_nearest_nodes(self.node_list, trg_node, 3)
        for n in near_m_nodes:
            nodes_min, succeed = self.connect_to_target(n, trg_node, connect_to_end=True)
            if succeed:
                break
        if succeed:
            cost_min = nodes_min[-1].cost_fs
            near_nodes = self.get_near_nodes_except_nearest(self.node_list, trg_node)
            for n in near_nodes:
                nodes, succeed = self.connect_to_target(n, trg_node, connect_to_end=True)
                if succeed and len(nodes) > 0:
                    if nodes[-1].cost_fs < cost_min:
                        nodes_min = nodes
                        cost_min = nodes[-1].cost_fs

            # Rewiring
            near_nodes = self.get_near_nodes_except_ancesters(self.node_list, trg_node, nodes_min[-1])
            self.node_list += nodes_min
            nodes_new = []
            for near_node in near_nodes:
                cost_min = near_node.cost_fs
                nodes, succeed = self.connect_to_target(near_node, nodes_min[-1], connect_to_end=True)
                if succeed:
                    if nodes[-1].cost_fs < cost_min:
                        if near_node.parent:
                            self.disconnect_nodes(near_node.parent, near_node)
                        if not nodes[-1].parent == near_node:
                            self.connect_nodes(nodes[-1], near_node)
                        self.propagate_cost_to_leaves(near_node)
                        nodes_new = nodes
                        cost_min = nodes[-1].cost_fs
            self.node_list += nodes_new
        else:
            self.node_list += nodes_min
        # self.draw([0, 20], [0, 20], (8, 8), self.known_obstacles, 0.5, True, False)

    def calc_new_cost(self, from_node: ChanceConstrainedRRT.Node, to_node: ChanceConstrainedRRT.Node):
        d = self.cost(self.to_pose(from_node), self.to_pose(to_node))
        return from_node.cost_fs + d

    def propagate_cost_to_leaves(self, parent_node: ChanceConstrainedRRT.Node):
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost_fs = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def update_childs_cost(self, node: ChanceConstrainedRRT.Node) -> None:
        print(self.node_list.index(node))
        for c in node.childs:
            if not c in self.node_list:
                # node.childs.remove(c)
                continue
            print("\t", self.node_list.index(c))
            c.cost_fs = node.cost_fs + self.cost(self.to_pose(node), self.to_pose(c))
            self.update_childs_cost(c)

    def get_near_nodes_except_nearest(self, node_list: list, trg_node: ChanceConstrainedRRT.Node) -> list[ChanceConstrainedRRT.Node]:
        # max_num = 10
        dlist = [self.distance(n, trg_node) for n in node_list]
        dlist = np.array(dlist)
        min_dist = np.min(dlist)
        a = np.logical_and(dlist <= self.rn, dlist > min_dist)
        near_idxes = np.where(a)[0]
        near_nodes = []
        for idx in near_idxes:
            near_nodes.append(node_list[idx])
        return near_nodes

    def get_near_nodes_except_ancesters(self, node_list: list, trg_node: ChanceConstrainedRRT.Node, n_min: ChanceConstrainedRRT.Node) -> list[ChanceConstrainedRRT.Node]:
        # max_num = 10
        near_idxes = []
        for idx, n in enumerate(node_list):
            dist = self.distance(n, trg_node)
            # if not n_min in n.childs and dist <= self.rn:
            if dist <= self.rn:
                near_idxes.append(idx)
        near_nodes = []
        # for n in node_list:
        #     near_nodes.append(n)
        for idx in near_idxes:
            near_nodes.append(node_list[idx])
        return near_nodes

    def prob_feas(self, nodes: list[ChanceConstrainedRRT.Node]):
        for n in nodes:
            safe_flag = self.is_safe(np.array([n.x, n.y]), n.cov)
            if safe_flag is False:
                return False
        return True

    @property
    def rn(self):
        n = len(self.node_list)
        d = 2.0
        zeta = math.pi
        gamma = 2**d * (1 + 1 / d) * self.mu_xfree
        r = (gamma * math.log(n) / (zeta * n)) ** (1 / d)
        return min(r, self.mu)

    def animate(
        self,
        xlim: list[float] = None, ylim: list[float] = None,
        figsize: tuple[float, float] = (8, 8),
        expand_dist: float = 0.0,
        end_step=None,
        draw_ellipse_flag: bool = True,
        axes_setting: list = [0.09, 0.07, 0.85, 0.9]
    ) -> None:
        self.fig, ax = set_fig_params(figsize, xlim, ylim, axes_setting)
        draw_obstacles(ax, self.known_obstacles, expand_dist)
        draw_start(ax, self.start_pos)
        draw_goal(ax, self.goal_pos)
        elems = []

        # Start Animation
        if end_step:
            animation_len = end_step
        else:
            animation_len = len(self.nodes_history)
        pbar = tqdm(total=animation_len)
        self.ani = anm.FuncAnimation(
            self.fig, self.animate_one_step, animation_len, interval=100, repeat=False,
            fargs=(ax, elems, xlim, ylim, draw_ellipse_flag, pbar),
        )
        plt.close()

    def animate_one_step(self, i: int, ax: Axes, elems: list, xlim: list, ylim: list, draw_ellipse_flag: bool, pbar):
        while elems:
            elems.pop().remove()

        elems.append(
            ax.text(
                xlim[0] * 0.01,
                ylim[1] * 1.02,
                f"steps = {i}",
                fontsize=10
            )
        )

        # draw nodes
        if i < len(self.nodes_history):
            nodes = self.nodes_history[i]
        else:
            nodes = self.nodes_history[-1]
        for node in nodes:
            if node.parent:
                x = [node.path_x[0], node.path_x[-1]]
                y = [node.path_y[0], node.path_y[-1]]
                elems += ax.plot(x, y, c="cyan")
                if draw_ellipse_flag:
                    p = np.array([node.x, node.y, node.head])
                    e = sigma_ellipse(p[0:2], node.cov[0:2, 0:2], 3, "blue")
                    elems.append(ax.add_patch(e))

        # draw path

        pbar.update(1)
