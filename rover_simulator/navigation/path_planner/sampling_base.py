
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from rover_simulator.navigation.path_planner import PathPlanner


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
            rnd_node = self.get_new_node()
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

    def get_new_node(self):
        if random.randint(0, 1) > self.goal_sample_rate:
            rnd = self.Node(
                random.uniform(self.explore_x_min, self.explore_x_max),
                random.uniform(self.explore_y_min, self.explore_y_max))
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
            rnd = self.get_new_node()
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
