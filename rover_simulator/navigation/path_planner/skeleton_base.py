import cv2
import sknw
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from rover_simulator.core import PathPlanner
from rover_simulator.utils.draw import set_fig_params, draw_grid_map
from rover_simulator.navigation.mapper import GridMapper


class SkeltonPlanner(PathPlanner):
    def __init__(self, rescale: float = 10) -> None:
        self.rescale = rescale
        self.graph = None
        self.path = None
        self.start = None
        self.goal = None

    def build_skelton_network(self, gridmap):
        sk_img = self.skeletonize(gridmap)
        self.graph = self.extract_network(sk_img)
        return self.graph

    def skeletonize(self, gridmap):
        v_map = np.zeros(gridmap.shape)
        idx = np.where(gridmap > 0.9)
        v_map[idx] = 255.0
        v_map = v_map.astype(dtype=np.uint8)
        v_map = cv2.resize(v_map, (v_map.shape[0] * self.rescale, v_map.shape[1] * self.rescale))
        # v_map = v_map.T

        _, binimg = cv2.threshold(v_map, 125, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        notmask = cv2.bitwise_not(binimg)
        notmask[0, :] = 0
        notmask[-1, :] = 0
        notmask[:, 0] = 0
        notmask[:, -1] = 0

        # Distance Transform
        dist_ = cv2.distanceTransform(notmask, cv2.DIST_L2, 3)
        dist = cv2.normalize(dist_, None, 0, 255, cv2.NORM_MINMAX)
        dist8u = dist.astype(np.uint8)
        idx = dist8u > 60
        dist8u[idx] = 60

        # Skeletonize
        _, bin_route = cv2.threshold(dist8u, 20, 255, cv2.THRESH_BINARY)
        skeleton = cv2.ximgproc.thinning(bin_route, thinningType=cv2.ximgproc.THINNING_GUOHALL)
        return skeleton

    def extract_network(self, skimg: np.ndarray) -> nx.Graph:
        graph = sknw.build_sknw(skimg.astype(np.uint32), multi=True)

        # Rescale
        for (s, e) in graph.edges():
            graph[s][e][0]['pts'] = graph[s][e][0]['pts'] / (2 * self.rescale)
        for i in graph.nodes():
            graph.nodes[i]['o'] = graph.nodes[i]['o'] / (2 * self.rescale)
        return graph

    def set_start_goal(self, start, goal):
        self.start = start
        self.goal = goal

        self.start_node = self.find_nearest_node(start)
        self.goal_node = self.find_nearest_node(goal)

    def find_nearest_node(self, pos):
        nodes = self.graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        dist = np.linalg.norm(ps - pos, axis=1)
        return np.argmin(dist)

    def calculate_path(self):
        path = nx.astar_path(self.graph, self.start_node, self.goal_node, weight='weight')
        self.path = np.zeros((0, 2))
        for s, e in zip(path[:-1], path[1:]):
            edge = self.graph[s][e][0]['pts']
            if s > e:
                edge = edge[::-1]
            self.path = np.concatenate([self.path, edge], axis=0)
        self.path = np.concatenate([np.array([self.start]), self.path, np.array([self.goal])], axis=0)
        return self.path

    def draw(
            self,
            mapper: GridMapper,
            figsize=(8, 8),
            xlim: list[float] = None, ylim: list[float] = None
    ):
        self.fig, ax = set_fig_params(figsize, xlim, ylim)
        extent = (
            -mapper.grid_width / 2,
            mapper.grid_width * mapper.grid_num[0],  # - mapper.grid_width / 2,
            -mapper.grid_width / 2, mapper.grid_width * mapper.grid_num[1]  # - mapper.grid_width / 2
        )
        draw_grid_map(ax, mapper.map, "Greys", 0.0, 1.0, 1.0, extent, 1.0)
        for (s, e) in self.graph.edges():
            ps = self.graph[s][e][0]['pts']
            plt.plot(ps[:, 0], ps[:, 1], 'green')
        nodes = self.graph.nodes()
        ps = np.array([nodes[i]['o'] for i in nodes])
        plt.plot(ps[:, 0], ps[:, 1], 'r.')

        plt.plot(self.start[0], self.start[1], 'bo')
        plt.plot(self.goal[0], self.goal[1], 'ro')
        plt.plot(self.path[:, 0], self.path[:, 1], 'b-')

        plt.show()
