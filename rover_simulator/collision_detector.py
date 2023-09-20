import numpy as np
from scipy.spatial import cKDTree
from rover_simulator.core import CollisionDetector, Rover, Obstacle


class IgnoreCollision(CollisionDetector):
    def __init__(self) -> None:
        pass

    def detect_collision(self, _) -> None:
        return False


class CollisionDetector(CollisionDetector):
    def __init__(self, obstacles: Obstacle) -> None:
        self.obstacles = obstacles
        obstacle_positions = [obstacle.pos for obstacle in self.obstacles]
        self.obstacle_kdTree = cKDTree(obstacle_positions)

    def detect_collision(self, rover: Rover) -> bool:
        indices = self.obstacle_kdTree.query_ball_point(rover.real_pose[0:2], r=4.0)
        for idx in indices:
            pos = self.obstacles[idx].pos
            distance = np.linalg.norm(rover.real_pose[0:2] - pos)
            if distance < rover.r + self.obstacles[idx].r:
                return True
        return False
