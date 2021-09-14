
import numpy as np
from typing import Dict, List
from scipy.spatial import cKDTree
from rover_simulator.core import Obstacle, Sensor, Rover
from rover_simulator.utils import angle_to_range, isInRange


class ImaginalSensor(Sensor):
    def __init__(self, range: float = 10.0, fov: float = np.pi / 2, obstacles: Obstacle = None) -> None:
        self.range = range
        self.fov = fov
        self.obstacles = obstacles
        obstacle_positions = [obstacle.pos for obstacle in self.obstacles] if not obstacles is None else None
        self.obstacle_kdTree = cKDTree(obstacle_positions) if not obstacle_positions is None else None

    def sense(self, rover: Rover) -> List[Dict]:
        sensed_obstacles = []
        if self.obstacle_kdTree is None:
            return sensed_obstacles
        indices = self.obstacle_kdTree.query_ball_point(rover.estimated_pose[0:2], r=self.range)
        for idx in indices:
            obstacle = self.obstacles[idx]
            obstacle_pos = obstacle.pos
            distance = np.linalg.norm(rover.estimated_pose[0:2] - obstacle_pos)
            angle = angle_to_range(
                np.arctan2(
                    obstacle_pos[1] - rover.estimated_pose[1],
                    obstacle_pos[0] - rover.estimated_pose[0]
                ) - rover.estimated_pose[2]
            )
            if isInRange(angle, - self.fov / 2, self.fov / 2):
                sensed_obstacles.append({'distance': distance, 'angle': angle, 'radius': obstacle.r})
        return sensed_obstacles


class NoisySensor(ImaginalSensor):
    def __init__(
        self,
        range: float = 10.0, fov: float = np.pi / 2, obstacles: Obstacle = None,
        distance_noise_std: float = 0.1, distanoce_noise_rate: float = 1.0,
        angle_noise_std: float = np.pi / 360, angle_noise_rate: float = 1.0,
        radius_noise_std: float = 0.01, radius_noise_rate: float = 1.0
    ) -> None:
        super().__init__(range, fov, obstacles)
        self.distance_noise = [distance_noise_std, distanoce_noise_rate]
        self.angle_noise = [angle_noise_std, angle_noise_rate]
        self.radius_noise = [radius_noise_std, radius_noise_rate]

    def sense(self, rover: Rover) -> List[Dict]:
        obstacles = super().sense(rover)
        sensed_obstacles = []
        for obstacle in obstacles:
            distance = obstacle['distance']
            angle = obstacle['angle'] + np.random.normal(0.0, self.angle_noise[0] * self.angle_noise[1] * distance)
            radius = obstacle['radius'] + np.random.normal(0.0, self.radius_noise[0] * self.radius_noise[1] * distance)
            distance += np.random.normal(0.0, self.distance_noise[0] * self.distance_noise[1] * distance)
            sensed_obstacles.append({'distance': distance, 'angle': angle, 'radius': radius})
        return sensed_obstacles
