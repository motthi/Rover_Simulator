import numpy as np
from rover_simulator.core import SensingPlanner


class SimpleSensingPlanner(SensingPlanner):
    def __init__(
        self,
        rover_pose: np.ndarray = np.array([1.0, 1.0, 0.0]),
        sense_interval_distance: float = 3.0,
        sense_interval_angle: float = 3 * np.pi / 16
    ) -> None:
        self.sense_interval_ditance = sense_interval_distance
        self.sense_interval_angle = sense_interval_angle
        self.rover_pose = rover_pose
        self.distance = self.sense_interval_ditance

    def decide(self, rover_pose: np.ndarray) -> None:
        self.distance += np.linalg.norm(self.rover_pose[0:2] - rover_pose[0:2])
        # d_theta = self.rover_pose[2] - rover_pose[2]
        d_theta = np.arctan2(rover_pose[1] - self.rover_pose[1], rover_pose[0] - self.rover_pose[0]) - self.rover_pose[2]
        while d_theta > np.pi:
            d_theta -= 2 * np.pi
        while d_theta < -np.pi:
            d_theta += 2 * np.pi
        if np.abs(d_theta) > self.sense_interval_angle:
            self.rover_pose = rover_pose
            return True
        d_theta = self.rover_pose[2] - rover_pose[2]
        while d_theta > np.pi:
            d_theta -= 2 * np.pi
        while d_theta < -np.pi:
            d_theta += 2 * np.pi
        if np.abs(d_theta) > self.sense_interval_angle:
            self.rover_pose = rover_pose
            return True

        if self.distance >= self.sense_interval_ditance:
            self.rover_pose = rover_pose
            self.distance = 0.0
            return True
        return False
