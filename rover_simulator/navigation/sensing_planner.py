import numpy as np
from rover_simulator.core import SensingPlanner


class SimpleSensingPlanner(SensingPlanner):
    def __init__(
        self,
        rover_pose: np.ndarray = np.array([1.0, 1.0, 0.0]),
        sense_interval_distance: float = 3.0,
        sense_interval_angle: float = np.pi / 4
    ) -> None:
        self.sense_interval_ditance = sense_interval_distance
        self.sense_interval_angle = sense_interval_angle
        self.rover_pose = rover_pose
        self.distance = self.sense_interval_ditance

    def decide(self, rover_pose: np.ndarray) -> None:
        if np.abs(self.rover_pose[2] - rover_pose[2]) > self.sense_interval_angle:
            return True

        self.distance += np.linalg.norm(self.rover_pose[0:2] - rover_pose[0:2])
        self.rover_pose = rover_pose
        if self.distance >= self.sense_interval_ditance:
            self.distance = 0.0
            return True
        return False
