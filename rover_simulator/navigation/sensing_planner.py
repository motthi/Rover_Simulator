import numpy as np
from rover_simulator.core import SensingPlanner


class SimpleSensingPlanner(SensingPlanner):
    def __init__(self, rover_pose: np.ndarray = np.array([1.0, 1.0, 0.0]), sense_interval_distance: float = 3.0) -> None:
        self.sense_interval_ditance = sense_interval_distance
        self.rover_pose = rover_pose
        self.distance = self.sense_interval_ditance

    def decide(self, rover_pose: np.ndarray) -> None:
        self.distance += np.linalg.norm(self.rover_pose[0:2] - rover_pose[0:2])
        self.rover_pose = rover_pose
        if self.distance >= self.sense_interval_ditance:
            self.distance = 0.0
            return True
        return False
