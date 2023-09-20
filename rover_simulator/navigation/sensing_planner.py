import numpy as np
from rover_simulator.core import SensingPlanner
from rover_simulator.utils.utils import set_angle_into_range, is_angle_in_range


class SimpleSensingPlanner(SensingPlanner):
    def __init__(
        self,
        rover_pose: np.ndarray = np.array([1.0, 1.0, 0.0]),
        sense_interval_distance: float = 3.0,
        sense_interval_angle: float = 3 * np.pi / 16
    ) -> None:
        self.sense_interval_distance = sense_interval_distance
        self.sense_interval_angle = sense_interval_angle
        self.rover_pose = rover_pose
        self.previous_pose = rover_pose
        self.previous_sensing_pose = np.array([-float('inf'), -float('inf')])
        self.distance = self.sense_interval_distance

    def decide(self, rover_pose: np.ndarray) -> None:
        dist = np.linalg.norm(self.previous_sensing_pose[0:2] - rover_pose[0:2])
        if dist > self.sense_interval_distance:
            self.previous_sensing_pose = rover_pose
            return True
        d_theta = set_angle_into_range(
            np.arctan2(
                rover_pose[1] - self.previous_sensing_pose[1], rover_pose[0] - self.previous_sensing_pose[0]
            ) - self.previous_sensing_pose[2]
        )
        if dist > 1e-5 and not is_angle_in_range(d_theta, -self.sense_interval_angle, self.sense_interval_angle):
            # dist > 1e-5 => Not Moved
            self.previous_sensing_pose = rover_pose
            return True
        d_theta = rover_pose[2] - self.previous_sensing_pose[2]
        if np.abs(d_theta) > self.sense_interval_angle:
            self.previous_sensing_pose = rover_pose
            return True
        return False
