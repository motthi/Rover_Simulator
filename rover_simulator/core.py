import numpy as np


class Rover():
    def __init__(self) -> None:
        pass

    def one_step(self, _):
        raise NotImplementedError


class Localizer():
    def __init__(self) -> None:
        pass

    def estimate_pose(self):
        raise NotImplementedError


class Sensor():
    def __init__(self) -> None:
        pass

    def sense(self):
        raise NotImplementedError


class Controller():
    def __init__(self) -> None:
        pass

    def calculate_control_inputs(self):
        raise NotImplementedError()


class Mapper():
    def __init__(self) -> None:
        pass

    def update(self):
        raise NotImplementedError


class Obstacle():
    def __init__(self, position: np.ndarray, radius: float, type: int = 1) -> None:
        self.pos = position
        self.r = radius
        self.type = type


class CollisionDetector:
    def __init__(self) -> None:
        pass

    def detect_collision(self, _) -> None:
        raise NotImplementedError


class SensingPlanner():
    def __init__(self) -> None:
        pass

    def decide(self, *args, **kwargs) -> None:
        return True
