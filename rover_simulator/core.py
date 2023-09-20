import numpy as np
import matplotlib.patches as patches
from matplotlib.axes import Axes


class History():
    real_poses: list
    estimated_poses: list
    waypoints: list
    sensing_results: list
    waypoint_color: str

    def __init__(self):
        pass

    def draw(self):
        raise NotImplementedError

    def append(self):
        raise NotImplementedError


class Sensor():
    range: float
    fov: float

    def __init__(self) -> None:
        pass

    def sense(self) -> list[dict]:
        raise NotImplementedError


class Rover():
    r: float
    real_pose: np.ndarray
    estimated_pose: np.ndarray
    history: History
    sensor: Sensor
    sensing_results: list
    waypoints: list
    waypoint_color: str

    def __init__(self) -> None:
        self.color = "black"

    def one_step(self, _) -> None:
        raise NotImplementedError


class Localizer():
    def __init__(self) -> None:
        pass

    def estimate_pose(self) -> np.ndarray:
        raise NotImplementedError


class Controller():
    def __init__(self) -> None:
        pass

    def calculate_control_inputs(self) -> tuple[float, float]:
        raise NotImplementedError()


class Mapper():
    def __init__(self) -> None:
        pass

    def update(self) -> None:
        raise NotImplementedError


class Obstacle():
    def __init__(self, position: np.ndarray, radius: float, type: int = 1) -> None:
        self.pos = position
        self.r = radius
        self.type = type

    def draw(self, ax: Axes, enlarge_range: float, alpha: float = 1.0, color_enlarge: str = 'black') -> None:
        enl_obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r + enlarge_range, fc=color_enlarge, ec=color_enlarge, zorder=-1.0, alpha=alpha)
        obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        ax.add_patch(enl_obs)
        ax.add_patch(obs)


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
