from __future__ import annotations
import numpy as np
import matplotlib.patches as patches


class Rover():
    def __init__(self) -> None:
        pass

    def one_step(self, _) -> None:
        raise NotImplementedError


class Localizer():
    def __init__(self) -> None:
        pass

    def estimate_pose(self) -> np.ndarray:
        raise NotImplementedError


class Sensor():
    def __init__(self) -> None:
        pass

    def sense(self) -> list[dict]:
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

    def draw(self, ax, enlarge_range, alpha) -> None:
        enl_obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r + enlarge_range, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        ax.add_patch(enl_obs)
        # obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        # ax.add_patch(obs)


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
