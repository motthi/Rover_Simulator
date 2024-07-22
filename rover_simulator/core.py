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
    def __init__(self) -> None:
        pass

    def check_collision_point(self, _) -> bool:
        raise NotImplementedError
    
    def check_collision_line(self, _) -> bool:
        raise NotImplementedError

    def draw(self) -> None:
        raise NotImplementedError

class CircularObstacle(Obstacle):
    def __init__(self, pos: np.ndarray, r: float) -> None:
        self.pos = pos
        self.r = r

    def check_collision_point(self, xy: np.ndarray, enlarge_range:float = 0.0) -> bool:
        dist = np.linalg.norm(xy - self.pos)
        return dist < self.r + enlarge_range
    
    def check_collision_line(self, xy1:np.ndarray, xy2:np.ndarray, enlarge_range: float = 0.0) -> bool:
        line_vec_x = xy2[0] - xy1[0]
        line_vec_y = xy2[1] - xy1[1]
        
        point_vec_x = self.pos[0] - xy1[0]
        point_vec_y = self.pos[1] - xy1[1]
        
        line_length_squared = line_vec_x**2 + line_vec_y**2
        if line_length_squared < 1e-8:
            t = 1
        else:
            t = (point_vec_x * line_vec_x + point_vec_y * line_vec_y) / line_length_squared
        
        t = max(0, min(1, t))
        
        closest_x = xy1[0] + t * line_vec_x
        closest_y = xy1[1] + t * line_vec_y
        
        dist = np.sqrt((closest_x - self.pos[0])**2 + (closest_y - self.pos[1])**2)
        return dist < self.r + enlarge_range

    def draw(self, ax: Axes, enlarge_range: float, alpha: float = 1.0, color_enlarge: str = 'black') -> None:
        enl_obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r + enlarge_range, fc=color_enlarge, ec=color_enlarge, zorder=-1.0, alpha=alpha)
        obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        ax.add_patch(enl_obs)
        ax.add_patch(obs)

class RectangularObstacle(Obstacle):
    def __init__(self, xy: np.ndarray, w: float, h: float, angle: float) -> None:
        self.xy = xy
        self.w = w
        self.h = h
        self.angle = angle

    def check_collision_point(self, xy: np.ndarray, enlarge_range:float = 0.0) -> bool:
        cos_th = np.cos(-np.deg2rad(self.angle))
        sin_th = np.sin(-np.deg2rad(self.angle))
        xy = xy - self.xy
        xy = np.array([[cos_th, -sin_th], [sin_th, cos_th]]) @ xy
        return -enlarge_range <= xy[0] <= self.w + enlarge_range and -enlarge_range <= xy[1] <= self.h + enlarge_range
    
    def check_collision_line(self, xy1:np.ndarray, xy2:np.ndarray, enlarge_range: float = 0.0) -> bool:
        def rotate_point(px, py, ox, oy, th):
            cos_th = np.cos(th)
            sin_th = np.sin(th)
            nx = ox + cos_th * (px - ox) - sin_th * (py - oy)
            ny = oy + sin_th * (px - ox) + cos_th * (py - oy)
            return nx, ny
        
        def line_intersects_line(p1, p2, q1, q2):
            def ccw(A, B, C):
                return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
            return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)

        corners = [
            (self.xy[0] - enlarge_range, self.xy[1]),
            (self.xy[0] + self.w + enlarge_range, self.xy[1] - enlarge_range),
            (self.xy[0] + self.w + enlarge_range, self.xy[1] + self.h + enlarge_range),
            (self.xy[0] - enlarge_range, self.xy[1] + self.h + enlarge_range)
        ]
        
        rotated_corners = [rotate_point(px, py, self.xy[0], self.xy[1], np.deg2rad(self.angle)) for px, py in corners]
        
        edges = [
            (rotated_corners[0], rotated_corners[1]),
            (rotated_corners[1], rotated_corners[2]),
            (rotated_corners[2], rotated_corners[3]),
            (rotated_corners[3], rotated_corners[0])
        ]

        for edge in edges:
            if line_intersects_line(edge[0], edge[1], (xy1[0], xy1[1]), (xy2[0], xy2[1])):
                return True
        return False

    def draw(self, ax: Axes, enlarge_range: float, alpha: float = 1.0, color_enlarge: str = 'black') -> None:
        enl_obs = patches.Rectangle(xy=(self.xy[0]-enlarge_range, self.xy[1]-enlarge_range), width=self.w + 2*enlarge_range, height=self.h + 2*enlarge_range, angle=self.angle, fc=color_enlarge, ec=color_enlarge, zorder=-1.0, alpha=alpha, rotation_point=(self.xy[0], self.xy[1]))
        obs = patches.Rectangle(xy=(self.xy[0], self.xy[1]), width=self.w, height=self.h, angle=self.angle, fc='black', ec='black', zorder=-1.0, alpha=alpha)
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
