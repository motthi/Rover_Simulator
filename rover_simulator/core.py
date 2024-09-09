import numpy as np
import matplotlib.patches as patches
import matplotlib.transforms as transforms
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
        self.type: str = None

    def sense(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class PathPlanner():
    def __init__(self) -> None:
        pass

    def set_map(self):
        raise NotImplementedError

    def calculate_path(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError


class Localizer():
    def __init__(self) -> None:
        pass

    def estimate_pose(self) -> np.ndarray:
        raise NotImplementedError


class Controller():
    def __init__(self) -> None:
        self.fig = None
        self.ani = None

    def calculate_control_inputs(self) -> tuple[float, float]:
        raise NotImplementedError()

    def animiate(self) -> None:
        raise NotImplementedError()


class Mapper():
    def __init__(self) -> None:
        pass

    def update(self) -> None:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError


class Rover():
    r: float
    real_pose: np.ndarray
    estimated_pose: np.ndarray
    history: History
    sensor: Sensor
    mapper: Mapper
    path_planner: PathPlanner
    sensing_results: list
    waypoints: list
    waypoint_color: str

    def __init__(self) -> None:
        self.color = "black"

    def one_step(self, _) -> None:
        raise NotImplementedError


class Obstacle():
    def __init__(self) -> None:
        self.type = None

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
        self.type = 'circular'

    def check_collision_point(self, xy: np.ndarray, expand_dist: float = 0.0) -> bool:
        dist = np.linalg.norm(xy - self.pos)
        return dist < self.r + expand_dist

    def check_collision_line(self, xy1: np.ndarray, xy2: np.ndarray, expand_dist: float = 0.0) -> bool:
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
        return dist < self.r + expand_dist

    def draw(self, ax: Axes, expand_dist: float, alpha: float = 1.0, color_enlarge: str = 'black') -> None:
        enl_obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r + expand_dist, fc=color_enlarge, ec=color_enlarge, zorder=-1.0, alpha=alpha)
        obs = patches.Circle(xy=(self.pos[0], self.pos[1]), radius=self.r, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        ax.add_patch(enl_obs)
        ax.add_patch(obs)


class RectangularObstacle(Obstacle):
    def __init__(self, xy: np.ndarray, w: float, h: float, angle: float) -> None:
        self.xy = xy
        self.w = w
        self.h = h
        self.angle = angle
        self.type = 'rectangular'

        self.corner_points = [
            np.array([0, 0]),
            np.array([self.w, 0]),
            np.array([self.w, self.h]),
            np.array([0, self.h])
        ]

        self.corners = [
            (self.xy[0], self.xy[1]),
            (self.xy[0] + self.w, self.xy[1]),
            (self.xy[0] + self.w, self.xy[1] + self.h),
            (self.xy[0], self.xy[1] + self.h)
        ]

    def check_collision_point(self, xy: np.ndarray, expand_dist: float = 0.0) -> bool:
        cos_th = np.cos(-np.deg2rad(self.angle))
        sin_th = np.sin(-np.deg2rad(self.angle))
        xy = xy - self.xy
        xy = np.array([[cos_th, -sin_th], [sin_th, cos_th]]) @ xy
        if 0 <= xy[0] <= self.w and 0 <= xy[1] <= self.h:
            return True

        if -expand_dist <= xy[0] <= self.w + expand_dist and -expand_dist <= xy[1] <= self.h + expand_dist:
            for corner in self.corner_points:
                if np.linalg.norm(xy - corner) <= expand_dist:
                    return True
        return False

    def check_collision_line(self, xy1: np.ndarray, xy2: np.ndarray, expand_dist: float = 0.0) -> bool:
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

        def line_intersects_circle(p1, p2, center, radius):
            d = np.array(p2) - np.array(p1)
            f = np.array(p1) - np.array(center)
            a = np.dot(d, d)
            b = 2 * np.dot(f, d)
            c = np.dot(f, f) - radius**2

            if a < 1e-8:
                # p1 == p2 の場合、直線が点 p1 に一致する
                distance_squared = np.dot(f, f)
                return distance_squared <= radius**2

            discriminant = b**2 - 4 * a * c
            if discriminant < 0:
                return False
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2 * a)
            t2 = (-b + discriminant) / (2 * a)
            return 0 <= t1 <= 1 or 0 <= t2 <= 1

        e1 = rotate_point(self.xy[0], self.xy[1] - expand_dist, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e2 = rotate_point(self.xy[0] + self.w, self.xy[1] - expand_dist, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e3 = rotate_point(self.xy[0] + self.w + expand_dist, self.xy[1], self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e4 = rotate_point(self.xy[0] + self.w + expand_dist, self.xy[1] + self.h, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e5 = rotate_point(self.xy[0] + self.w, self.xy[1] + self.h + expand_dist, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e6 = rotate_point(self.xy[0], self.xy[1] + self.h + expand_dist, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e7 = rotate_point(self.xy[0] - expand_dist, self.xy[1] + self.h, self.xy[0], self.xy[1], np.deg2rad(self.angle))
        e8 = rotate_point(self.xy[0] - expand_dist, self.xy[1], self.xy[0], self.xy[1], np.deg2rad(self.angle))

        expanded_edges = [
            (e1, e2), (e2, e3), (e3, e4), (e4, e5), (e5, e6), (e6, e7), (e7, e8), (e8, e1)
        ]

        for edge in expanded_edges:
            if line_intersects_line(edge[0], edge[1], (xy1[0], xy1[1]), (xy2[0], xy2[1])):
                return True

        for corner in self.corners:
            rotated_corner = rotate_point(corner[0], corner[1], self.xy[0], self.xy[1], np.deg2rad(self.angle))
            if line_intersects_circle((xy1[0], xy1[1]), (xy2[0], xy2[1]), rotated_corner, expand_dist):
                return True

        return False

    def draw(self, ax: Axes, expand_dist: float, alpha: float = 1.0, color_enlarge: str = 'black') -> None:
        enl_obs = patches.FancyBboxPatch(
            (self.xy[0], self.xy[1]),
            self.w,
            self.h,
            boxstyle=f"round,pad={expand_dist}",
            fc=color_enlarge,
            ec=color_enlarge,
            zorder=-1.0,
            alpha=alpha
        )
        trans = transforms.Affine2D().rotate_deg_around(self.xy[0], self.xy[1], self.angle) + ax.transData
        enl_obs.set_transform(trans)
        obs = patches.Rectangle(xy=(self.xy[0], self.xy[1]), width=self.w, height=self.h, angle=self.angle, fc='black', ec='black', zorder=-1.0, alpha=alpha)
        ax.add_patch(enl_obs)
        ax.add_patch(obs)


class BaseCollisionDetector:
    def __init__(self) -> None:
        pass

    def detect_collision(self, _) -> None:
        raise NotImplementedError


class SensingPlanner():
    def __init__(self) -> None:
        pass

    def decide(self, *args, **kwargs) -> None:
        return True
