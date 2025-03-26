import numpy as np
from scipy.spatial import cKDTree
from matplotlib.axes import Axes
from matplotlib import patches
from rover_simulator.core import Obstacle, Sensor, Rover
from rover_simulator.utils.utils import set_angle_into_range, is_angle_in_range


def generate_points_in_circle(r: float, d: float = 0.1):
    points = []
    current_radius = 0
    while current_radius <= r:
        circumference = 2 * np.pi * current_radius
        num_points_in_ring = max(1, int(circumference / d))
        for j in range(num_points_in_ring):
            theta = 2 * np.pi * j / num_points_in_ring
            x = current_radius * np.cos(theta)
            y = current_radius * np.sin(theta)
            points.append((x, y))
        current_radius += d * np.sqrt(3) / 2  # Hexagonal packing
    return np.array(points)


def generate_points_in_rectangle(w: float, h: float, d: float = 0.1):
    points = []
    xs = np.arange(0, w, d) + d / 2
    ys = np.arange(0, h, d) + d / 2
    for x in xs:
        for y in ys:
            points.append((x, y))
    return np.array(points)


def find_intersections_with_circle(xr: float, yr: float, xe: float, ye: float, xc: float, yc: float, r: float):
    dx = xe - xr
    dy = ye - yr

    fx = xr - xc
    fy = yr - yc

    a = dx**2 + dy**2
    b = 2 * (fx * dx + fy * dy)
    c = fx**2 + fy**2 - r**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return []
    else:
        sqrt_disc = np.sqrt(discriminant)

        t1 = (-b + sqrt_disc) / (2 * a)
        t2 = (-b - sqrt_disc) / (2 * a)

        poi1 = (xr + t1 * dx, yr + t1 * dy)
        poi2 = (xr + t2 * dx, yr + t2 * dy)

        poi = []
        if 0 <= t1 <= 1:
            poi.append(poi1)
        if 0 <= t2 <= 1:
            poi.append(poi2)

        return poi


def find_intersection_with_rectangle(xr: float, yr: float, xe: float, ye: float, xc: float, yc: float, w: float, h: float, ang: float):
    def rotate_point(x, y, cx, cy, angle):
        rad = np.deg2rad(angle)
        cos_rad = np.cos(rad)
        sin_rad = np.sin(rad)
        x_shifted = x - cx
        y_shifted = y - cy
        x_rot = x_shifted * cos_rad - y_shifted * sin_rad
        y_rot = x_shifted * sin_rad + y_shifted * cos_rad
        return x_rot + cx, y_rot + cy

    corners = [
        (xc, yc),
        (xc + w, yc),
        (xc + w, yc + h),
        (xc, yc + h)
    ]

    rotated_corners = [rotate_point(x, y, xc, yc, ang) for x, y in corners]

    edges = [
        (rotated_corners[0], rotated_corners[1]),
        (rotated_corners[1], rotated_corners[2]),
        (rotated_corners[2], rotated_corners[3]),
        (rotated_corners[3], rotated_corners[0])
    ]

    def line_intersection(p0, p1, p2, p3):
        s1_x = p1[0] - p0[0]
        s1_y = p1[1] - p0[1]
        s2_x = p3[0] - p2[0]
        s2_y = p3[1] - p2[1]

        s = (-s1_y * (p0[0] - p2[0]) + s1_x * (p0[1] - p2[1])) / (-s2_x * s1_y + s1_x * s2_y)
        t = (s2_x * (p0[1] - p2[1]) - s2_y * (p0[0] - p2[0])) / (-s2_x * s1_y + s1_x * s2_y)

        if 0 <= s <= 1 and 0 <= t <= 1:
            i_x = p0[0] + (t * s1_x)
            i_y = p0[1] + (t * s1_y)
            return (i_x, i_y)
        return None

    poi = []
    for edge in edges:
        intersection = line_intersection((xr, yr), (xe, ye), edge[0], edge[1])
        if intersection:
            poi.append(intersection)

    return poi


class ImaginalStereoCamera(Sensor):
    def __init__(self, range: float = 10.0, fov: float = np.pi / 2, obstacles: Obstacle = None) -> None:
        self.range = range
        self.fov = fov
        self.obstacles: list[Obstacle] = obstacles
        self.type = 'stereo_camera'

    def sense(self, rover: Rover) -> np.ndarray:
        pts = []
        for obstacle in self.obstacles:
            if obstacle.type == 'circular':
                smp_pts = generate_points_in_circle(obstacle.r)
                smp_pts = smp_pts + obstacle.pos
            elif obstacle.type == 'rectangular':
                ang = np.deg2rad(obstacle.angle)
                smp_pts = generate_points_in_rectangle(obstacle.w, obstacle.h)
                smp_pts = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]) @ smp_pts.T
                smp_pts = smp_pts.T + obstacle.xy

            for pt in smp_pts:
                distance = np.linalg.norm(rover.real_pose[0:2] - pt)
                angle = set_angle_into_range(
                    np.arctan2(pt[1] - rover.real_pose[1], pt[0] - rover.real_pose[0]) - rover.real_pose[2]
                )
                if is_angle_in_range(angle, - self.fov / 2, self.fov / 2) and distance < self.range:
                    pts.append(pt - rover.real_pose[:2])
        return np.array(pts)

    def draw(self, ax: Axes, elems: list, result: np.ndarray, rover_pose: np.ndarray, c='blue'):
        if len(result) > 0:
            result = result + rover_pose[:2]
            elems += ax.plot(result[:, 0], result[:, 1], 'o', color=c, markersize=1)

        sensing_range = patches.Wedge(
            (rover_pose[0], rover_pose[1]), self.range,
            theta1=np.rad2deg(rover_pose[2] - self.fov / 2),
            theta2=np.rad2deg(rover_pose[2] + self.fov / 2),
            alpha=0.5,
            color="mistyrose"
        )
        elems.append(ax.add_patch(sensing_range))


class ImaginalLiDAR(Sensor):
    def __init__(self, range: float = 10.0, fov: float = np.pi, d_ang: float = np.pi / 360, obstacles: Obstacle = None) -> None:
        self.range = range
        self.fov = fov
        self.d_ang = d_ang
        self.smp_ang = np.arange(-self.fov / 2, self.fov / 2, d_ang)
        self.smp_num = len(self.smp_ang)
        self.obstacles: list[Obstacle] = obstacles
        self.type = 'lidar'

    def sense(self, rover: Rover) -> np.ndarray:
        smp_ang = np.arange(-self.fov / 2, self.fov / 2, self.d_ang) + rover.real_pose[2]
        xr = rover.real_pose[0]
        yr = rover.real_pose[1]

        pts = []
        for ang in smp_ang:
            xe = xr + self.range * np.cos(ang)
            ye = yr + self.range * np.sin(ang)
            min_dist = float('inf')
            for obstacle in self.obstacles:
                if obstacle.type == 'circular':
                    psoi = find_intersections_with_circle(xr, yr, xe, ye, obstacle.pos[0], obstacle.pos[1], obstacle.r)
                elif obstacle.type == 'rectangular':
                    psoi = find_intersection_with_rectangle(xr, yr, xe, ye, obstacle.xy[0], obstacle.xy[1], obstacle.w, obstacle.h, obstacle.angle)
                for poi in psoi:
                    dist = np.linalg.norm(np.array(poi) - np.array([xr, yr]))
                    if min_dist > dist:
                        min_dist = dist
            pts.append([min_dist, ang - rover.real_pose[2]])
        return np.array(pts)

    def draw(self, ax: Axes, elems: list, result: np.ndarray, rover_pose: np.ndarray, c='blue'):
        pts = []
        for pt in result:
            x = rover_pose[0] + pt[0] * np.cos(pt[1] + rover_pose[2])
            y = rover_pose[1] + pt[0] * np.sin(pt[1] + rover_pose[2])
            pts.append([x, y])
        if len(pts) > 0:
            pts = np.array(pts)
            elems += ax.plot(pts[:, 0], pts[:, 1], 'o', color=c, markersize=1)

        sensing_range = patches.Wedge(
            (rover_pose[0], rover_pose[1]), self.range,
            theta1=np.rad2deg(rover_pose[2] - self.fov / 2),
            theta2=np.rad2deg(rover_pose[2] + self.fov / 2),
            alpha=0.5,
            color="mistyrose"
        )
        elems.append(ax.add_patch(sensing_range))
