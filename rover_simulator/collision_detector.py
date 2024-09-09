from rover_simulator.core import BaseCollisionDetector, Rover, Obstacle


class IgnoreCollision(BaseCollisionDetector):
    def __init__(self) -> None:
        pass

    def detect_collision(self, _) -> None:
        return False


class CollisionDetector(BaseCollisionDetector):
    def __init__(self, obstacles: Obstacle) -> None:
        self.obstacles: list[Obstacle] = obstacles

    def detect_collision(self, rover: Rover) -> bool:
        for obstacle in self.obstacles:
            if obstacle.check_collision_point(rover.real_pose[0:2], rover.r):
                return True
        return False
