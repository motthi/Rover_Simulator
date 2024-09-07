class PathNotCalculatedError(Exception):
    pass


class PathNotFoundError(Exception):
    pass


class PathPlanner():
    def __init__(self) -> None:
        pass

    def set_map(self):
        raise NotImplementedError

    def calculate_path(self):
        raise NotImplementedError

    def draw(self):
        raise NotImplementedError
