class PathNotCalculatedError(Exception):
    pass


class PathNotFoundError(Exception):
    pass


class PathPlanner():
    def __init__(self) -> None:
        pass

    def calculate_path(self):
        raise NotImplementedError
