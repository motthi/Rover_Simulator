import numpy as np
from scipy.interpolate import splprep, splev


class PathSmoother:
    pass


class SplineSmoother(PathSmoother):
    def __init__(self, n_interpolate: int = 10):
        self.n_interpolate = n_interpolate

    def calculate(self, path: np.ndarray) -> np.ndarray:
        n_points = len(path)
        x, y = path[:, 0], path[:, 1]
        n_interpolate = n_points * self.n_interpolate
        tck, u = splprep([x, y], s=0, k=3)
        u_new = np.linspace(u.min(), u.max(), n_interpolate)
        x_new, y_new = splev(u_new, tck, der=0)
        path = np.array([x_new, y_new]).T
        return path


class IterativeSmoother(PathSmoother):
    def __init__(self, n_iter: int = 100, alpha: float = 0.1, beta: float = 0.1):
        self.n_iter = n_iter
        self.alpha = alpha
        self.beta = beta

    def calculate(self, path: np.ndarray) -> np.ndarray:
        for i in range(self.n_iter):
            path = self._one_iter(path)
        return path

    def _one_iter(self, path: np.ndarray) -> np.ndarray:
        n_points = len(path)
        new_path = np.copy(path)
        for i in range(1, n_points - 1):
            new_path[i] = new_path[i] - self.alpha * (new_path[i] - path[i])
            new_path[i] = new_path[i] - self.beta * (2 * new_path[i] - path[i - 1] - path[i + 1])
        new_path[0] = path[0]
        new_path[-1] = path[-1]
        return new_path
