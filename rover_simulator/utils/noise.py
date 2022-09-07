import math
import numpy as np
from scipy.stats import expon, norm


class MotionNoise():
    def __init__(
        self, r,
        noise_per_meter: float = 5.0, noise_std: float = math.pi / 60,
        bias_rate_nu: float = 0.1, bias_rate_omega: float = 0.1,
        expected_stuck_time: float = 1e100,
        expected_escape_time: float = 1e-100
    ) -> None:
        self.r = r

        # Noise
        self.noise_per_meter = noise_per_meter
        self.noise_std = noise_std
        self.noise_pdf = expon(scale=1.0 / (1e-100 + self.noise_per_meter))
        self.distance_until_noise = self.noise_pdf.rvs()
        self.theta_noise = norm(scale=noise_std)

        # Bias
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_nu)
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_omega)

        # Stuck
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()

    def noise(self, pose: np.ndarray, nu: float, omega: float, dt: float) -> np.ndarray:
        self.distance_until_noise -= abs(nu) * dt + self.r * abs(omega) * dt
        if self.distance_until_noise <= 0.0:
            self.distance_until_noise += self.noise_pdf.rvs()
            pose[2] += self.theta_noise.rvs()
        return pose

    def bias(self, nu: float, omega: float):
        return nu * self.bias_rate_nu, omega * self.bias_rate_omega

    def stuck(self, nu: float, omega: float, dt):
        if self.is_stuck:
            self.time_until_escape -= dt
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False
        else:
            self.time_until_stuck -= dt
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.stuck_pdf.rvs()
                self.is_stuck = True
        return nu * (not self.is_stuck), omega * (not self.is_stuck)
