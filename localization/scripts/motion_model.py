import numpy as np
from utils import wrap_angle


class MotionModel:
    def __init__(self, x_std=0.04, y_std=0.04, yaw_std=0.01):
        self.x_std = x_std
        self.y_std = y_std
        self.yaw_std = yaw_std

    def predict(self, particles, dx_l, dy_l, dyaw):
        N = particles.shape[0]
        scale_x = self.x_std * abs(dx_l) + 0.001
        scale_y = self.y_std * abs(dy_l) + 0.001
        scale_yaw = self.yaw_std * abs(dyaw) + 0.001
        noisy_dx = dx_l + np.random.normal(0, scale_x, N)
        noisy_dy = dy_l + np.random.normal(0, scale_y, N)
        noisy_yaw = dyaw + np.random.normal(0, scale_yaw, N)
        cos_t = np.cos(particles[:, 2])
        sin_t = np.sin(particles[:, 2])
        particles[:, 0] += noisy_dx * cos_t - noisy_dy * sin_t
        particles[:, 1] += noisy_dx * sin_t + noisy_dy * cos_t
        particles[:, 2] += noisy_yaw
        particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))
        return particles