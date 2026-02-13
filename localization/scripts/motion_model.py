import numpy as np
from utils import wrap_angle

class MotionModel:
    def __init__(self, x_std=0.04, y_std=0.04, yaw_std=0.01):
        self.x_std = x_std
        self.y_std = y_std
        self.yaw_std = yaw_std

    def predict(self, particles, dx_l, dy_l, dyaw):
        """
        AMCL算法中的预测点云更新逻辑
        
        :param self: 上面的生成参数
        :param praticles: 粒子参数
        :param dx_l/dy_l/dyaw: 三个参量变化的尺度
        """
        N = particles.shape[0]
        scale_x = self.x_std * abs(dx_l) + 0.001 # 防止为0
        scale_y = self.y_std * abs(dy_l) + 0.001
        scale_yaw = self.yaw_std * abs(dyaw) + 0.001
        # 噪声生成
        noise_dx = np.random.normal(0, scale_x, N) + dx_l
        noise_dy = np.random.normal(0, scale_y, N) + dy_l
        noise_yaw = np.random.normal(0, scale_yaw, N) + dyaw
        # 状态更新
        cos_yaw = np.cos(particles[:, 2])
        sin_yaw = np.sin(particles[:, 2])
        particles[:, 0] += noise_dx * cos_yaw + noise_dy * -sin_yaw
        particles[:, 1] += noise_dx * sin_yaw + noise_dy * cos_yaw
        particles[:, 2] += noise_yaw
        particles[:, 2] = np.array([wrap_angle(yaw) for yaw in particles[:, 2]])
        return particles
