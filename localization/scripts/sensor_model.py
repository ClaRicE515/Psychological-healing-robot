#!/usr/bin/env python3
# coding: utf-8
import rospy
import math
import numpy as np
from utils import world_to_map, pose_in_collision

try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class SensorModel:
    def __init__(self, downsample_step=1, laser_offset=0.0,  sigma_hit=0.2,
                 z_hit=0.95, z_rand=0.05, oob_mode='skip', max_searchg_cells=6, robot_radius=0.28):
        """
        AMCL算法中的传感器模型逻辑
        
        :param downsample_size: 激光雷达数据下采样比例
        :param laser_offset: 激光雷达到机器人中心的偏移，单位米
        :param sigma_hit: 传感器模型参数：测距误差的宽度
        :param z_hit: 传感器模型参数：测量与预期匹配的可信度
        :param z_rand: 传感器模型参数：出现任意（无关）读数的概率
        :param oob_mode: 粒子出界处理方式，可选 'skip', 'count_as_free', 'count_as_occupied'
        :param max_searchg_cells: 最大搜索距离，单位：地图格子数(在无法获取距离场时使用，直接暴力搜索最近占据格子)
        :param robot_radius: 机器人实体半径，单位：米
        """
        self.downsample_step = downsample_step # 激光雷达数据下采样步长
        self.laser_offset = laser_offset # 激光雷达到机器人中心的偏移，单位米
        self.z_hit = z_hit # 传感器模型参数：测量与预期匹配的可信度
        self.z_rand = z_rand # 传感器模型参数：出现任意（无关）读数的概率
        self.sigma_hit = sigma_hit # 传感器模型参数：测距误差的宽度
        self.robot_radius = robot_radius # 机器人实体半径，单位：米
        self.oob_mode = oob_mode # 粒子出界处理方式
        self.max_searchg_cells = max_searchg_cells # 最大搜索距离，单位：地图格子数

        self.downsampled_angles = None # 下采样后的激光雷达角度数组
        self.input_angle_min = 0.0 # 激光雷达输入的最小角度
        self.input_angle_increment = 0.0 # 激光雷达输入的角度增量
        self.num_beams = 0 # 激光束数量
        self.occupancy = None # 地图占据矩阵
        self.angles_initialized = False # 激光雷达角度是否已初始化
        self.distance_field = None # 是否获得了距离场
        self.map_info = None
        self.map_data = None

        self.explain_tol = rospy.get_param('~explain_tol', 0.20)  # 米
        self.explain_fraction_thresh = rospy.get_param('~explain_fraction_thresh', 0.80)  # 最低解释比例
        
        if not _HAVE_SCIPY:
            rospy.logwarn_once("SCIPY not found.")

    def set_scan_geometory(self, angle_min, angle_increment, num_beams):
        """
        设置激光雷达的几何参数
        
        :param angle_min: 激光雷达开始扫描的角度
        :param angle_increment: 每次采样的角度增量
        :param num_beams: 激光束数量
        :return: None
        """
        indices = np.arange(num_beams)
        all_angles = angle_min + indices * angle_increment
        self.downsampled_angles = all_angles[::self.downsample_step] # 下采样后的激光雷达角度数组
        # 记录输入的参数，方便后续使用
        self.input_angle_min = angle_min
        self.input_angle_increment = angle_increment
        self.num_beams = num_beams
        self.angles_initialized = True

    def set_scan_geometry(self, angle_min, angle_increment, num_beams):
        return self.set_scan_geometory(angle_min, angle_increment, num_beams)

    def set_map(self, map_data, map_info):
        """
        设置地图数据，并预处理距离变换矩阵
        
        :param map_data: 地图数据数组
        :param map_info: 地图信息对象，包含原点位置，map分辨率等
        :return: None
        """
        self.map_data = map_data
        self.map_info = map_info

        width = map_info.width
        height = map_info.height

        array = np.array(map_data, dtype=np.int8).reshape((height, width))
        occupancy = (array > 50).astype(np.uint8)
        self.occupancy = occupancy
        if _HAVE_SCIPY: 
            try:
                # 得到一个与地图网格同尺寸的 2D NumPy 浮点数组，表示每个栅格到最近占据（障碍）栅格的欧氏距离（单位：米）
                dist_cells = distance_transform_edt(1 - occupancy)
                self.distance_field = dist_cells * map_info.resolution
            except Exception:
                rospy.logwarn_once("Error computing distance field.")
                self.distance_field = None
        else:
            rospy.logwarn_once("SCIPY not found, distance field not computed.")
            self.distance_field = None
        
    def get_weight(self, candidate_pose, scan_msg, map_data, map_info):
        """
        计算单个粒子的权重
        
        :param candidate_pose: 候选粒子位姿 [x, y, yaw]
        :param scan_msg: 激光雷达消息对象
        :param map_data: 地图数据数组
        :param map_info: 地图信息对象，包含原点位置，map分辨率等
        :return: 该粒子的权重值
        """
        if not self.angles_initialized:
            try:
                num_beams = len(scan_msg.ranges)
                self.set_scan_geometory(scan_msg.angle_min,
                                        scan_msg.angle_increment,
                                        num_beams)
            except Exception as e:
                rospy.logerr("Failed to set scan geometry: {}".format(e))
                return 1e-8
            
        
        full_ranges = np.array(scan_msg.ranges)
        ranges = full_ranges[::self.downsample_step]
        angles = self.downsampled_angles

        if angles.shape[0] != ranges.shape[0]:
            L = min(angles.shape[0], ranges.shape[0])
            angles = angles[:L]
            ranges = ranges[:L]

        valid = (~np.isnan(ranges)) & (ranges < scan_msg.range_max) & (ranges > scan_msg.range_min) & (~np.isinf(ranges))
        # 若全部是无效数据，直接返回极小权重
        if not np.any(valid):
            return 1e-8
        
        px = candidate_pose[0]
        py = candidate_pose[1]
        pyaw = candidate_pose[2]
        world_angles = angles + pyaw + self.laser_offset # 考虑激光雷达偏移
        world_angles = np.arctan2(np.sin(world_angles), np.cos(world_angles)) # 归一化到[-pi, pi]
        valid_ranges = ranges[valid] # 仅保留有效测距数据
        valid_angles = world_angles[valid] # 仅保留有效角度数据

        p_beams = [] # 各激光束符合的概率
        explained_count = 0 # 解释上的测距数量
        total_count = 0

        for r, angle in zip(valid_ranges, valid_angles):
            total_count += 1
            # 计算激光点的世界坐标（range长度*航向角三角关系）
            lx = px + r * math.cos(angle)
            ly = py + r * math.sin(angle)
            # 找到对应的地图索引
            idx = world_to_map(lx, ly, self.map_info)

            distance = None
            if idx is not None and self.distance_field is not None:
                # 如果在地图范围内，而且有欧氏距离数组，则获取距离值
                width = self.map_info.width
                cell_x = idx % width; cell_y = idx // width
                distance = float(self.distance_field[cell_y, cell_x])
            else:
                if idx is None:
                    # 粒子出界处理
                    if self.oob_mode == 'skip':
                        continue
                    elif self.oob_mode == 'count_as_free':
                        p = self.z_rand # 视为自由空间
                        p_beams.append(p) # 记录该激光束的概率
                        continue
                    elif self.oob_mode == 'count_as_occupied':
                        distance = 0.0 # 视为障碍物
                else:
                    # 不能获取距离场，搜索最近占据格并用其距离作为 distance
                    width = self.map_info.width; height = self.map_info.height
                    cell_x = idx % width; cell_y = idx // width
                    min_dist = None
                    # 在一个方形区域内暴力搜索最近占据格子
                    for dx in range(-self.max_searchg_cells, self.max_searchg_cells + 1):
                        for dy in range(-self.max_searchg_cells, self.max_searchg_cells + 1):
                            nx = cell_x + dx
                            ny = cell_y + dy
                            if 0 <= nx < width and 0 <= ny < height:
                                if self.occupancy[ny, nx] == 1: # 占据格
                                    dist = math.sqrt((dx * self.map_info.resolution) ** 2 + (dy * self.map_info.resolution) ** 2) # 计算距离
                                    if min_dist is None or dist < min_dist:
                                        min_dist = dist
                    distance = min_dist
                    #如果仍然没有找到，用最大搜索距离代替
                    if distance is None:
                        distance = self.max_searchg_cells * self.map_info.resolution

            # 计算该激光束的概率
            # 综合概率计算公式为：p = z_hit * p_hit + z_rand * p_rand
            # 直观上理解是，测距值符合预期的概率 + 测距值是随机的概率
            p_hit = (1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)) * math.exp(-0.5 * (distance / self.sigma_hit) ** 2) if distance is not None else 0.0
            p_rand = 1.0 / (scan_msg.range_max - scan_msg.range_min + 1e-6)
            p = self.z_hit * p_hit + self.z_rand * p_rand
            p = max(p, 1e-12)  
            p_beams.append(p)
            if distance is not None and distance < self.explain_tol:
                explained_count += 1

        # 若该位姿处于地图碰撞区域，直接返回极小权重
        if pose_in_collision(candidate_pose[0], candidate_pose[1], map_data, map_info, getattr(self, 'robot_radius', 0.28)):
            return 1e-12
        
        # 若 particle 不能解释足够多的 beams，视为不合理
        explain_frac = float(explained_count) / float(total_count) if total_count > 0 else 0.0
        if explain_frac < self.explain_fraction_thresh:
            return 1e-12

        if len(p_beams) == 0:
            # 没有有效的激光束，返回极小权重
            return 1e-8

        log_mean = float(np.mean(np.log(p_beams)))
        weight = math.exp(log_mean)
        return float(max(weight, 1e-12))



        

        

        

