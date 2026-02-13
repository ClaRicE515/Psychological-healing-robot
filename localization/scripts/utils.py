#!/usr/bin/env python3
# coding: utf-8
import math
import numpy as np
from tf.transformations import euler_from_quaternion

def world_to_map(world_x, world_y, map_info):
    """
    将世界坐标转换为地图坐标
    
    :param world_x: 世界坐标的x值
    :param world_y: 世界坐标的y值
    :param map_info: 地图信息对象，包含原点位置，map分辨率等
    """
    # 提取地图信息
    resolution = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    width = map_info.width
    height = map_info.height

    # 计算地图坐标
    # 现在用的是int强制转换，如果后续发现问题，可以改成math.floor或者math.ceil或整除
    px =  int((world_x - origin_x) / resolution)
    py =  int((world_y - origin_y) / resolution)


    if px >= 0 and px < width and py >= 0 and py < height:
        return py * width + px
    else:
        return None
    

def map_to_world(index, map_info):
    """
    将地图坐标转换为世界坐标

    :param index: 在map中的索引 (px, py), 满足算式:index = py * width + px
    所以 px = index % width, py = index // width‘
    
    :param map_info: 地图信息对象，包含原点位置，map分辨率等
    """
    # 提取地图信息
    resolution = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    width = map_info.width

    px = index % width
    py = index // width

    # 计算世界坐标，取格子中心点，所以加0.5
    world_x = (px + 0.5) * resolution + origin_x 
    world_y = (py + 0.5) * resolution + origin_y 

    return world_x, world_y

def quaternion_to_yaw(quaternion):
    """
    将四元数转换为偏航角（yaw）
    直接调用tf库的函数进行转换
    
    :param quaternion: 接收一个四元数对象，x, y, z, w
    """
    quar = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]
    roll, pitch, yaw = euler_from_quaternion(quar)
    return yaw

def wrap_angle(angle):
    """
    将角度限制在[-pi, pi]范围内
    利用atan2函数的周期性实现角度包裹

    :param angle: 输入角度（弧度）
    """
    return math.atan2(math.sin(angle), math.cos(angle))

def is_obstacle(world_x, world_y, map_data, map_info, threshold=70):
    """
    判断给定的世界坐标是否为障碍物
    
    :param world_x: world坐标的x值
    :param world_y: world坐标的y值
    :param map_data: 地图数据数组
    :param map_info: 地图信息对象，包含原点位置，map分辨率等
    """

    index = world_to_map(world_x, world_y, map_info) # 获取地图坐标
    if index is None:
        return True  # 超出地图范围，视为障碍物
    val = map_data[index]
    if val >= threshold or val == -1:
        return True  # 障碍物或未知区域
    else:
        return False  # 自由空间


def pose_in_collision(world_x, world_y, map_data, map_info, robot_radius = 0.28, threshold = 70):
    """
    判断机器人在给定位置是否与障碍物发生碰撞
    
    :param world_x: 世界坐标的x值
    :param world_y: 世界坐标的y值
    :param map_data: map的数据数组
    :param map_info: 有关地图的信息，包括分辨率，原点位置等
    :param robot_radius: 机器人实体半径，单位：米
    :param threshold: 障碍物阈值
    """
    r2 = (robot_radius + 1e-6) ** 2 # 预先计算距离平方，优化了之前开根的成本
    if map_info is None or map_data is None:
        return False
    
    # 获取map信息
    resolution = map_info.resolution
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    width = map_info.width
    height = map_info.height

    center_index = world_to_map(world_x, world_y, map_info)
    if center_index is None:
        return True  # 超出地图范围，视为碰撞
    
    # 考虑机器人半径，计算需要检查的格子范围
    # 使用 ceil 保证覆盖到半径所及的所有格子（向上取整）
    radius_in_cells = int(math.ceil(robot_radius / resolution))
    center_px = center_index % width
    center_py = center_index // width

    # 需要检查的格子范围
    # 这里其实是一个正方形范围，但后续会在循环中判断是否在圆形范围内
    min_px = max(0, center_px - radius_in_cells)
    max_px = min(width - 1, center_px + radius_in_cells)
    min_py = max(0, center_py - radius_in_cells)
    max_py = min(height - 1, center_py + radius_in_cells)

    for px in range(min_px, max_px + 1):
        for py in range(min_py, max_py + 1):
            index = py * width + px
            val = map_data[index]
            if val == -1 or val > threshold:
                cell_center_x = origin_x + (px + 0.5) * resolution
                cell_center_y = origin_y + (py + 0.5) * resolution
                dx = cell_center_x - world_x
                dy = cell_center_y - world_y

                # 判断出现上述碰撞的点在不在圆形范围内
                if dx ** 2 + dy ** 2 <= r2:
                    return True
            
    return False
    