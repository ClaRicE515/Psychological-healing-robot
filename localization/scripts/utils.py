#!/usr/bin/env python3
# coding: utf-8
import math
import numpy as np
from tf.transformations import euler_from_quaternion


def world_to_map(world_x, world_y, map_info):
    res = map_info.resolution
    width = map_info.width
    height = map_info.height
    origin_x = map_info.origin.position.x
    origin_y = map_info.origin.position.y
    px = int((world_x - origin_x) / res)
    py = int((world_y - origin_y) / res)
    if 0 <= px < width and 0 <= py < height:
        return px + py * width
    else:
        return None


def map_to_world(index, map_info):
    width = map_info.width
    px = index % width
    py = index // width
    wx = map_info.origin.position.x + (px + 0.5) * map_info.resolution
    wy = map_info.origin.position.y + (py + 0.5) * map_info.resolution
    return wx, wy


def quaternion_to_yaw(q):
    quar = [q.x, q.y, q.z, q.w]
    roll, pitch, yaw = euler_from_quaternion(quar)
    return yaw


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def is_obstacle(world_x, world_y, map_data, map_info, thresh=70):
    idx = world_to_map(world_x, world_y, map_info)
    if idx is None:
        return True
    val = map_data[idx]
    if val == -1 or val > thresh:
        return True
    return False


# 新增：在给定位姿周围半径内检查碰撞（圆形近似）
def pose_in_collision(world_x, world_y, map_data, map_info, robot_radius=0.28, thresh=70):
    """
    返回 True 当 pose 在地图边界外或与占据/未知格子发生碰撞。
    """
    if map_info is None or map_data is None:
        return False

    res = map_info.resolution
    width = map_info.width
    height = map_info.height
    ox = map_info.origin.position.x
    oy = map_info.origin.position.y

    # 中心是否出界
    center_idx = world_to_map(world_x, world_y, map_info)
    if center_idx is None:
        return True

    r_cells = int(math.ceil(robot_radius / res))
    px = center_idx % width
    py = center_idx // width
    x0 = max(0, px - r_cells)
    x1 = min(width - 1, px + r_cells)
    y0 = max(0, py - r_cells)
    y1 = min(height - 1, py + r_cells)

    arr = map_data
    for yy in range(y0, y1 + 1):
        for xx in range(x0, x1 + 1):
            idx = xx + yy * width
            val = arr[idx]
            if val == -1 or val > thresh:
                # 检查格心到机器人中心距离
                cell_cx = ox + (xx + 0.5) * res
                cell_cy = oy + (yy + 0.5) * res
                if math.hypot(cell_cx - world_x, cell_cy - world_y) <= robot_radius + 1e-6:
                    return True
    return False