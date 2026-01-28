#!/usr/bin/env python3
# coding: utf-8
import rospy
import math
import numpy as np
from utils import world_to_map, map_to_world, pose_in_collision

try:
    from scipy.ndimage import distance_transform_edt
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False


class SensorModel:
    def __init__(self, downsample_step=1, laser_offset=0.0,
                 sigma_hit=0.2, z_hit=0.95, z_rand=0.05,
                 oob_mode='skip', max_search_cells=6, robot_radius=0.28):
        self.step = downsample_step
        self.laser_offset = laser_offset
        self.sigma_hit = sigma_hit
        self.z_hit = z_hit
        self.z_rand = z_rand
        self.robot_radius = robot_radius
        self.oob_mode = oob_mode
        self._angles_initialized = False
        self._input_angle_min = 0.0
        self._input_angle_inc = 0.0
        self._num_beams = 0
        self.precomputed_angles = None

        self.map_info = None
        self.occupancy = None 
        self.distance_field = None  
        self.max_search_cells = max_search_cells

        # 判据（当 particle 很难解释 scan 时直接降权）
        self.explain_tol = rospy.get_param('~explain_tol', 0.20)  # 米
        self.explain_fraction_thresh = rospy.get_param('~explain_fraction_thresh', 0.20)  # 最低解释比例

    def set_scan_geometry(self, angle_min, angle_inc, num_beams):
        indices = np.arange(num_beams)
        all_angles = angle_min + indices * angle_inc
        self.precomputed_angles = all_angles[::self.step]
        self._angles_initialized = True
        self._input_angle_min = angle_min
        self._input_angle_inc = angle_inc
        self._num_beams = num_beams

    def set_map(self, map_data, map_info):
        self.map_info = map_info
        w = map_info.width; h = map_info.height
        arr = np.array(map_data, dtype=np.int8).reshape((h, w))
        occ = (arr > 50).astype(np.uint8)
        self.occupancy = occ
        if _HAVE_SCIPY:
            try:
                dist_cells = distance_transform_edt(1 - occ)
                self.distance_field = dist_cells * map_info.resolution
            except Exception:
                self.distance_field = None
        else:
            self.distance_field = None

    def get_weight(self, candidate_pose, scan_msg, map_data, map_info):
        if not self._angles_initialized:
            try:
                self.set_scan_geometry(scan_msg.angle_min, scan_msg.angle_increment, len(scan_msg.ranges))
            except Exception:
                return 1e-8

        full_ranges = np.array(scan_msg.ranges)
        ranges = full_ranges[::self.step]
        angles = self.precomputed_angles
        if angles.shape[0] != ranges.shape[0]:
            L = min(angles.shape[0], ranges.shape[0])
            angles = angles[:L]; ranges = ranges[:L]

        valid_mask = (~np.isnan(ranges)) & (~np.isinf(ranges)) & (ranges <= scan_msg.range_max) & (ranges >= scan_msg.range_min)
        if not np.any(valid_mask):
            return 1e-8

        px = candidate_pose[0]; py = candidate_pose[1]; pth = candidate_pose[2]
        world_angles = pth + self.laser_offset + angles
        world_angles = np.arctan2(np.sin(world_angles), np.cos(world_angles))
        valid_ranges = ranges[valid_mask]; valid_angles = world_angles[valid_mask]

        p_beams = []
        explained = 0
        total = 0
        for r_meas, ang in zip(valid_ranges, valid_angles):
            total += 1
            hx = px + r_meas * math.cos(ang)
            hy = py + r_meas * math.sin(ang)
            idx = world_to_map(hx, hy, self.map_info)
            d = None
            if idx is not None and self.distance_field is not None:
                w = self.map_info.width
                cell_x = idx % w; cell_y = idx // w
                d = float(self.distance_field[cell_y, cell_x])
            else:
                if idx is None:
                    if self.oob_mode == 'skip':
                        continue
                    elif self.oob_mode == 'count_as_free':
                        p = self.z_rand
                        p_beams.append(p); continue
                    elif self.oob_mode == 'count_as_obstacle':
                        d = 0.0
                else:
                    w = self.map_info.width; cell_x = idx % w; cell_y = idx // w
                    found = False
                    R = self.max_search_cells
                    arr = self.occupancy
                    hgt, wid = arr.shape
                    min_x = max(0, cell_x - R); max_x = min(wid-1, cell_x + R)
                    min_y = max(0, cell_y - R); max_y = min(hgt-1, cell_y + R)
                    sub = arr[min_y:max_y+1, min_x:max_x+1]
                    if np.any(sub):
                        ys, xs = np.nonzero(sub)
                        xs_world = (min_x + xs + 0.5) * self.map_info.resolution + self.map_info.origin.position.x
                        ys_world = (min_y + ys + 0.5) * self.map_info.resolution + self.map_info.origin.position.y
                        dists = np.hypot(xs_world - hx, ys_world - hy)
                        d = float(np.min(dists))
                        found = True
                    if not found:
                        d = float(R * self.map_info.resolution)  

            p_hit = (1.0 / (math.sqrt(2.0 * math.pi) * self.sigma_hit)) * math.exp(-0.5 * (d / self.sigma_hit) ** 2) if d is not None else 0.0
            p_rand = 1.0 / (scan_msg.range_max - scan_msg.range_min + 1e-6)
            p = self.z_hit * p_hit + self.z_rand * p_rand
            p = max(p, 1e-12)  
            p_beams.append(p)

        # 若该位姿处于地图碰撞区域，直接返回极小权重
        if pose_in_collision(candidate_pose[0], candidate_pose[1], map_data, map_info, getattr(self, 'robot_radius', 0.28)):
            return 1e-12

        # 若 particle 不能解释足够多的 beams，视为不合理
        explain_frac = float(explained) / float(total) if total > 0 else 0.0
        if explain_frac < self.explain_fraction_thresh:
            return 1e-12

        if len(p_beams) == 0:
            return 1e-8

        log_mean = float(np.mean(np.log(p_beams)))
        weight = math.exp(log_mean)
        return float(max(weight, 1e-12))

