#!/usr/bin/env python3
# coding: utf-8
import rospy
import numpy as np
import math
import tf
from tf.transformations import quaternion_from_euler

from nav_msgs.msg import Odometry, OccupancyGrid
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, PoseArray, Pose

from utils import quaternion_to_yaw, wrap_angle, pose_in_collision, map_to_world
from sensor_model import SensorModel
from motion_model import MotionModel


class ParticleFilterNode:
    """
    AMCL-like particle filter node.
    Subscribes: /map, /odom, /scan, /initialpose
    Publishes: /pf_pose (PoseWithCovarianceStamped), /pf_particles (PoseArray)
    Broadcasts: TF map->odom
    """
    def __init__(self):
        rospy.init_node('localization')
        # 机器人半径（用于碰撞检测）
        self.robot_radius = rospy.get_param('~robot_radius', 0.28)

        # 参数
        self.num_particles = rospy.get_param('~num_particles', 600)
        self.downsample = rospy.get_param('~laser_downsample', 1)
        self.laser_offset = rospy.get_param('~laser_offset', 0.0)
        self.oob_mode = rospy.get_param('~oob_mode', 'skip')  # 'skip'|'count_as_free'|'count_as_obstacle'
        self.sigma_hit = rospy.get_param('~sigma_hit', 0.2)
        self.z_hit = rospy.get_param('~z_hit', 0.95)
        self.z_rand = rospy.get_param('~z_rand', 0.03)
        self.resample_ratio = rospy.get_param('~resample_ratio', 0.4)

        self.particles = np.zeros((self.num_particles, 3))
        self.weights = np.ones(self.num_particles) / float(self.num_particles)
        self.map_data = None
        self.map_info = None
        self.map_received = False
        self.initialized = False
        self.last_odom = None
        self.initial_pose_hyp = None
        self.force_update_after_initialpose = False

        # 模型
        self.motion_model = MotionModel()
        self.sensor_model = SensorModel(downsample_step=self.downsample,
                                        laser_offset=self.laser_offset,
                                        oob_mode=self.oob_mode,
                                        robot_radius=self.robot_radius)

        # ROS·订阅与发布
        rospy.Subscriber('/map', OccupancyGrid, self.map_callback, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=5)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=5)
        rospy.Subscriber('/initialpose', PoseWithCovarianceStamped, self.initial_pose_callback, queue_size=1)

        self.pose_pub = rospy.Publisher('/pf_pose', PoseWithCovarianceStamped, queue_size=1)
        self.particles_pub = rospy.Publisher('/pf_particles', PoseArray, queue_size=1)
        self.tf_br = tf.TransformBroadcaster()

        rospy.loginfo("ParticleFilterNode started with %d particles", self.num_particles)

    def initial_pose_callback(self, msg):
        self.initial_pose_hyp = msg
        self.force_update_after_initialpose = True
        rospy.loginfo("Received initialpose (cached).")
        if self.map_received:
            self.apply_initial_pose()

    def apply_initial_pose(self):
        if self.initial_pose_hyp is None:
            return
        p = self.initial_pose_hyp.pose.pose
        x = p.position.x; y = p.position.y; yaw = quaternion_to_yaw(p.orientation)
        rospy.loginfo("Applying initialpose: x=%.3f y=%.3f yaw=%.3f", x, y, yaw)
        self.seed_at(x, y, yaw)
        self.initial_pose_hyp = None
        self.force_update_after_initialpose = True

    def seed_at(self, x, y, yaw, std_xy=0.2, std_yaw=0.1):
        self.particles[:, 0] = np.random.normal(x, std_xy, self.num_particles)
        self.particles[:, 1] = np.random.normal(y, std_xy, self.num_particles)
        self.particles[:, 2] = np.random.normal(yaw, std_yaw, self.num_particles)
        self.particles[:, 2] = np.arctan2(np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2]))
        self.weights = np.ones(self.num_particles) / float(self.num_particles)
        self.initialized = True

    def seed_uniform_free_space(self):
        if self.map_data is None or self.map_info is None:
            rospy.logwarn("No map to seed.")
            return
        free_idx = [i for i,v in enumerate(self.map_data) if v == 0]
        if not free_idx:
            rospy.logwarn("No free cells found; seeding around origin.")
            self.seed_at(0.0, 0.0, 0.0)
            return
        picks = np.random.choice(free_idx, size=self.num_particles, replace=True)
        w = self.map_info.width; res = self.map_info.resolution
        ox = self.map_info.origin.position.x; oy = self.map_info.origin.position.y
        xs = []; ys = []
        for idx in picks:
            px = idx % w; py = idx // w
            xs.append(ox + (px + 0.5)*res); ys.append(oy + (py + 0.5)*res)
        self.particles[:,0] = np.array(xs); self.particles[:,1] = np.array(ys)
        self.particles[:,2] = np.random.uniform(-math.pi, math.pi, self.num_particles)
        self.weights[:] = 1.0 / float(self.num_particles)
        self.initialized = True

    def map_callback(self, msg):
        self.map_data = msg.data
        self.map_info = msg.info
        first_map = not self.map_received
        self.map_received = True
        rospy.loginfo_once("Map received.")
        # 设置传感器模型的地图
        self.sensor_model.set_map(self.map_data, self.map_info)
        # 初始化粒子
        if self.initial_pose_hyp is not None:
            self.apply_initial_pose()
            return
        if not self.initialized:
            if self.last_odom is not None:
                ox = self.last_odom.pose.pose.position.x
                oy = self.last_odom.pose.pose.position.y
                oyaw = quaternion_to_yaw(self.last_odom.pose.pose.orientation)
                rospy.loginfo("Seeding around last odom.")
                self.seed_at(ox, oy, oyaw)
            else:
                rospy.loginfo("Seeding uniformly in free space.")
                self.seed_uniform_free_space()

    def odom_callback(self, msg):
        if self.last_odom is None:
            self.last_odom = msg
            return
        prev = self.last_odom.pose.pose; curr = msg.pose.pose
        prev_yaw = quaternion_to_yaw(prev.orientation); curr_yaw = quaternion_to_yaw(curr.orientation)
        dx_world = curr.position.x - prev.position.x
        dy_world = curr.position.y - prev.position.y
        dx_local = dx_world * math.cos(-prev_yaw) - dy_world * math.sin(-prev_yaw)
        dy_local = dx_world * math.sin(-prev_yaw) + dy_world * math.cos(-prev_yaw)
        dyaw = wrap_angle(curr_yaw - prev_yaw)
        self.particles = self.motion_model.predict(self.particles, dx_local, dy_local, dyaw)
        self.last_odom = msg

    def scan_callback(self, msg):
        if not (self.map_received and self.initialized):
            if not (self.force_update_after_initialpose and self.map_received and self.initialized):
                return
        # 计算权重
        for i in range(self.num_particles):
            self.weights[i] = self.sensor_model.get_weight(self.particles[i], msg, self.map_data, self.map_info)

        # 立即把与障碍冲突的粒子权重置为很小（加快剔除）
        for i in range(self.num_particles):
            if pose_in_collision(self.particles[i,0], self.particles[i,1], self.map_data, self.map_info, self.robot_radius):
                self.weights[i] = 1e-12

        # 规范化权重
        wsum = np.sum(self.weights)
        if wsum <= 0 or not np.isfinite(wsum):
            self.weights[:] = 1.0 / float(self.num_particles)
        else:
            self.weights /= wsum
        # 估计位姿 (加权)
        mean_x = float(np.sum(self.weights * self.particles[:,0]))
        mean_y = float(np.sum(self.weights * self.particles[:,1]))
        s = float(np.sum(self.weights * np.sin(self.particles[:,2])))
        c = float(np.sum(self.weights * np.cos(self.particles[:,2])))
        mean_yaw = math.atan2(s, c)
        # 基于 N_eff 进行重采样
        n_eff = 1.0 / np.sum(self.weights ** 2)
        if n_eff < (self.num_particles * self.resample_ratio):
            self._low_variance_resample()
        # 发布
        self.publish_particles()
        self.publish_pose([mean_x, mean_y, mean_yaw])
        self.force_update_after_initialpose = False

    def _low_variance_resample(self):
        N = self.num_particles
        positions = (np.arange(N) + np.random.rand()) / N
        cumulative = np.cumsum(self.weights)
        newp = np.zeros_like(self.particles)
        i = 0
        for j,pos in enumerate(positions):
            while pos > cumulative[i]:
                i += 1
            newp[j] = self.particles[i]
        self.particles = newp
        self.weights[:] = 1.0/float(N)
        rospy.logdebug("Resampled particles low-variance.")

    def publish_particles(self):
        pa = PoseArray()
        pa.header.stamp = rospy.Time.now()
        pa.header.frame_id = "map"
        pa.poses = []
        for x,y,yaw in self.particles:
            p = Pose()
            p.position.x = float(x); p.position.y = float(y); p.position.z = 0.0
            q = quaternion_from_euler(0,0,float(yaw))
            p.orientation = Quaternion(*q)
            pa.poses.append(p)
        self.particles_pub.publish(pa)

    def publish_pose(self, pose):
        est_x, est_y, est_yaw = pose
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = "map"
        msg.pose.pose.position.x = est_x; msg.pose.pose.position.y = est_y
        msg.pose.pose.orientation = Quaternion(*quaternion_from_euler(0,0,est_yaw))
        # 计算协方差
        var_x = float(np.sum(self.weights * (self.particles[:,0] - est_x)**2))
        var_y = float(np.sum(self.weights * (self.particles[:,1] - est_y)**2))
        s = float(np.sum(self.weights * np.sin(self.particles[:,2])))
        c = float(np.sum(self.weights * np.cos(self.particles[:,2])))
        var_yaw = max(0.0, 1.0 - (s*s + c*c))
        msg.pose.covariance[0] = var_x
        msg.pose.covariance[7] = var_y
        msg.pose.covariance[35] = var_yaw
        self.pose_pub.publish(msg)
        # 广播 TF
        if self.last_odom is None:
            return
        def pose_to_mat(x,y,yaw):
            c = math.cos(yaw); s = math.sin(yaw)
            return np.array([[c, -s, x],[s, c, y],[0,0,1]])
        T_map_base = pose_to_mat(est_x, est_y, est_yaw)
        od = self.last_odom.pose.pose
        T_odom_base = pose_to_mat(od.position.x, od.position.y, quaternion_to_yaw(od.orientation))
        T_map_odom = T_map_base.dot(np.linalg.inv(T_odom_base))
        tx = float(T_map_odom[0,2]); ty = float(T_map_odom[1,2])
        q_map_odom = quaternion_from_euler(0,0, math.atan2(T_map_odom[1,0], T_map_odom[0,0]))
        self.tf_br.sendTransform((tx,ty,0.0), q_map_odom, rospy.Time.now(), "odom", "map")


if __name__ == '__main__':
    try:
        node = ParticleFilterNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass