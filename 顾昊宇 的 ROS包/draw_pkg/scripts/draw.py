#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表情生成程序 - ROS 节点版（路径独立）
通过 rospkg 获取包路径，确保在任何工作目录下都能找到资源文件
"""

import numpy as np
from PIL import Image
import os
import math
import time
import cv2
import rospy
from std_msgs.msg import String, Float32MultiArray
import rospkg  # 新增：用于获取包路径

# 从新的 translation 模块导入 new_usage
from translation import new_usage

# ==================== 全局常量 ====================
sqrt = math.sqrt
sqrt5 = sqrt(5)
sqrt2 = sqrt(2)
sqrt10 = sqrt(10)
sqrt3 = sqrt(3)
golden_ratio = (sqrt5 - 1) / 2
silver_ratio = sqrt2 - 1

# 全局缓存：预加载的几何数据
_ring_points = None
_ellipse_points = None

# 获取包路径（draw_pkg）
rospack = rospkg.RosPack()
package_path = rospack.get_path('draw_pkg')
scripts_path = os.path.join(package_path, 'scripts')  # 所有资源都在 scripts 下


# ==================== 几何数据加载（使用绝对路径）====================
def load_geometry_data():
    """加载所有几何数据（一次性加载，提高性能）"""
    global _ring_points, _ellipse_points
    if _ring_points is None:
        _ring_points = load_ring_points()
    if _ellipse_points is None:
        _ellipse_points = load_ellipses_points()
    return _ring_points, _ellipse_points


def load_ring_points():
    """加载保存的圆环点（使用包内绝对路径）"""
    base_path = os.path.join(scripts_path, "base", "base.npy")
    if os.path.exists(base_path):
        return np.load(base_path)
    else:
        rospy.logwarn(f"圆环坐标文件不存在: {base_path}")
        return None


def load_ellipses_points():
    """加载保存的椭圆点（使用包内绝对路径）"""
    ellipses_path = os.path.join(scripts_path, "base", "ellipses.npy")
    if os.path.exists(ellipses_path):
        return np.load(ellipses_path)
    else:
        rospy.logwarn(f"椭圆坐标文件不存在: {ellipses_path}")
        return None


# ==================== 底层绘图函数（保持不变）====================
def draw_ring_from_points(image, ring_points):
    for x, y in ring_points:
        if 0 <= x < 1024 and 0 <= y < 1024:
            image[y, x] = 0
    return image


def draw_ellipses_from_points(image, ellipse_points):
    for x, y in ellipse_points:
        if 0 <= x < 1024 and 0 <= y < 1024:
            image[y, x] = 0
    return image


def draw_line_band(image, k):
    y_avg = 204.455197760054
    x_min = 218.650503679919
    x_max = 414.217101439973
    x_mid = 316.433802559946
    y_thick = 12.0

    for x in np.arange(x_min, x_max + 0.01, 1.0):
        x = float(x)
        y_center = k * (x - x_mid) + y_avg
        y_min_val = y_center - y_thick
        y_max_val = y_center + y_thick
        y_min_int = max(0, int(np.floor(y_min_val)))
        y_max_int = min(1023, int(np.ceil(y_max_val)))

        if y_min_int <= y_max_int:
            for y in range(y_min_int, y_max_int + 1):
                point1_x = int(x)
                point2_x = int(1024 - x)
                if 0 <= point1_x < 1024 and 0 <= y < 1024:
                    image[y, point1_x] = 0
                if 0 <= point2_x < 1024 and 0 <= y < 1024:
                    image[y, point2_x] = 0
    return image


def draw_parabola_region(image, curve1, height1, curve2, height2, x_max):
    for x_dist in np.arange(0, x_max + 0.01, 1.0):
        x_dist = float(x_dist)
        y1 = curve1 * (x_dist ** 2) + height1
        y2 = curve2 * (x_dist ** 2) + height2

        if y1 <= y2:
            y_min_int = max(0, int(np.floor(y1)))
            y_max_int = min(1023, int(np.ceil(y2)))
            if y_min_int <= y_max_int:
                for y in range(y_min_int, y_max_int + 1):
                    point1_x = int(512 - x_dist)
                    point2_x = int(512 + x_dist)
                    if 0 <= point1_x < 1024 and 0 <= y < 1024:
                        image[y, point1_x] = 0
                    if 0 <= point2_x < 1024 and 0 <= y < 1024:
                        image[y, point2_x] = 0
    return image


# ==================== 核心绘图函数（保持不变）====================
def create_expression_image(k, parabola_params, ring_points=None, ellipse_points=None):
    image = np.ones((1024, 1024), dtype=np.uint8) * 255

    if ring_points is not None:
        image = draw_ring_from_points(image, ring_points)
    if ellipse_points is not None:
        image = draw_ellipses_from_points(image, ellipse_points)
    image = draw_line_band(image, k)
    image = draw_parabola_region(
        image,
        parabola_params["curve1"],
        parabola_params["height1"],
        parabola_params["curve2"],
        parabola_params["height2"],
        parabola_params["x_max"]
    )
    return image


# ==================== 中性表情参数（保持不变）====================
def get_neutral_expression_params():
    parabola_curve1 = (1 - sqrt2) / 512
    parabola_height1 = 256 * (5 - sqrt5) + 128 * (3 - sqrt5) * (sqrt2 - 1) * sqrt5
    parabola_curve2 = (1 - sqrt2) / 256
    parabola_height2 = 128 * (9 - 3 * sqrt5 + sqrt2 + sqrt10)
    x_max_parabola = 256 * (sqrt5 - 1)
    k = 0

    return {
        "k": k,
        "parabola": {
            "curve1": parabola_curve1,
            "height1": parabola_height1,
            "curve2": parabola_curve2,
            "height2": parabola_height2,
            "x_max": x_max_parabola
        }
    }


# ==================== 情感数据 → 几何参数（临时切换工作目录）====================
def emotion_to_expression_params(emotion_probs, emotion_label=None):
    """
    调用 new_usage 将情感概率转换为几何参数
    为确保 translation 内部的相对路径正确，临时切换到 scripts_path
    """
    cwd = os.getcwd()  # 保存当前工作目录
    try:
        # 切换到 scripts 目录，使 translation 能找到 base/ 和 train/
        os.chdir(scripts_path)
        D, result = new_usage(emotion_probs, emotion_label)
        k = D * golden_ratio
        return {
            "k": k,
            "parabola": {
                "curve1": result[0],
                "height1": result[1],
                "curve2": result[2],
                "height2": result[3],
                "x_max": result[4]
            }
        }
    except Exception as e:
        rospy.logerr(f"调用 new_usage 失败: {e}")
        return get_neutral_expression_params()
    finally:
        os.chdir(cwd)  # 恢复原工作目录


# ==================== ROS 节点类 ====================
class ExpressionDrawer:
    def __init__(self):
        self.current_image = None
        self.new_label = None
        self.new_probs = None
        self.ring_points, self.ellipse_points = load_geometry_data()

        # 初始化中性表情
        neutral_params = get_neutral_expression_params()
        self.current_image = create_expression_image(
            neutral_params["k"],
            neutral_params["parabola"],
            self.ring_points,
            self.ellipse_points
        )

        cv2.namedWindow('Machine Expression', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Machine Expression', 512, 512)

        rospy.Subscriber('/emotion_label', String, self.label_callback)
        rospy.Subscriber('/emotion_probs', Float32MultiArray, self.probs_callback)
        rospy.loginfo("表情绘制节点已启动，等待情感数据...")

    def label_callback(self, msg):
        self.new_label = msg.data

    def probs_callback(self, msg):
        self.new_probs = np.array(msg.data, dtype=np.float64)

    def process_new_data(self):
        if self.new_label is not None and self.new_probs is not None:
            rospy.loginfo(f"收到情感数据: {self.new_label}")
            params = emotion_to_expression_params(self.new_probs, self.new_label)
            self.current_image = create_expression_image(
                params["k"],
                params["parabola"],
                self.ring_points,
                self.ellipse_points
            )
            self.new_label = None
            self.new_probs = None

    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():
            self.process_new_data()
            cv2.imshow('Machine Expression', self.current_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                rospy.loginfo("用户请求退出")
                break
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"expression_{timestamp}.png"
                cv2.imwrite(filename, self.current_image)
                rospy.loginfo(f"表情已保存为 {filename}")

            rate.sleep()

        cv2.destroyAllWindows()


# ==================== 主函数 ====================
def main():
    rospy.init_node('expression_drawer', anonymous=True)
    drawer = ExpressionDrawer()
    drawer.run()
    rospy.loginfo("表情绘制节点已退出")


if __name__ == "__main__":
    main()