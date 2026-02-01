#!/usr/bin/env python3

# -*- coding: utf-8 -*-
"""
表情生成程序 - 通过绘制几何图形生成表情
坐标系统：x轴向右，y轴向下
"""

# ==================== ①说明/库区 ====================
import numpy as np
from PIL import Image
import os
import math
from translation import usage

# ==================== ②全局常量区 ====================
sqrt = math.sqrt
sqrt5 = sqrt(5)
sqrt2 = sqrt(2)
sqrt10 = sqrt(10)
sqrt3 = sqrt(3)
golden_ratio = (sqrt5 - 1) / 2
silver_ratio = sqrt2 - 1


# ==================== ③绘图函数区 ====================
# 1、圆环相关函数
def load_ring_points():
    """加载保存的圆环点"""
    base_path = r"base/base.npy"

    if os.path.exists(base_path):
        points = np.load(base_path)
        return points
    else:
        print(f"Warning: Ring coordinate file does not exist: {base_path}")
        return None


def draw_ring_from_points(image, ring_points):
    """从加载的点绘制圆环"""
    for x, y in ring_points:
        if 0 <= x < 1024 and 0 <= y < 1024:
            image[y, x] = 0  # 填充黑色
    return image


# 2、椭圆相关函数
def load_ellipses_points():
    """加载保存的椭圆点"""
    ellipses_path = r"base/ellipses.npy"

    if os.path.exists(ellipses_path):
        points = np.load(ellipses_path)
        return points
    else:
        print(f"Warning: Ellipse coordinate file does not exist: {ellipses_path}")
        print(f"Please run the ellipse generation program to create ellipses.npy file")
        return None


def draw_ellipses_from_points(image, ellipse_points):
    """从加载的点绘制椭圆"""
    for x, y in ellipse_points:
        if 0 <= x < 1024 and 0 <= y < 1024:
            image[y, x] = 0  # 填充黑色
    return image


# 3、线段带相关函数
def draw_line_band(image, k):
    """
    绘制线段带

    Parameters:
    image: 图像数组
    k: 线段带斜率
    """
    # 线段带常量参数
    y_avg = 204.455197760054  # 平均y值（中心线在x_mid处的y值）
    x_min = 218.650503679919  # x域左端点
    x_max = 414.217101439973  # x域右端点
    x_mid = 316.433802559946  # x域中点

    # 线段带厚度（常量）
    y_thick = 12.0

    # 迭代x从x_min到x_max
    for x in np.arange(x_min, x_max + 0.01, 1.0):
        x = float(x)

        # 计算中心线在x处的y值
        y_center = k * (x - x_mid) + y_avg

        # 计算y范围
        y_min_val = y_center - y_thick
        y_max_val = y_center + y_thick

        # 确定整数y范围
        y_min_int = max(0, int(np.floor(y_min_val)))
        y_max_int = min(1023, int(np.ceil(y_max_val)))

        if y_min_int <= y_max_int:
            # 迭代y从y_min到y_max
            for y in range(y_min_int, y_max_int + 1):
                # 两个对称点
                point1_x = int(x)
                point2_x = int(1024 - x)

                # 填充黑色
                if 0 <= point1_x < 1024 and 0 <= y < 1024:
                    image[y, point1_x] = 0

                if 0 <= point2_x < 1024 and 0 <= y < 1024:
                    image[y, point2_x] = 0

    return image


# 4、抛物线相关函数
def draw_parabola_region(image, curve1, height1, curve2, height2, x_max):
    """
    绘制抛物线区域

    Parameters:
    image: 图像数组
    curve1, curve2: 抛物线系数
    height1, height2: 抛物线常数项
    x_max: 最大x值（距中心的距离）
    """
    for x_dist in np.arange(0, x_max + 0.01, 1.0):
        x_dist = float(x_dist)

        # 计算两条抛物线的y值
        y1 = curve1 * (x_dist ** 2) + height1
        y2 = curve2 * (x_dist ** 2) + height2

        # 检查y1和y2的关系确定y范围
        # 如果y1 <= y2，绘制从y1到y2的区域；如果y1 > y2，不绘制
        if y1 <= y2:
            # 确定y最小值和最大值
            y_min_val = y1
            y_max_val = y2

            # 迭代y从y_min到y_max
            y_min_int = max(0, int(np.floor(y_min_val)))
            y_max_int = min(1023, int(np.ceil(y_max_val)))

            if y_min_int <= y_max_int:
                for y in range(y_min_int, y_max_int + 1):
                    # 两个对称点
                    point1_x = int(512 - x_dist)
                    point2_x = int(512 + x_dist)

                    # 填充黑色
                    if 0 <= point1_x < 1024 and 0 <= y < 1024:
                        image[y, point1_x] = 0

                    if 0 <= point2_x < 1024 and 0 <= y < 1024:
                        image[y, point2_x] = 0

    return image


# 5、总绘制函数
def create_figure_custom(ring_points, ellipse_points, k, parabola_params, output_path):
    """
    创建自定义图形：包括圆环、椭圆、线段带和抛物线区域
    绘制顺序：先画圆，再画椭圆，而后画线段带，最后画抛物线

    Parameters:
    ring_points: 圆环坐标点
    ellipse_points: 椭圆坐标点
    k: 线段带斜率
    parabola_params: 抛物线参数字典，包含：
        - curve1: 抛物线1系数
        - height1: 抛物线1常数项
        - curve2: 抛物线2系数
        - height2: 抛物线2常数项
        - x_max: 最大x值（距中心的距离）
    output_path: 输出图像路径
    """
    # 创建1024x1024白色图像
    image = np.ones((1024, 1024), dtype=np.uint8) * 255

    # 1. 绘制圆环
    if ring_points is not None:
        image = draw_ring_from_points(image, ring_points)

    # 2. 绘制椭圆（从预生成的坐标点）
    if ellipse_points is not None:
        image = draw_ellipses_from_points(image, ellipse_points)

    # 3. 绘制线段带区域
    image = draw_line_band(image, k)

    # 4. 绘制抛物线区域
    image = draw_parabola_region(
        image,
        parabola_params["curve1"],
        parabola_params["height1"],
        parabola_params["curve2"],
        parabola_params["height2"],
        parabola_params["x_max"]
    )

    # 保存图像
    img = Image.fromarray(image, mode='L')

    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img.save(output_path)
    print(f"\n图片成功保存到: {output_path}")


# ==================== ④参数输入区 ====================

def using_usage():
    k, result = usage()

    return {
        "k": k * golden_ratio,
        "parabola": {
            "curve1": result[0],
            "height1": result[1],
            "curve2": result[2],
            "height2": result[3],
            "x_max": result[4]
        }
    }


# ==================== ⑤主函数区 ====================
def main():
    """主程序"""

    results_dir = r"results"

    # 1. 加载圆环点
    ring_points = load_ring_points()

    # 2. 加载椭圆点
    ellipse_points = load_ellipses_points()

    if ellipse_points is None:
        print("Warning: Cannot load ellipse points, will skip ellipse drawing")
        print("Please run the ellipse generation program to create ellipses.npy file")

    # 3. 生成图像
    # 获取参数
    params = using_usage()

    # 创建图像文件路径
    output_path = os.path.join(results_dir, f"生成的结果.png")

    # 创建图像（参数顺序已调整为：ring_points, ellipse_points, k, parabola_params, output_path）
    create_figure_custom(
        ring_points,
        ellipse_points,
        params["k"],  # 先输入线段带相关参数
        params["parabola"],  # 再输入抛物线相关参数
        output_path
    )

    print("\n程序执行完成！")

    """
    # 显示几何形状统计信息
    if ring_points is not None:
        print(f"\n几何形状统计：")
        print(f"  圆环点数：{len(ring_points)}")

    if ellipse_points is not None:
        print(f"  椭圆点数：{len(ellipse_points)}")
    """


if __name__ == "__main__":
    main()