#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
情感识别系统（ROS 节点版，路径独立）
通过 rospkg 获取包路径，确保在任何工作目录下都能找到资源文件
"""

# ==================== ①库导入区 ====================
import os
from keras.preprocessing.image import img_to_array
import imutils
import cv2
import time
import config
from keras.models import load_model
import numpy as np
import rospy
from std_msgs.msg import String, Float32MultiArray
import rospkg  # 新增：用于获取包路径

# ==================== ②配置加载区（路径独立化）====================
# 获取当前包（test_pkg）的绝对路径
rospack = rospkg.RosPack()
package_path = rospack.get_path('test_pkg')

# 构建资源文件的绝对路径（假设它们位于 scripts 子文件夹内）
# 注意：根据用户描述，所有程序和子文件夹都在 scripts 中
scripts_path = os.path.join(package_path, 'scripts')
cascade_relative = config.CASCADE_PATH  # 例如 "haarcascade_files/haarcascade_frontalface_default.xml"
model_relative = config.MODEL_PATH      # 例如 "models/_mini_XCEPTION.102-0.66.hdf5"

detection_model_path = os.path.join(scripts_path, cascade_relative)
emotion_model_path = os.path.join(scripts_path, model_relative)

# 加载模型（使用绝对路径）
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# ==================== ③初始化函数区 ====================
def init_camera():
    """初始化摄像头，返回摄像头对象"""
    camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if camera.isOpened():
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("摄像头初始化成功")
    else:
        print("警告：摄像头初始化失败")
    return camera

# ==================== ④持久化状态变量初始化区 ====================
last_face_box = (0, 0, 0, 0)
has_face_ever_detected = False
frames_since_last_detection = 0
MAX_FRAMES_WITHOUT_DETECTION = 30

current_label = ""
preds = np.zeros(len(EMOTIONS))

frame_count = 0
camera_error_count = 0
MAX_CAMERA_ERRORS = 10

# ==================== ⑤ROS 节点初始化与窗口初始化 ====================
print("=" * 60)
print("情感识别系统（ROS 节点版 - 路径独立）")
print("按 'q' 键退出程序")
print("=" * 60)

rospy.init_node('emotion_publisher', anonymous=True)

label_pub = rospy.Publisher('emotion_label', String, queue_size=1)
prob_pub = rospy.Publisher('emotion_probs', Float32MultiArray, queue_size=1)
rospy.loginfo("情感发布节点已启动，等待识别结果...")

cv2.namedWindow('Your Face', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Your Face', 600, 400)
cv2.namedWindow("Emotion Probabilities", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Probabilities", 400, 300)

camera = init_camera()
if not camera.isOpened():
    print("错误：无法打开摄像头，请检查连接")
    exit(1)

# 预读一帧，确保摄像头正常工作
_, _ = camera.read()
camera.release()
time.sleep(2)
camera = init_camera()

print("系统准备就绪，开始识别...")

# ==================== ⑥主循环处理区 ====================
while not rospy.is_shutdown():
    ret, frame = camera.read()
    
    if not ret:
        camera_error_count += 1
        print(f"摄像头读取失败，尝试重新初始化 ({camera_error_count}/{MAX_CAMERA_ERRORS})")
        if camera_error_count >= MAX_CAMERA_ERRORS:
            print("摄像头连续失败次数过多，退出程序")
            break
        camera.release()
        time.sleep(2)
        print("重新初始化摄像头...")
        camera = init_camera()
        if not camera.isOpened():
            print("摄像头重新初始化失败")
            time.sleep(2)
            continue
        camera_error_count = 0
        print("摄像头重新初始化成功")
        continue

    camera_error_count = 0

    frame = imutils.resize(frame, width=300)
    frameClone = frame.copy()
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    frame_count += 1
    should_process = (frame_count % config.SKIP_FRAMES == 0)
    
    current_has_face = has_face_ever_detected
    current_face_box = last_face_box
    current_emotion_label = current_label
    current_emotion_probs = preds.copy()
    
    if should_process:
        faces = face_detection.detectMultiScale(
            gray,
            scaleFactor=config.SCALE_FACTOR,
            minNeighbors=config.MIN_NEIGHBORS,
            minSize=config.MIN_SIZE,
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            current_face_box = faces
            (fX, fY, fW, fH) = current_face_box

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (config.IMG_SIZE, config.IMG_SIZE))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            try:
                current_emotion_probs = emotion_classifier.predict(roi)[0]
                current_emotion_label = EMOTIONS[current_emotion_probs.argmax()]
                
                print(f"[{time.strftime('%H:%M:%S')}] 用户情感: {current_emotion_label}")
                
                # ROS 发布
                label_msg = String()
                label_msg.data = current_emotion_label
                label_pub.publish(label_msg)
                
                prob_msg = Float32MultiArray()
                prob_msg.data = current_emotion_probs.tolist()
                prob_pub.publish(prob_msg)
                
                rospy.loginfo(f"已发布情感: {current_emotion_label}")
                
                # 更新状态
                last_face_box = current_face_box
                current_label = current_emotion_label
                preds = current_emotion_probs.copy()
                has_face_ever_detected = True
                frames_since_last_detection = 0
                
            except Exception as e:
                print(f"情感预测错误: {e}")
        else:
            frames_since_last_detection += 1
            if frames_since_last_detection > MAX_FRAMES_WITHOUT_DETECTION:
                has_face_ever_detected = False
                current_label = ""
                preds = np.zeros(len(EMOTIONS))
    else:
        frames_since_last_detection += 1
        if frames_since_last_detection > MAX_FRAMES_WITHOUT_DETECTION:
            has_face_ever_detected = False
            current_label = ""
            preds = np.zeros(len(EMOTIONS))
    
    # 绘图与展示
    if has_face_ever_detected:
        (fX, fY, fW, fH) = last_face_box
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            color = (0, 165, 255) if emotion == current_label else (0, 0, 255)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), color, -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.putText(frameClone, current_label, (fX, fY - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)
        
        status_text = "Real-time" if should_process else f"Cached ({frames_since_last_detection}/{MAX_FRAMES_WITHOUT_DETECTION})"
        status_color = (0, 255, 0) if should_process else (0, 255, 255)
        cv2.putText(frameClone, status_text, (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1)
        
        cv2.putText(canvas, f"Primary: {current_label}", (10, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    else:
        cv2.putText(frameClone, "No Face Detected", (80, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(canvas, "Please Face Camera", (60, 125),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(canvas, "Keep Distance", (80, 160),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow('Your Face', frameClone)
    cv2.imshow("Emotion Probabilities", canvas)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("用户请求退出")
        break
    elif key == ord('v'):
        if has_face_ever_detected:
            print(f"当前主要情感: {current_label}")
            print(f"概率分布: {preds}")
    elif key == ord('d'):
        print("=" * 60)
        print(f"调试信息 - 时间: {time.strftime('%H:%M:%S')}")
        print(f"  帧计数: {frame_count} (跳过帧: {config.SKIP_FRAMES})")
        print(f"  当前为处理帧: {'是' if should_process else '否'}")
        print(f"  曾经检测到人脸: {'是' if has_face_ever_detected else '否'}")
        print(f"  距离上次检测: {frames_since_last_detection} 帧")
        print(f"  主要情感: {current_label if current_label else '无'}")
        print("=" * 60)

# ==================== ⑦资源清理区 ====================
camera.release()
cv2.destroyAllWindows()
rospy.signal_shutdown("节点正常退出")
print("程序已退出")
print("=" * 60)