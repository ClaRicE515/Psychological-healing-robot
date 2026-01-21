from keras.preprocessing.image import img_to_array
import imutils
import cv2
import time
from data_logger import EmotionLogger
import config  # 确保你已经创建了 config.py
from keras.models import load_model
import numpy as np

# 1. 从 config 加载配置
detection_model_path = config.CASCADE_PATH
emotion_model_path = config.MODEL_PATH

# 2. 加载模型
face_detection = cv2.CascadeClassifier(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# 3. 初始化控制变量
frame_count = 0
current_label = ""
preds = np.zeros(len(EMOTIONS)) # 初始概率分布
fX, fY, fW, fH = 0, 0, 0, 0     # 初始人脸位置

cv2.namedWindow('your_face')
camera = cv2.VideoCapture(0)

# --- 初始化日志相关变量 ---

# 1. 获取当前时间作为文件名的一部分，确保唯一性
# 格式为：20231027_153005 (年月日_时分秒)
current_time_str = time.strftime("%Y%m%d_%H%M%S")
unique_filename = f"emotion_{current_time_str}.txt"

# 2. 实例化 Logger，指定存放的文件夹名为 "data_logs"
logger = EmotionLogger(folder="data_logs", filename=unique_filename)

last_log_time = time.time()
collected_preds = []

while True:
    # 1. 读取摄像头帧
    ret, frame = camera.read()
    if not ret:
        break

    # 预处理画面显示尺寸
    frame = imutils.resize(frame, width=300)
    frameClone = frame.copy()
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. 跳帧逻辑：只有在特定帧才运行 AI
    frame_count += 1
    if frame_count % config.SKIP_FRAMES == 0:
        # 人脸检测
        faces = face_detection.detectMultiScale(
            gray, 
            scaleFactor=config.SCALE_FACTOR, 
            minNeighbors=config.MIN_NEIGHBORS, 
            minSize=config.MIN_SIZE, 
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) > 0:
            # 选取最大的脸
            faces = sorted(faces, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = faces

            # 提取 ROI 并预处理
            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (config.IMG_SIZE, config.IMG_SIZE))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # AI 预测
            preds = emotion_classifier.predict(roi)[0]
            current_label = EMOTIONS[preds.argmax()]
            
            # 将预测结果加入缓存，用于后续计算 0.5s 平均值
            collected_preds.append(preds)
        else:
            current_label = ""

    # 3. 定时记录逻辑：每 0.5 秒计算并保存一次
    current_time = time.time()
    if current_time - last_log_time >= 0.5:
        if len(collected_preds) > 0:
            # 计算平均概率
            avg_preds = np.mean(collected_preds, axis=0)
            logger.log(avg_preds)
            # 重置缓存和计时器
            collected_preds = []
        last_log_time = current_time

    # 4. 绘图与展示（保证每一帧都刷新，视觉丝滑）
    if current_label != "":
        # 绘制左侧概率条
        for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
            text = "{}: {:.2f}%".format(emotion, prob * 100)
            w = int(prob * 300)
            cv2.rectangle(canvas, (7, (i * 35) + 5), (w, (i * 35) + 35), (0, 0, 255), -1)
            cv2.putText(canvas, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        # 绘制人脸框和标签
        cv2.putText(frameClone, current_label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH), (0, 0, 255), 2)

    cv2.imshow('your_face', frameClone)
    cv2.imshow("Probabilities", canvas)

    # 按 'q' 退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 清理资源
camera.release()
cv2.destroyAllWindows()
