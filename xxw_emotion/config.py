# config.py

# 1. 性能优化参数
SKIP_FRAMES = 3      # 每隔几帧处理一次 AI 识别
IMG_SIZE = 64        # 必须与模型训练时的尺寸一致（你之前源码是64）

# 2. 人脸检测参数 (Haar Cascade)
SCALE_FACTOR = 1.1   # 扫描时的缩放步长
MIN_NEIGHBORS = 5    # 确定人脸需要的相邻矩形数（检测不到时调小，误报多时调大）
MIN_SIZE = (30, 30)  # 最小检测人脸尺寸

# 3. 文件路径配置
MODEL_PATH = 'models/_mini_XCEPTION.102-0.66.hdf5'
CASCADE_PATH = 'haarcascade_files/haarcascade_frontalface_default.xml'