实时情绪识别优化版 (Emotion Recognition Optimized)
本项目是基于卷积神经网络（CNN）的实时表情识别系统的优化版本。主要针对原版代码在现代环境下的兼容性、运行效率以及数据记录能力进行了重构。

🌟 核心改进 (Optimization Highlights)
性能飞跃：跳帧检测机制

引入了 SKIP_FRAMES 逻辑，允许每隔 N 帧进行一次深度学习推理，极大降低了 CPU/GPU 负载，解决了实时预览画面的卡顿问题。

自动化统计：0.5s 数据日志

新增 data_logger.py 模块，系统每 0.5 秒自动计算并记录这段时间内检测到的平均情绪概率，支持科研级的数据分析。

工程化重构：配置解耦

将原本硬编码在代码中的模型路径、检测参数、图像尺寸等抽离至 config.py，实现“一处修改，全局生效”。

环境兼容性优化

针对 NumPy 1.24+ 移除 np.bool 导致的崩溃问题，通过版本锁定和代码重构完成了环境适配。

📂 文件夹结构说明
Plaintext
xxw_emotion/
├── config.py                # 项目核心配置文件（参数微调）
├── data_logger.py           # 数据持久化模块（负责生成 .txt 日志）
├── real_time_video.py       # 优化后的主程序（带跳帧和记录逻辑）
├── requirements.txt         # 针对本模块优化后的依赖清单
├── models/                  # 预训练模型权重文件 (.hdf5)
├── haarcascade_files/       # Haar Cascade 人脸检测特征文件
└── data_logs/               # [自动生成] 存放历史运行的情绪记录数据
🚀 快速开始
1. 环境准备
建议在 Python 3.8+ 虚拟环境下运行：

Bash
# 升级基础构建工具
pip install --upgrade pip setuptools wheel

# 安装优化版依赖
pip install -r requirements.txt

2. 数据集准备 (可选)
本项目模型基于 FER-2013 数据集训练。如果需要重新训练或查看原始数据，请通过以下 Kaggle 链接下载：

下载地址: FER-2013 Dataset (Kaggle)

放置路径: 下载解压后，请将数据放入 xxw_emotion/fer2013/ 目录下。

3. 启动识别
直接运行主程序，系统会自动调用摄像头并开始记录数据：

Bash
python3 real_time_video.py
退出程序：在视频窗口按下 q 键。

查看数据：运行结束后，可在 data_logs/ 目录下找到以时间戳命名的 .txt 文件。

📊 数据日志格式说明
生成的日志文件采用标准 CSV 格式，方便导入 Excel 或 Python 进行二次分析： | Time | Angry | Happy | Sad | Surprised | ... | | :--- | :--- | :--- | :--- | :--- | :--- | | 14:30:01 | 0.05% | 85.20% | 2.10% | 1.50% | ... |

🛠 开发备注
数据集：本项目模型基于 FER-2013 数据集训练。

检测器：默认使用 OpenCV 的 Haar Cascade 进行人脸定位，可在 config.py 中调整 SCALE_FACTOR 以优化远距离人脸检测。