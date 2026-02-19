# 实时情感识别ROS节点 (Emotion Recognition ROS Node)

本项目是基于卷积神经网络（CNN）的实时表情识别系统的ROS封装版本。它将摄像头捕捉的人脸图像实时识别为7类情感概率，并通过ROS话题发布，供其他节点（如表情生成系统）使用。

🌟 核心改进
ROS话题发布：识别结果通过 /emotion_label 和 /emotion_probs 话题实时发布，实现模块解耦。

路径独立性：使用 rospkg 自动定位资源文件，可在任意目录下通过 rosrun 启动，无需手动配置路径。

跳帧优化：引入 SKIP_FRAMES 机制，降低推理频率，保证实时预览流畅。

无缝集成：专为与 draw_pkg 表情生成节点配合设计，构成完整的情感交互流水线。

📂 文件夹结构说明
test_pkg/
├── config.py                # 核心配置文件（模型路径、检测参数等）
├── test.py                  # 主程序（ROS节点）
├── models/                  # 预训练情感识别模型权重文件 (_mini_XCEPTION.102-0.66.hdf5)
├── haarcascade_files/       # Haar Cascade 人脸检测特征文件
└── scripts/                 # （所有上述文件均位于此目录，但为清晰列出）
    ├── config.py
    ├── test.py
    ├── models/
    └── haarcascade_files/
🚀 快速开始
1. 环境准备
建议在 Ubuntu 20.04 + ROS Noetic 环境下运行。首先安装 ROS，然后安装 Python 依赖：

Bash
# 安装系统依赖
sudo apt install ros-noetic-desktop-full  # 已安装可跳过

# 安装 Python 包
pip install keras==2.3.1 tensorflow==1.15.0 opencv-python imutils numpy rospkg

2. 启动识别
确保 roscore 已运行：

Bash
roscore

另开终端，运行节点：

Bash
rosrun test_pkg test.py

程序会自动打开摄像头并显示两个窗口（人脸画面和概率条）。按 q 键退出。

3. 查看发布的话题
在另一终端查看发布的数据：

Bash
rostopic echo /emotion_label
rostopic echo /emotion_probs

📊 数据输出格式
/emotion_label (std_msgs/String)：当前主导情感标签，如 "happy"。

/emotion_probs (std_msgs/Float32MultiArray)：7个情感概率的列表，顺序为 ["angry","disgust","scared","happy","sad","surprised","neutral"]。

示例输出：
data: [0.05, 0.01, 0.02, 0.85, 0.03, 0.02, 0.02]

🛠 开发备注
模型来源：情感识别模型基于 _mini_XCEPTION 架构，在 FER-2013 数据集上训练。

人脸检测器：使用 OpenCV 的 Haar Cascade 分类器，可在 config.py 中调整 scaleFactor、minNeighbors 等参数以优化检测效果。

兼容性：代码已适配 NumPy 1.24+，移除了对 np.bool 的依赖。