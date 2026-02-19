情感交互系统 (Emotion Interaction System) - ROS包集合
📋 概述
情感交互系统是一套基于ROS（机器人操作系统）的实时情感识别与表情生成框架。系统由三个独立的ROS包组成：

draw_a_face_old：旧版几何表情生成包，通过手动输入VAD三维向量生成静态表情图像。

draw_pkg：新版实时表情生成ROS节点，订阅情感话题并动态绘制表情。

test_pkg：实时情感识别ROS节点，通过摄像头捕捉人脸并发布7类情感概率。

三个包可独立使用，也可串联构成完整的“感知-表达”闭环：test_pkg识别用户情感，通过ROS话题传递给draw_pkg生成对应的机器表情，实现实时情感交互。

目录
一、系统概述与包结构大类：
第一部分————系统组成：三个包分别是做什么的？
第二部分————环境与依赖：需要什么前提？

二、各包详解与使用大类：
第三部分————draw_a_face_old包（旧版表情生成）
第四部分————draw_pkg包（新版实时表情生成ROS节点）
第五部分————test_pkg包（情感识别ROS节点）

三、系统集成与原理大类：
第六部分————如何让整个系统运行起来？主程序使用方式
第七部分————技术原理与数据流
第八部分————已知限制与未来改进

第一部分————系统组成：三个包分别是做什么的？
🗂️ 一、draw_a_face_old —— 离线表情生成器

功能：通过手动输入VAD三维向量（Valence, Arousal, Dominance），调用扩散模型生成几何参数，绘制1024×1024灰度表情图像并保存。

适用场景：算法验证、静态表情生成、离线测试。

交互方式：命令行交互输入。

🗂️ 二、draw_pkg —— 实时表情生成ROS节点

功能：订阅ROS话题 /emotion_label 和 /emotion_probs，接收情感数据，实时生成并显示表情图像。

适用场景：与test_pkg配合，构建实时情感反馈系统。

交互方式：ROS话题通信，自动响应。

🗂️ 三、test_pkg —— 实时情感识别ROS节点

功能：通过摄像头实时检测人脸，识别7类情感（生气、厌恶、恐惧、开心、悲伤、惊讶、中性），将结果通过ROS话题发布。

适用场景：提供用户情感输入，驱动表情生成。

交互方式：ROS话题发布，同时显示实时视频窗口。

第二部分————环境与依赖：需要什么前提？
🔧 一、系统要求

操作系统：Ubuntu 20.04（ROS Noetic）

ROS：已安装并配置ROS Noetic环境

Python：3.8+

📦 二、公共Python依赖

bash
pip install numpy pillow torch rospkg opencv-python imutils
注意：test_pkg 需要TensorFlow/Keras环境，推荐使用 tensorflow==1.15.0 和 keras==2.3.1，具体见各包内requirements.txt。

📂 三、包路径说明
所有包应放置于ROS工作空间的src目录下，例如：

text
~/catkin_ws/src/
├── draw_a_face_old/
├── draw_pkg/
└── test_pkg/
每个包的内部结构详见后续部分。

第三部分————draw_a_face_old包（旧版表情生成）
🗂️ 一、目录结构

text
draw_a_face_old/
├── CMakeLists.txt
├── package.xml
├── scripts/
│   ├── translation.py          # 扩散模型参数生成
│   ├── draw.py                  # 主绘图程序
│   ├── base/                     # 基础数据
│   │   ├── base.npy
│   │   ├── ellipses.npy
│   │   └── extra.npy
│   ├── train/                    # 预训练模型
│   │   └── train_all.pth
│   └── results/                  # 输出图像
⚙️ 二、使用方式

bash
# 在ROS环境下运行（不依赖roscore）
rosrun draw_a_face_old draw.py
# 或直接运行Python脚本
cd ~/catkin_ws/src/draw_a_face_old/scripts
python3 draw.py
程序提示输入三个浮点数（V A D），生成表情并保存至results/。

🔍 三、输入输出说明

输入：三个浮点数，分别代表Valence、Arousal、Dominance（推荐范围[-1,1]）。

输出：1024×1024灰度PNG图像，路径在控制台打印。

第四部分————draw_pkg包（新版实时表情生成ROS节点）
🗂️ 一、目录结构

text
draw_pkg/
├── CMakeLists.txt
├── package.xml
├── scripts/
│   ├── draw.py                  # ROS节点主程序
│   ├── translation.py            # 改进版扩散模型（含new_usage）
│   ├── base/                     # 基础数据
│   │   ├── base.npy
│   │   ├── ellipses.npy
│   │   └── extra.npy
│   ├── train/                    # 预训练模型
│   │   └── train_all.pth
│   └── results/                  # 手动保存图像目录
📡 二、ROS通信接口

订阅话题：

/emotion_label (std_msgs/String)：主导情感标签，如"happy"。

/emotion_probs (std_msgs/Float32MultiArray)：7类情感概率数组。

🚀 三、启动节点

bash
# 确保roscore已运行
roscore
# 新终端启动节点
rosrun draw_pkg draw.py
节点启动后显示“Machine Expression”窗口，等待数据更新。

💾 四、按键控制

q：退出程序

s：保存当前表情图像到results/目录

第五部分————test_pkg包（情感识别ROS节点）
🗂️ 一、目录结构

text
test_pkg/
├── CMakeLists.txt
├── package.xml
├── config.py                     # 配置文件
├── scripts/
│   ├── test.py                   # 主程序（ROS节点）
│   ├── models/                    # 情感识别模型
│   │   └── _mini_XCEPTION.102-0.66.hdf5
│   └── haarcascade_files/         # 人脸检测器
│       └── haarcascade_frontalface_default.xml
📡 二、ROS通信接口

发布话题：

/emotion_label (std_msgs/String)：当前主导情感标签。

/emotion_probs (std_msgs/Float32MultiArray)：7类情感概率。

🚀 三、启动节点

bash
# 确保roscore已运行
rosrun test_pkg test.py
程序打开摄像头，显示“Your Face”和“Emotion Probabilities”窗口，实时显示识别结果。

📊 四、数据输出示例

bash
rostopic echo /emotion_label
data: "happy"
---
rostopic echo /emotion_probs
data: [0.05, 0.01, 0.02, 0.85, 0.03, 0.02, 0.02]
---
第六部分————如何让整个系统运行起来？主程序使用方式
🚀 一、启动完整交互系统

启动ROS核心：roscore

启动情感识别节点（新终端）：rosrun test_pkg test.py

启动表情生成节点（新终端）：rosrun draw_pkg draw.py

此时，摄像头画面、情感概率条、机器表情窗口将同时显示。当识别到人脸并更新情感时，表情窗口会自动变化。

📡 二、数据流示意图

text
摄像头 → test_pkg → 情感概率+标签 → ROS话题 → draw_pkg → 表情图像
test_pkg 以约5Hz频率（可配置）发布结果。

draw_pkg 收到新数据后立即更新表情（生成耗时约1-3秒，期间显示上一帧）。

🔄 三、独立使用模式

若只需生成特定表情，可运行draw_a_face_old手动输入VAD。

若只需情感识别，可单独运行test_pkg（不启动draw_pkg）。

第七部分————技术原理与数据流
🧠 一、情感识别 (test_pkg)

人脸检测：OpenCV Haar Cascade分类器。

情感分类：预训练的CNN模型（_mini_XCEPTION），输出7类概率。

跳帧优化：每隔N帧执行一次推理，降低负载。

🎭 二、情感→VAD转换 (translation.py in draw_pkg)

从7类概率计算主导情感，并通过心理学映射得到基准VAD。

应用引导策略，调整VAD以促进用户开心。

最终VAD中，V和A作为扩散模型条件，D直接控制眉毛斜率。

🔄 三、扩散模型生成几何参数

基于DDPM的轻量级网络，将条件向量[V, A]映射为5维参数（抛物线形状）。

模型预训练于特定数据集，确保参数合理性。

📐 四、几何绘图 (draw.py)

固定元素：圆环（脸型）、椭圆（眼睛）从.npy文件加载。

动态元素：

眉毛：线段带，斜率 k = D * golden_ratio。

嘴巴：抛物线区域，由5个扩散模型输出参数决定。

第八部分————已知限制与未来改进
⚠️ 一、当前限制

包	限制	影响
test_pkg	依赖TensorFlow 1.x，环境配置较复杂	中等
draw_pkg	表情生成速度慢（1-3秒），无法实时视频流	高
draw_pkg	模型文件较大，且需与脚本同路径（已通过临时切换目录解决）	低
整体	未实现参数动态平滑，表情切换可能生硬	中等
🔮 二、未来改进方向

性能优化：将扩散模型转为ONNX或TensorRT，加速推理。

实时平滑：增加表情插值，使过渡自然。

参数学习：将D也纳入扩散模型，实现VAD端到端控制。

个性化：支持自定义脸型、眼睛模板。

ROS2迁移：适配ROS2以利用现代工具链。

📄 许可证与引用

版本：实验性代码 v0.3

作者：顾昊宇 / 未来技术小分队

最后更新：2026/2/15

引用：如使用本系统，请引用相关论文（待发表）。