#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translation.py - 情感转换与扩散生成融合模块

功能：
1. 将7类离散情感概率转换为VAD向量（使用transform模块）
2. 取VAD的前两维(V,A)作为条件，通过扩散模型生成5维表情参数
3. 返回第三维D和生成的5维向量

此版本为ROS节点，订阅情感话题并实时处理。
"""

import os
import numpy as np
import torch
import torch.nn as nn
import rospy
from std_msgs.msg import String, Float32MultiArray

# ==================== 1. 复制 transform.py 内容（情感→VAD转换）====================
class EmotionTransformer:
    """
    情感转换器：将离散情感分类转换为连续VAD向量
    """
    def __init__(self):
        """初始化情感到VAD的映射规则"""
        self.emotion_vad_mapping = {
            'angry':      [-0.85, 0.80, 0.70],
            'disgust':    [-0.90, 0.60, 0.30],
            'scared':     [-0.75, 0.85, -0.60],
            'happy':      [0.90, 0.65, 0.40],
            'sad':        [-0.85, -0.40, -0.65],
            'surprised':  [0.30, 0.90, 0.10],
            'neutral':    [0.00, 0.00, 0.00]
        }
        self.strategy_params = {
            'positive_boost': 0.3,
            'arousal_boost': 0.2,
            'dominance_guide': 0.25,
            'dominance_reduce': 0.15
        }
        self.emotion_order = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

    def map_emotion_to_vad(self, emotion_label: str):
        if emotion_label not in self.emotion_vad_mapping:
            raise ValueError(f"未知的情感标签: {emotion_label}")
        return self.emotion_vad_mapping[emotion_label].copy()

    def weighted_average_vad(self, emotion_probs: np.ndarray):
        if len(emotion_probs) != len(self.emotion_order):
            raise ValueError(f"概率数组长度应为{len(self.emotion_order)}")
        vad_sum = [0.0, 0.0, 0.0]
        for i, emotion in enumerate(self.emotion_order):
            weight = emotion_probs[i]
            base_vad = self.emotion_vad_mapping[emotion]
            vad_sum[0] += weight * base_vad[0]
            vad_sum[1] += weight * base_vad[1]
            vad_sum[2] += weight * base_vad[2]
        return vad_sum

    def apply_guidance_strategy(self, vad_vector: list, user_emotion_label: str):
        V, A, D = vad_vector
        if user_emotion_label in ['angry', 'disgust', 'scared', 'sad']:
            V = min(V + self.strategy_params['positive_boost'], 1.0)
            if user_emotion_label == 'angry':
                D = max(D - 0.1, -1.0)
                A = min(A + 0.1, 1.0)
            elif user_emotion_label == 'sad':
                D = min(D + self.strategy_params['dominance_guide'], 1.0)
        elif user_emotion_label == 'happy':
            V = min(V + 0.15, 1.0)
            D = max(D - self.strategy_params['dominance_reduce'], -1.0)
        elif user_emotion_label == 'neutral':
            if V < 0.3:
                V = min(V + 0.25, 1.0)
            if A < 0.2:
                A = min(A + self.strategy_params['arousal_boost'], 1.0)
        elif user_emotion_label == 'surprised':
            A = max(A - 0.1, -1.0)

        V = max(min(V, 1.0), -1.0)
        A = max(min(A, 1.0), -1.0)
        D = max(min(D, 1.0), -1.0)

        if V > 0.7 and A < -0.5:
            A = -0.2
        if V < -0.7 and D < -0.7:
            D = max(D + 0.2, -0.5)

        return [round(V, 3), round(A, 3), round(D, 3)]

    def calculate_vad(self, emotion_probs: np.ndarray, emotion_label: str = None):
        if emotion_label is None:
            emotion_label = self.emotion_order[np.argmax(emotion_probs)]
        weighted_vad = self.weighted_average_vad(emotion_probs)
        base_vad = self.map_emotion_to_vad(emotion_label)
        guided_vad = self.apply_guidance_strategy(weighted_vad, emotion_label)
        conversion_info = {
            'primary_emotion': emotion_label,
            'primary_probability': float(emotion_probs[np.argmax(emotion_probs)]),
            'base_vad': [round(v, 3) for v in base_vad],
            'weighted_vad': [round(v, 3) for v in weighted_vad],
            'guided_vad': guided_vad,
            'strategy_applied': self.get_strategy_description(emotion_label)
        }
        return guided_vad, conversion_info

    def get_strategy_description(self, emotion_label: str):
        strategy_descriptions = {
            'angry': "积极安抚策略：提高愉悦度，避免冲突",
            'disgust': "温和引导策略：提高愉悦度，温和回应",
            'scared': "安抚稳定策略：提高愉悦度，提供安全感",
            'happy': "共享强化策略：匹配并稍高愉悦度，减少控制",
            'sad': "温暖鼓励策略：显著提高愉悦度，适度增加引导",
            'surprised': "稳定回应策略：保持稳定，提供支持",
            'neutral': "激发兴趣策略：适度提高愉悦度和唤醒度"
        }
        return strategy_descriptions.get(emotion_label, "默认策略：适度引导")

# 全局转换器实例和简化接口
transformer = EmotionTransformer()

def emotion_to_vad(emotion_probs: np.ndarray, emotion_label: str = None) -> list:
    vad_vector, _ = transformer.calculate_vad(emotion_probs, emotion_label)
    return vad_vector

def emotion_to_vad_detailed(emotion_probs: np.ndarray, emotion_label: str = None):
    return transformer.calculate_vad(emotion_probs, emotion_label)

# ==================== 2. 保留原 translation.py 的扩散模型部分 ====================
class SimpleVectorDiffusion(nn.Module):
    def __init__(self, input_dim=5, condition_dim=2, hidden_dim=128, time_dim=64):
        super(SimpleVectorDiffusion, self).__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )
        self.network = nn.Sequential(
            nn.Linear(input_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )
        self.time_dim = time_dim

    def _get_time_embed(self, t):
        half_dim = self.time_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb

    def forward(self, x, condition, t):
        t_embed = self._get_time_embed(t)
        t_embed = self.time_embed(t_embed)
        condition_embed = self.condition_embed(condition)
        combined = torch.cat([x, t_embed, condition_embed], dim=1)
        return self.network(combined)

class VectorDiffusion:
    def __init__(self, time_steps=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.time_steps = time_steps
        self.device = device
        self.betas = torch.linspace(beta_start, beta_end, time_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)
        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def sample_time_steps(self, batch_size):
        return torch.randint(0, self.time_steps, (batch_size,))

def normalize_input(input_vector):
    extra_file_path = "base/extra.npy"
    input_vec = np.array(input_vector, dtype=np.float64)
    if input_vec.shape != (2,):
        raise ValueError(f"输入向量维度应为(2,)，但得到{input_vec.shape}")
    extra_data = np.load(extra_file_path)
    input_min = extra_data[0, :2]
    input_max = extra_data[0, 2:4]
    input_range = input_max - input_min
    input_range[input_range == 0] = 1.0
    normalized_input = 2 * (input_vec - input_min) / input_range - 1
    return normalized_input

def denormalize_output(normalized_output):
    extra_file_path = "base/extra.npy"
    norm_output = np.array(normalized_output, dtype=np.float64)
    if norm_output.shape != (5,):
        raise ValueError(f"输入向量维度应为(5,)，但得到{norm_output.shape}")
    extra_data = np.load(extra_file_path)
    output_min = extra_data[1, :5]
    output_max = extra_data[1, 5:]
    output_range = output_max - output_min
    output_range[output_range == 0] = 1.0
    denormalized_output = (norm_output + 1) / 2 * output_range + output_min
    return denormalized_output

def generate_from_model(input_condition_vector, model_path="train/train_all.pth"):
    condition_vector = normalize_input(input_condition_vector)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model = SimpleVectorDiffusion(input_dim=5, condition_dim=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    diffusion_params = checkpoint['diffusion_params']
    diffusion = VectorDiffusion(
        time_steps=diffusion_params['time_steps'],
        beta_start=diffusion_params['betas'][0].item(),
        beta_end=diffusion_params['betas'][-1].item(),
        device=device
    )
    diffusion.betas = diffusion_params['betas']
    diffusion.alphas = diffusion_params['alphas']
    diffusion.alpha_bars = diffusion_params['alpha_bars']

    # 准备条件向量
    if isinstance(condition_vector, list):
        condition_vector = torch.tensor(condition_vector, dtype=torch.float32)
    elif isinstance(condition_vector, np.ndarray):
        condition_vector = torch.from_numpy(condition_vector).float()
    elif isinstance(condition_vector, torch.Tensor):
        condition_vector = condition_vector.float()
    else:
        raise TypeError(f"不支持的输入类型: {type(condition_vector)}")

    condition_vector = condition_vector.to(device)
    if condition_vector.dim() == 1:
        condition_vector = condition_vector.unsqueeze(0)

    batch_size = condition_vector.shape[0]
    generated = torch.randn(batch_size, 5, device=device)

    with torch.no_grad():
        for t in reversed(range(diffusion.time_steps)):
            t_tensor = torch.tensor([t] * batch_size, device=device)
            predicted_noise = model(generated, condition_vector, t_tensor)
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]
            beta_t = diffusion.betas[t]
            if t > 0:
                noise = torch.randn_like(generated)
            else:
                noise = torch.zeros_like(generated)
            generated = (1 / torch.sqrt(alpha_t)) * (
                    generated - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    return denormalize_output(generated.cpu().numpy()[0])

# ==================== 3. 新的 new_usage 函数 ====================
def new_usage(emotion_probs, emotion_label=None):
    """
    输入：7类情感概率（list或np.array），可选情感标签
    输出：(D, 5维向量)，D为VAD的第三维，5维向量由扩散模型生成
    """
    # 确保概率为numpy数组
    if not isinstance(emotion_probs, np.ndarray):
        emotion_probs = np.array(emotion_probs, dtype=np.float64)
    # 调用情感转换，得到VAD（引导后）
    V, A, D = emotion_to_vad(emotion_probs, emotion_label)  # 返回列表 [V,A,D]
    # 前两维作为扩散条件
    condition = [V, A]
    # 生成5维向量
    output = generate_from_model(condition)
    return D, output

# ==================== 4. ROS 节点：订阅情感数据，调用 new_usage ====================
class EmotionListener:
    def __init__(self):
        self.label = None
        self.probs = None
        rospy.Subscriber('/emotion_label', String, self.label_cb)
        rospy.Subscriber('/emotion_probs', Float32MultiArray, self.probs_cb)
        rospy.loginfo("EmotionListener 已启动，等待数据...")

    def label_cb(self, msg):
        self.label = msg.data
        self.try_process()

    def probs_cb(self, msg):
        # Float32MultiArray 的 data 属性是列表
        self.probs = np.array(msg.data, dtype=np.float64)
        self.try_process()

    def try_process(self):
        if self.label is not None and self.probs is not None:
            try:
                D, output = new_usage(self.probs, self.label)
                rospy.loginfo(f"接收到情感: {self.label}")
                rospy.loginfo(f"生成的5维向量: {output}")
                rospy.loginfo(f"对应的D值: {D:.3f}")
                # 打印详细概率分布（可选）
                rospy.loginfo(f"概率分布: {self.probs}")
            except Exception as e:
                rospy.logerr(f"处理出错: {e}")
            finally:
                # 重置以便接收下一组数据
                self.label = None
                self.probs = None

def main():
    rospy.init_node('translation_node', anonymous=True)
    listener = EmotionListener()
    rospy.spin()

if __name__ == "__main__":
    main()