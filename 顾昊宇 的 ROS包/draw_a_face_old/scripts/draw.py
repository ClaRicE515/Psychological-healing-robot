#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
translation.py - 扩散模型参数生成（路径独立版）
使用 rospkg 定位模型文件，支持在任何目录下被 draw.py 调用
"""

import torch
import torch.nn as nn
import numpy as np
import os
import rospkg

# ==================== 获取包路径 ====================
rospack = rospkg.RosPack()
package_path = rospack.get_path('draw_a_face_old')
scripts_path = os.path.join(package_path, 'scripts')
base_path = os.path.join(scripts_path, 'base')
train_path = os.path.join(scripts_path, 'train')

EXTRA_FILE = os.path.join(base_path, 'extra.npy')
MODEL_FILE = os.path.join(train_path, 'train_all.pth')

# ==================== 扩散模型定义 ====================
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

# ==================== 归一化函数 ====================
def normalize_input(input_vector):
    """使用 extra.npy 对输入向量归一化"""
    input_vec = np.array(input_vector, dtype=np.float64)
    if input_vec.shape != (2,):
        raise ValueError(f"输入向量维度应为(2,)，但得到{input_vec.shape}")
    extra_data = np.load(EXTRA_FILE)
    input_min = extra_data[0, :2]
    input_max = extra_data[0, 2:4]
    input_range = input_max - input_min
    input_range[input_range == 0] = 1.0
    normalized_input = 2 * (input_vec - input_min) / input_range - 1
    return normalized_input

def denormalize_output(normalized_output):
    """使用 extra.npy 对输出向量反归一化"""
    norm_output = np.array(normalized_output, dtype=np.float64)
    if norm_output.shape != (5,):
        raise ValueError(f"输入向量维度应为(5,)，但得到{norm_output.shape}")
    extra_data = np.load(EXTRA_FILE)
    output_min = extra_data[1, :5]
    output_max = extra_data[1, 5:]
    output_range = output_max - output_min
    output_range[output_range == 0] = 1.0
    denormalized_output = (norm_output + 1) / 2 * output_range + output_min
    return denormalized_output

# ==================== 生成函数 ====================
def generate_from_model(input_condition_vector):
    """使用训练好的模型生成5维向量（路径独立版）"""
    condition_vector = normalize_input(input_condition_vector)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"模型文件未找到: {MODEL_FILE}")

    checkpoint = torch.load(MODEL_FILE, map_location=device, weights_only=True)
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

# ==================== 交互式函数 ====================
def usage():
    """
    交互式输入 V, A, D，返回 (k, 5维向量)
    若模型文件缺失则抛出 FileNotFoundError
    """
    print("请输入3个浮点数，分别表示\"效度(V)\"、\"唤醒度(A)\"和\"优势度(D)\"")

    # 提前检查模型文件
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"错误: 模型文件未找到: {MODEL_FILE}\n请先训练模型或确保模型文件存在")

    while True:
        try:
            input_str = input("\n请输入3个浮点数（用空格分隔）: ")
            values = input_str.strip().split()
            if len(values) != 3:
                print("错误: 请输入恰好3个数字")
                continue

            num1 = float(values[0])
            num2 = float(values[1])
            k = float(values[2])
            condition_vector = [num1, num2]

            output_values = generate_from_model(condition_vector)
            return k, output_values

        except ValueError:
            print("错误: 请输入有效的浮点数")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")

# ==================== 测试入口（可选）====================
if __name__ == "__main__":
    # 直接运行此文件可用于测试
    k, result = usage()
    print(f"k = {k}")
    print(f"5维向量 = {result}")