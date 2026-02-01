#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import os


# 简单的全连接扩散模型 - 针对低维向量任务
class SimpleVectorDiffusion(nn.Module):
    def __init__(self, input_dim=5, condition_dim=2, hidden_dim=128, time_dim=64):
        super(SimpleVectorDiffusion, self).__init__()

        # 时间步嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # 条件处理
        self.condition_embed = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
        )

        # 主网络 - 简单的多层感知机
        self.network = nn.Sequential(
            nn.Linear(input_dim + 2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, input_dim)
        )

        # 时间嵌入维度
        self.time_dim = time_dim

    def _get_time_embed(self, t):
        """正弦位置编码"""
        half_dim = self.time_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)

        if self.time_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)

        return emb

    def forward(self, x, condition, t):
        # 时间步嵌入
        t_embed = self._get_time_embed(t)
        t_embed = self.time_embed(t_embed)

        # 条件嵌入
        condition_embed = self.condition_embed(condition)

        # 拼接所有输入
        combined = torch.cat([x, t_embed, condition_embed], dim=1)

        # 通过网络
        return self.network(combined)


# 简化的扩散过程
class VectorDiffusion:
    def __init__(self, time_steps=200, beta_start=1e-4, beta_end=0.02, device='cpu'):
        self.time_steps = time_steps
        self.device = device

        # 定义beta调度
        self.betas = torch.linspace(beta_start, beta_end, time_steps).to(device)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        """向前扩散过程：添加噪声"""
        sqrt_alpha_bar = torch.sqrt(self.alpha_bars[t]).view(-1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars[t]).view(-1, 1)

        noise = torch.randn_like(x0)
        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

        return xt, noise

    def sample_time_steps(self, batch_size):
        """随机采样时间步"""
        return torch.randint(0, self.time_steps, (batch_size,))


def normalize_input(input_vector,):
    """
    使用extra文件中的参数对输入向量进行归一化

    参数:
    input_vector: 二维向量，可以是list, tuple或np.array
    extra_file_path: extra.npy文件的路径

    返回:
    归一化后的二维向量
    """

    extra_file_path = "base/extra.npy"

    # 确保输入是numpy数组
    input_vec = np.array(input_vector, dtype=np.float64)

    # 检查输入维度
    if input_vec.shape != (2,):
        raise ValueError(f"输入向量维度应为(2,)，但得到{input_vec.shape}")

    # 加载归一化参数
    extra_data = np.load(extra_file_path)

    # 提取输入的最小值和最大值
    # 第一行: 输入参数，前2个是最小值，接着2个是最大值
    input_min = extra_data[0, :2]
    input_max = extra_data[0, 2:4]

    # 计算范围并避免除零
    input_range = input_max - input_min
    input_range[input_range == 0] = 1.0

    # 归一化到[-1, 1]范围
    # 公式: 2 * (x - min) / (max - min) - 1
    normalized_input = 2 * (input_vec - input_min) / input_range - 1

    return normalized_input


def denormalize_output(normalized_output,):
    """
    使用extra文件中的参数对归一化的输出向量进行反归一化

    参数:
    normalized_output: 5维归一化向量，可以是list, tuple或np.array
    extra_file_path: extra.npy文件的路径

    返回:
    反归一化后的5维向量
    """

    extra_file_path = "base/extra.npy"

    # 确保输入是numpy数组
    norm_output = np.array(normalized_output, dtype=np.float64)

    # 检查输入维度
    if norm_output.shape != (5,):
        raise ValueError(f"输入向量维度应为(5,)，但得到{norm_output.shape}")

    # 加载归一化参数
    extra_data = np.load(extra_file_path)

    # 提取输出的最小值和最大值
    # 第二行: 输出参数，前5个是最小值，接着5个是最大值
    output_min = extra_data[1, :5]
    output_max = extra_data[1, 5:]

    # 计算范围并避免除零
    output_range = output_max - output_min
    output_range[output_range == 0] = 1.0

    # 反归一化公式: (y_norm + 1) / 2 * (max - min) + min
    denormalized_output = (norm_output + 1) / 2 * output_range + output_min

    return denormalized_output


def generate_from_model(input_condition_vector, model_path="train/train_all.pth"):
    condition_vector = normalize_input(input_condition_vector)

    """使用训练好的模型生成5维向量"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件未找到: {model_path}")

    # 加载检查点
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # 创建模型和扩散过程
    model = SimpleVectorDiffusion(input_dim=5, condition_dim=2).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # 恢复扩散参数
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

    # 准备条件向量 - 修复的部分
    # 将各种类型的输入统一转换为PyTorch张量
    if isinstance(condition_vector, list):
        # 如果是列表，转换为张量
        condition_vector = torch.tensor(condition_vector, dtype=torch.float32)
    elif isinstance(condition_vector, np.ndarray):
        # 如果是numpy数组，转换为张量
        condition_vector = torch.from_numpy(condition_vector).float()
    elif isinstance(condition_vector, torch.Tensor):
        # 如果是张量，确保类型正确
        condition_vector = condition_vector.float()
    else:
        raise TypeError(f"不支持的输入类型: {type(condition_vector)}")

    # 移动到设备
    condition_vector = condition_vector.to(device)

    # 确保有正确的维度
    if condition_vector.dim() == 1:
        condition_vector = condition_vector.unsqueeze(0)

    # 从纯噪声开始
    batch_size = condition_vector.shape[0]
    generated = torch.randn(batch_size, 5, device=device)

    # 反向扩散过程
    with torch.no_grad():
        for t in reversed(range(diffusion.time_steps)):
            t_tensor = torch.tensor([t] * batch_size, device=device)

            # 预测噪声
            predicted_noise = model(generated, condition_vector, t_tensor)

            # 去噪步骤
            alpha_t = diffusion.alphas[t]
            alpha_bar_t = diffusion.alpha_bars[t]
            beta_t = diffusion.betas[t]

            if t > 0:
                noise = torch.randn_like(generated)
            else:
                noise = torch.zeros_like(generated)

            # 更新生成结果
            generated = (1 / torch.sqrt(alpha_t)) * (
                    generated - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            ) + torch.sqrt(beta_t) * noise

    return denormalize_output(generated.cpu().numpy()[0])


def interactive_generation():
    """交互式生成函数"""
    print("=" * 60)
    print("扩散模型交互式生成器")
    print("=" * 60)
    print("请输入两个浮点数作为条件向量")
    print("（输入 'q' 退出程序，输入 'r' 重新输入）")
    print("-" * 60)

    # 检查模型文件是否存在
    model_path = "train/train_all.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        print("请先训练模型或确保模型文件存在")
        return

    print(f"加载模型: {model_path}")

    while True:
        try:
            # 获取用户输入
            input_str = input("\n请输入两个浮点数（用空格分隔）: ")

            # 检查退出命令
            if input_str.lower() == 'q':
                print("程序退出")
                break

            # 检查重新输入命令
            if input_str.lower() == 'r':
                print("重新输入...")
                continue

            # 解析输入
            values = input_str.strip().split()

            if len(values) != 2:
                print("错误: 请输入恰好两个数字")
                continue

            # 转换为浮点数
            condition_vector = [float(val) for val in values]

            print(f"\n输入的条件向量: {condition_vector}")

            # 生成输出
            print("正在生成...")
            output_values = generate_from_model(condition_vector, model_path)
            print(f"\n生成的5维向量:")

            # 格式化每个数字，保留6位小数
            for i, val in enumerate(output_values):
                print(f"  维度{i + 1}: {val:.6f}")

            print(f"\n完整向量: [{', '.join([f'{val:.6f}' for val in output_values])}]")

            # 是否继续
            continue_input = input("\n是否继续生成？(y/n): ")
            if continue_input.lower() != 'y':
                print("程序退出")
                break

        except ValueError:
            print("错误: 请输入有效的浮点数")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")


def test_with_training_data():
    """使用训练数据进行测试"""
    print("\n" + "=" * 60)
    print("使用训练数据进行测试")
    print("=" * 60)

    try:
        # 加载训练数据
        data_path = "base/train.npy"
        if not os.path.exists(data_path):
            print(f"错误: 训练数据文件未找到: {data_path}")
            return

        data = np.load(data_path)
        print(f"训练数据形状: {data.shape}")

        # 使用前几个样本进行测试
        num_test_samples = min(3, len(data))

        for i in range(num_test_samples):
            condition = data[i, :2]
            target = denormalize_output(data[i, 2:7])

            print(f"\n测试样本 {i + 1}:")
            print(f"  输入条件: {condition}")
            print(f"  真实输出: {target}")

            # 生成
            generated = generate_from_model(condition)
            print(f"  生成输出: {generated}")

            # 计算误差
            error = np.abs(target - generated)
            print(f"  绝对误差: {error}")
            print(f"  平均误差: {np.mean(error):.6f}")

    except Exception as e:
        print(f"测试过程中出现错误: {e}")


def usage():
    print("请输入3个浮点数，分别表示\"效度(V)\"、\"唤醒度(A)\"和\"优势度(D)\"")

    # 检查模型文件是否存在
    model_path = "train/train_all.pth"
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到: {model_path}")
        print("请先训练模型或确保模型文件存在")
        return

    while True:
        try:
            # 获取用户输入
            input_str = input("\n请输入3个浮点数（用空格分隔）: ")

            # 解析输入
            values = input_str.strip().split()

            if len(values) != 3:
                print("错误: 请输入恰好3个数字")
                continue

            # 转换为浮点数
            num1 = float(values[0])
            num2 = float(values[1])
            k = float(values[2])
            condition_vector = [num1,num2]

            # 生成输出
            output_values = generate_from_model(condition_vector, model_path)

            return k, output_values

        except ValueError:
            print("错误: 请输入有效的浮点数")
        except Exception as e:
            print(f"生成过程中出现错误: {e}")


def main():
    """主函数"""
    print("扩散模型5维向量生成器")
    print("=" * 60)

    #print(usage())

    # 显示菜单
    while True:
        print("\n请选择操作:")
        print("1. 手动输入两个数生成五个数")
        print("2. 使用训练数据进行测试")
        print("3. 退出程序")

        choice = input("请输入选择 (1-3): ")

        if choice == '1':
            interactive_generation()
        elif choice == '2':
            test_with_training_data()
        elif choice == '3':
            print("程序退出")
            break
        else:
            print("无效选择，请重新输入")


if __name__ == "__main__":
    main()