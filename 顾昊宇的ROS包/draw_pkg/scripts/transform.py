"""
transform.py - 情感格式转换模块

功能：将7类离散情感概率转换为VAD三维向量
输入：test.py输出的情感概率分布
输出：draw_a_face_old需要的VAD三维向量
"""

import numpy as np
from typing import Tuple, List, Dict


class EmotionTransformer:
    """
    情感转换器：将离散情感分类转换为连续VAD向量
    """
    
    def __init__(self):
        """初始化情感到VAD的映射规则"""
        # 基于心理学研究的基准映射（基于Russell和Mehrabian的VAD模型研究）
        self.emotion_vad_mapping = {
            # 情感: [Valence愉悦度, Arousal唤醒度, Dominance优势度]
            'angry':      [-0.85, 0.80, 0.70],      # 愤怒：不愉快、高唤醒、高控制
            'disgust':    [-0.90, 0.60, 0.30],      # 厌恶：极不愉快、中高唤醒、中等控制
            'scared':     [-0.75, 0.85, -0.60],     # 恐惧：不愉快、高唤醒、低控制
            'happy':      [0.90, 0.65, 0.40],       # 快乐：极愉快、中高唤醒、中等控制
            'sad':        [-0.85, -0.40, -0.65],    # 悲伤：不愉快、低唤醒、低控制
            'surprised':  [0.30, 0.90, 0.10],       # 惊讶：略愉快、高唤醒、低控制
            'neutral':    [0.00, 0.00, 0.00]        # 中性：中性、平静、中立
        }
        
        # 情感引导策略参数
        self.strategy_params = {
            # 策略：不同用户情感下，机器应如何调整VAD值来引导用户开心
            'positive_boost': 0.3,      # 对消极情感提高的愉悦度值
            'arousal_boost': 0.2,       # 对中性/低唤醒提高的唤醒度值
            'dominance_guide': 0.25,    # 对需要引导的情感提高的控制度值
            'dominance_reduce': 0.15    # 对积极情感降低的控制度值
        }
        
        # 情感类别顺序（必须与test.py中的EMOTIONS顺序一致）
        self.emotion_order = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
    
    def map_emotion_to_vad(self, emotion_label: str) -> List[float]:
        """
        将单一情感标签映射到基准VAD值
        
        参数:
            emotion_label: 情感标签字符串
        
        返回:
            对应情感的基准VAD向量
        """
        if emotion_label not in self.emotion_vad_mapping:
            raise ValueError(f"未知的情感标签: {emotion_label}")
        
        return self.emotion_vad_mapping[emotion_label].copy()
    
    def weighted_average_vad(self, emotion_probs: np.ndarray) -> List[float]:
        """
        基于概率分布的加权平均VAD计算
        
        参数:
            emotion_probs: 7类情感概率的numpy数组
        
        返回:
            加权平均后的VAD向量
        """
        if len(emotion_probs) != len(self.emotion_order):
            raise ValueError(f"概率数组长度应为{len(self.emotion_order)}，实际为{len(emotion_probs)}")
        
        # 初始化VAD累加器
        vad_sum = [0.0, 0.0, 0.0]
        
        # 对每种情感进行加权累加
        for i, emotion in enumerate(self.emotion_order):
            weight = emotion_probs[i]
            base_vad = self.emotion_vad_mapping[emotion]
            
            vad_sum[0] += weight * base_vad[0]  # Valence
            vad_sum[1] += weight * base_vad[1]  # Arousal
            vad_sum[2] += weight * base_vad[2]  # Dominance
        
        return vad_sum
    
    def apply_guidance_strategy(self, vad_vector: List[float], 
                                user_emotion_label: str) -> List[float]:
        """
        应用情感引导策略：调整VAD值以达到"让用户开心"的目标
        
        策略原理：
        1. 不是简单镜像用户情感，而是策略性引导
        2. 对消极情感：提高愉悦度以感染用户
        3. 对积极情感：匹配并稍高以强化积极氛围
        4. 对中性/低唤醒：适度提高唤醒度以激发兴趣
        5. 根据需要调整控制度：需要引导时提高，需要自主时降低
        
        参数:
            vad_vector: 加权平均后的VAD向量
            user_emotion_label: 用户主要情感标签
        
        返回:
            应用策略后的VAD向量
        """
        V, A, D = vad_vector
        
        # 策略1：基于用户情感类型的引导
        if user_emotion_label in ['angry', 'disgust', 'scared', 'sad']:
            # 用户消极 → 机器更积极（提高愉悦度）
            V = min(V + self.strategy_params['positive_boost'], 1.0)
            
            # 根据具体消极情感微调
            if user_emotion_label == 'angry':
                # 对愤怒用户：降低控制度以避免冲突，提高唤醒度以匹配能量
                D = max(D - 0.1, -1.0)
                A = min(A + 0.1, 1.0)
            elif user_emotion_label == 'sad':
                # 对悲伤用户：适度提高控制度以提供引导
                D = min(D + self.strategy_params['dominance_guide'], 1.0)
        
        elif user_emotion_label == 'happy':
            # 用户快乐 → 机器匹配并稍高（共享快乐）
            V = min(V + 0.15, 1.0)
            # 减少控制度，让用户自主享受快乐
            D = max(D - self.strategy_params['dominance_reduce'], -1.0)
        
        elif user_emotion_label == 'neutral':
            # 用户中性 → 机器适度积极以激发兴趣
            if V < 0.3:
                V = min(V + 0.25, 1.0)
            if A < 0.2:
                A = min(A + self.strategy_params['arousal_boost'], 1.0)
        
        elif user_emotion_label == 'surprised':
            # 用户惊讶 → 机器保持适度唤醒，提供稳定感
            A = max(A - 0.1, -1.0)  # 稍降低唤醒度以提供稳定
        
        # 策略2：确保VAD值的合理范围（防止极端值）
        V = max(min(V, 1.0), -1.0)
        A = max(min(A, 1.0), -1.0)
        D = max(min(D, 1.0), -1.0)
        
        # 策略3：确保VAD组合的合理性（避免矛盾组合）
        if V > 0.7 and A < -0.5:
            # 高度愉快但极低唤醒 → 调整唤醒度
            A = -0.2
        if V < -0.7 and D < -0.7:
            # 极不愉快且极低控制 → 适度提高控制度
            D = max(D + 0.2, -0.5)
        
        return [round(V, 3), round(A, 3), round(D, 3)]
    
    def calculate_vad(self, emotion_probs: np.ndarray, 
                     emotion_label: str = None) -> Tuple[List[float], Dict[str, float]]:
        """
        完整的情感到VAD转换流程
        
        参数:
            emotion_probs: 7类情感概率数组
            emotion_label: 主要情感标签（可选，如不提供则从概率中获取）
        
        返回:
            tuple: (VAD向量, 详细转换信息字典)
        """
        # 1. 如果没有提供情感标签，从概率中获取主要情感
        if emotion_label is None:
            emotion_label = self.emotion_order[np.argmax(emotion_probs)]
        
        # 2. 计算加权平均VAD
        weighted_vad = self.weighted_average_vad(emotion_probs)
        
        # 3. 获取基准VAD（主要情感的VAD）
        base_vad = self.map_emotion_to_vad(emotion_label)
        
        # 4. 应用引导策略
        guided_vad = self.apply_guidance_strategy(weighted_vad, emotion_label)
        
        # 5. 收集详细转换信息
        conversion_info = {
            'primary_emotion': emotion_label,
            'primary_probability': float(emotion_probs[np.argmax(emotion_probs)]),
            'base_vad': [round(v, 3) for v in base_vad],
            'weighted_vad': [round(v, 3) for v in weighted_vad],
            'guided_vad': guided_vad,
            'strategy_applied': self.get_strategy_description(emotion_label)
        }
        
        return guided_vad, conversion_info
    
    def get_strategy_description(self, emotion_label: str) -> str:
        """
        获取当前应用的策略描述
        
        参数:
            emotion_label: 用户情感标签
        
        返回:
            策略描述字符串
        """
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


# 全局转换器实例（供test.py直接导入使用）
transformer = EmotionTransformer()


# 简化接口函数（供test.py直接调用）
def emotion_to_vad(emotion_probs: np.ndarray, emotion_label: str = None) -> List[float]:
    """
    简化的情感到VAD转换函数（主接口）
    
    参数:
        emotion_probs: 情感概率数组（7个值）
        emotion_label: 可选，主要情感标签
    
    返回:
        VAD三维向量 [V, A, D]
    """
    vad_vector, _ = transformer.calculate_vad(emotion_probs, emotion_label)
    return vad_vector


def emotion_to_vad_detailed(emotion_probs: np.ndarray, 
                           emotion_label: str = None) -> Tuple[List[float], Dict[str, float]]:
    """
    详细的情感到VAD转换函数（返回转换信息）
    
    参数:
        emotion_probs: 情感概率数组（7个值）
        emotion_label: 可选，主要情感标签
    
    返回:
        tuple: (VAD向量, 转换信息字典)
    """
    return transformer.calculate_vad(emotion_probs, emotion_label)


# 测试函数（如果直接运行此文件）
if __name__ == "__main__":
    # 测试各种情感情况
    test_cases = [
        # (情感标签, 模拟概率分布)
        ("happy", [0.1, 0.0, 0.0, 0.8, 0.0, 0.05, 0.05]),
        ("sad", [0.0, 0.0, 0.1, 0.0, 0.85, 0.0, 0.05]),
        ("angry", [0.9, 0.0, 0.0, 0.0, 0.0, 0.05, 0.05]),
        ("neutral", [0.1, 0.0, 0.0, 0.1, 0.1, 0.1, 0.6]),
        ("surprised", [0.0, 0.0, 0.1, 0.1, 0.0, 0.75, 0.05]),
    ]
    
    print("=" * 60)
    print("情感到VAD转换测试")
    print("=" * 60)
    
    for label, probs in test_cases:
        probs_array = np.array(probs)
        vad_result, info = emotion_to_vad_detailed(probs_array, label)
        
        print(f"\n用户情感: {label} (概率: {probs[info['primary_probability']]*100:.1f}%)")
        print(f"基准VAD: {info['base_vad']}")
        print(f"加权VAD: {info['weighted_vad']}")
        print(f"引导后VAD: {info['guided_vad']}")
        print(f"策略: {info['strategy_applied']}")
    
    print("\n" + "=" * 60)
    print("实时转换示例（模拟test.py调用）:")
    print("=" * 60)
    
    # 模拟test.py的调用方式
    test_probs = np.array([0.1, 0.05, 0.1, 0.6, 0.05, 0.05, 0.05])  # 主要happy
    result = emotion_to_vad(test_probs)
    print(f"输入概率: {test_probs}")
    print(f"输出VAD: {result}")