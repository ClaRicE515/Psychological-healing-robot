# data_logger.py
import time
import os

class EmotionLogger:
    def __init__(self, folder="data", filename="log.txt"):
        # 1. 组合完整路径 (例如: data/emotion_log_2023.txt)
        self.full_path = os.path.join(folder, filename)
        
        # 2. 如果文件夹不存在，则创建它
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"已创建数据文件夹: {folder}")

        # 3. 写入表头（因为是新文件，直接写即可）
        with open(self.full_path, "w", encoding="utf-8") as f:
            f.write("Time, Angry, Disgust, Scared, Happy, Sad, Surprised, Neutral\n")

    def log(self, avg_preds):
        """将平均比例追加到文件末尾"""
        timestamp = time.strftime("%H:%M:%S")
        preds_str = ", ".join(["{:.2f}%".format(x * 100) for x in avg_preds])
        
        # 使用追加模式 "a"
        with open(self.full_path, "a", encoding="utf-8") as f:
            f.write(f"{timestamp}, {preds_str}\n")