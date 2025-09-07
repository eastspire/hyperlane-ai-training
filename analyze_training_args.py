import torch
import os
from dotenv import load_dotenv
from trl import SFTConfig

# 加载环境变量
load_dotenv()

# 从环境变量获取配置
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

def analyze_training_args():
    """分析训练参数"""
    try:
        # 添加安全全局变量
        torch.serialization.add_safe_globals([SFTConfig])
        
        # 加载训练参数
        training_args_path = os.path.join(OUTPUT_DIR, "training_args.bin")
        if os.path.exists(training_args_path):
            print(f"Loading training args from {training_args_path}")
            training_args = torch.load(training_args_path, weights_only=False)
            print("Training arguments:")
            for key, value in training_args.__dict__.items():
                # 过滤掉一些不必要的属性
                if not key.startswith('_') and not callable(value):
                    print(f"  {key}: {value}")
        else:
            print(f"Training args file not found at {training_args_path}")
    except Exception as e:
        print(f"Error analyzing training args: {e}")

if __name__ == "__main__":
    analyze_training_args()