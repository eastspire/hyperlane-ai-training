import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import traceback

# 加载环境变量
load_dotenv()

# 从环境变量获取配置
MERGED_MODEL_DIR = os.getenv("MERGED_MODEL_DIR")


def load_model():
    """加载合并后的模型"""
    try:
        print(f"Loading merged model from {MERGED_MODEL_DIR}")
        # 加载合并后的模型
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_DIR,
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        print("Merged model loaded successfully")

        # 加载tokenizer
        print(f"Loading tokenizer from {MERGED_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_DIR, trust_remote_code=True
        )
        print("Tokenizer loaded successfully")

        # 设置pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None, None


def generate_response(model, tokenizer, prompt):
    """生成模型响应"""
    try:
        # 准备输入
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # 生成响应
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 解码响应
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return ""


def main():
    """主函数"""
    # 加载模型和tokenizer
    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer")
        return

    # 设置提示模板
    alpaca_prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    # 问题
    question = "Who are you?"
    system_prompt = (
        "You are an AI assistant for the Hyperlane Web framework written in Rust."
    )

    # 构建完整提示
    prompt = alpaca_prompt.format(system_prompt, question)

    # 生成响应
    print(f"提问: {question}")
    response = generate_response(model, tokenizer, prompt)
    print(f"回答: {response}")


if __name__ == "__main__":
    main()
