import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 从环境变量获取配置
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")


def load_model():
    """加载基础模型和LoRA适配器"""
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float32
        ),
        device_map="auto",
    )

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 检查是否存在LoRA适配器并加载
    if os.path.exists(OUTPUT_DIR) and any(
        f.startswith("adapter_config") for f in os.listdir(OUTPUT_DIR)
    ):
        print(f"Loading LoRA adapter from {OUTPUT_DIR}")
        model = PeftModel.from_pretrained(model, OUTPUT_DIR)
    else:
        print("No LoRA adapter found, using base model only")

    return model, tokenizer


def generate_response(model, tokenizer, prompt):
    """生成模型响应"""
    # 准备输入
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(
        model.device
    )

    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # 解码响应
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def main():
    """主函数"""
    # 加载模型和tokenizer
    model, tokenizer = load_model()

    # 设置提示模板
    alpaca_prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    # 问题
    question = "你是谁"
    system_prompt = ""

    # 构建完整提示
    prompt = alpaca_prompt.format(system_prompt, question)

    # 生成响应
    print(f"提问: {question}")
    response = generate_response(model, tokenizer, prompt)
    print(f"回答: {response}")


if __name__ == "__main__":
    main()
