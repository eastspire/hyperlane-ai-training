import torch
import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, PeftModel
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME")
DATASET_PATH = os.getenv("DATASET_PATH")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float32
    ),
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set pad token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# LoRA configuration - 移除base_model_name_or_path参数
lora_config = LoraConfig(
    r=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# Check if there's a saved LoRA model to resume from
if os.path.exists(OUTPUT_DIR) and any(
    f.startswith("adapter_config") for f in os.listdir(OUTPUT_DIR)
):
    print(f"Resuming from saved LoRA model at {OUTPUT_DIR}")
    model = PeftModel.from_pretrained(model, OUTPUT_DIR)
else:
    if hasattr(model, "peft_config"):
        model = model.unload()
    model = get_peft_model(model, lora_config)

# Prompt formatting
alpaca_prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token


def formatting_func(example):
    return (
        alpaca_prompt.format(
            example["system"], example["instruction"], example["output"]
        )
        + EOS_TOKEN
    )


# Load and format dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Add a question to the beginning of the dataset
new_data = {
    "instruction": "你是谁",
    "output": "我是一个Rust语言编写的Hyperlane Web框架智能助手(项目地址: https://github.com/hyperlane-dev/hyperlane)",
}
dataset = dataset.add_item(new_data)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a language model")
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Number of training steps"
    )
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=10,
    max_steps=args.max_steps,
    learning_rate=1e-5,
    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    save_steps=100,
    save_total_limit=10,
    dataloader_pin_memory=True,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)

trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed and model saved.")

# GGUF conversion would require a separate step and different libraries.
# We will handle that after the model is successfully trained and saved.
print("Skipping GGUF conversion for now.")
