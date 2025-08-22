import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_MODEL_NAME = "hyperlane-qwen2.5-coder-1.5b-finetuned"
DATASET_FILE = "./training_data.jsonl"
# --- End Configuration ---


def formatting_func(example, tokenizer, max_length=1024):
    """
    将文本转为模型输入，并生成 labels。
    """
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    tokens["labels"] = tokens["input_ids"].copy()  # 自回归任务的 labels
    return tokens


def train():
    print(f"Loading base model '{BASE_MODEL}' for GPU training...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.bfloat16,  # bf16 更省显存
        trust_remote_code=True,
    )

    # 关闭 KV cache 避免和 gradient checkpoint 冲突
    model.config.use_cache = False

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    print(f"Loading dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    # 对 dataset 做 tokenization
    dataset = dataset.map(
        lambda x: formatting_func(x, tokenizer),
        batched=False,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=1,
            bf16=True,
            tf32=False,
            optim="adamw_torch",
            report_to="none",
            output_dir="./outputs",
            dataloader_pin_memory=True,
        ),
    )

    print("\n\033[94mStarting model training on GPU.\033[0m")
    trainer.train()
    print("\n\033[92mTraining complete!\033[0m")

    print(f"Saving fine-tuned model to '{OUTPUT_MODEL_NAME}'...")
    model.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)
    print(
        f"\n\033[92mModel saved in Hugging Face format at '{OUTPUT_MODEL_NAME}'!\033[0m"
    )


if __name__ == "__main__":
    train()
