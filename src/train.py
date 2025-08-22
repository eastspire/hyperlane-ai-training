import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_MODEL_NAME = "hyperlane-qwen2.5-coder-1.5b-finetuned"
DATASET_FILE = "./training_data.jsonl"
MAX_LENGTH = 32768


def formatting_func(example):
    tokenizer = formatting_func.tokenizer
    tokens = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


def train():
    print(f"Loading base model '{BASE_MODEL}' for GPU training...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatting_func.tokenizer = tokenizer

    # Auto precision selection
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    if use_bf16:
        torch_dtype = torch.bfloat16
        print("Using bf16 precision for training.")
    elif use_fp16:
        torch_dtype = torch.float16
        print("Using fp16 precision for training.")
    else:
        torch_dtype = torch.float32
        print("Using fp32 precision for training (GPU not available).")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    print(f"Loading dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")
    dataset = dataset.map(formatting_func, batched=False)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=None,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=3,
            learning_rate=2e-4,
            logging_steps=1,
            bf16=use_bf16,
            fp16=use_fp16,
            tf32=False,
            optim="adamw_torch",
            report_to="none",
            output_dir="./outputs",
            dataloader_pin_memory=True,
        ),
    )

    print("\nStarting model training on GPU.")
    trainer.train()
    print("\nTraining complete!")

    print(f"Saving fine-tuned model to '{OUTPUT_MODEL_NAME}'...")
    model.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)
    print(f"Model saved in Hugging Face format at '{OUTPUT_MODEL_NAME}'!")


if __name__ == "__main__":
    train()
