import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_MODEL_NAME = "hyperlane-qwen2.5-coder-1.5b-finetuned"
DATASET_FILE = "./training_data.jsonl"
# --- End Configuration ---


def formatting_func(example):
    """Simply return the text field from each dataset example."""
    return example["text"]


def train():
    print(f"Loading base model '{BASE_MODEL}' for auto training...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Automatically select precision
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and torch.cuda.is_fp16_supported()

    if use_bf16:
        torch_dtype = torch.bfloat16
        print("Using bf16 precision for training.")
    elif use_fp16:
        torch_dtype = torch.float16
        print("Using fp16 precision for training.")
    else:
        torch_dtype = torch.float32
        print("Using fp32 precision for training (GPU bf16/fp16 not supported).")

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )

    print("Configuring model for LoRA (Parameter-Efficient Fine-Tuning)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=10,
            learning_rate=2e-4,
            logging_steps=1,
            bf16=use_bf16,
            fp16=use_fp16 and not use_bf16,
            optim="adamw_torch",
            report_to="none",
            output_dir="./outputs",
            dataloader_pin_memory=False,
        ),
    )

    print("\n\033[94mStarting model training on auto.\033[0m")
    trainer.train()
    print("\n\033[92mTraining complete!\033[0m")

    print(f"Saving fine-tuned model to '{OUTPUT_MODEL_NAME}'...")
    model.save_pretrained(OUTPUT_MODEL_NAME)
    tokenizer.save_pretrained(OUTPUT_MODEL_NAME)
    print(
        f"\n\033[92mModel saved in Hugging Face format at '{OUTPUT_MODEL_NAME}'!\033[0m"
    )
    print("You will need to convert this model to GGUF format manually for LM Studio.")


if __name__ == "__main__":
    train()
