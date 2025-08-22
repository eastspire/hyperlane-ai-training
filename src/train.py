import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
OUTPUT_MODEL_NAME = "hyperlane-qwen2.5-coder-1.5b-finetuned"
DATASET_FILE = "./training_data.jsonl"
# --- End Configuration ---


# Modern TRL library prefers a formatting function over 'dataset_text_field'.
def formatting_func(example):
    return example["text"]


def train():
    """
    Loads the base model, the dataset, and runs the fine-tuning process with LoRA + 4bit quantization.
    """
    print(
        f"Loading base model '{BASE_MODEL}' with 4-bit quantization for GPU training..."
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 4-bit quantization config ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # 更安全，显存更省
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config,
        trust_remote_code=True,
    )

    # 启用 gradient checkpointing 节省显存
    model.gradient_checkpointing_enable()

    print("Configuring model for LoRA (Parameter-Efficient Fine-Tuning)...")
    lora_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print(f"Loading dataset from {DATASET_FILE}...")
    dataset = load_dataset("json", data_files=DATASET_FILE, split="train")

    # --- Training config ---
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,
        args=TrainingArguments(
            per_device_train_batch_size=1,  # 改为1，避免OOM
            gradient_accumulation_steps=32,  # 增大学习等效batch
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
    print("You will need to convert this model to GGUF format manually for LM Studio.")


if __name__ == "__main__":
    train()
