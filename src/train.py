import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# --- Configuration ---
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_MODEL_NAME = "hyperlane-qwen2-finetuned-model"
DATASET_FILE = "./training_data.jsonl"
# --- End Configuration ---


# Modern TRL library prefers a formatting function over 'dataset_text_field'.
# This function simply takes a sample from the dataset and returns the text.
def formatting_func(example):
    return example["text"]


def train():
    """
    Loads the base model, the dataset, and runs the fine-tuning process.
    """
    print(f"Loading base model '{BASE_MODEL}' for GPU training...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
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

    # Configure the trainer using the modern API with formatting_func
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        formatting_func=formatting_func,  # Use the new formatting_func
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=2,
            num_train_epochs=3,  # Increased epochs for better learning
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_torch",
            report_to="none",  # Disable wandb integration
            output_dir="./outputs",
            dataloader_pin_memory=False,  # Disable for CPU-only environments
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
