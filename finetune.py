import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Constants
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
DATASET_PATH = "dataset/dataset.json"
OUTPUT_DIR = "deepseek-coder-1.3b-instruct"


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

# LoRA configuration
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

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=100,
    learning_rate=5e-5,
    fp16=False,
    bf16=False,
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
)

# Create and run trainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
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
