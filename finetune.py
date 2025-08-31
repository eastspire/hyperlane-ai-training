import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# Constants
MODEL_NAME = "qwen/qwen2-1.5b"
DATASET_PATH = "dataset/dataset.json"
OUTPUT_DIR = "qwen3-4b"
MAX_SEQ_LENGTH = 2048

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
    per_device_train_batch_size=1,  # Reduced for CPU
    gradient_accumulation_steps=1,  # Reduced for CPU
    warmup_steps=1,
    max_steps=3,  # Reduced for CPU to run faster as a test
    learning_rate=2e-4,
    fp16=False,  # Not using fp16 on CPU
    bf16=False,  # Not using bf16 on CPU
    logging_steps=1,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="linear",
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
    # The following arguments are now correctly placed or replaced
    # tokenizer=tokenizer, # Replaced by processing_class, but SFTTrainer might not even need it explicitly
    # dataset_text_field="text", # Replaced by formatting_func
    # max_seq_length=MAX_SEQ_LENGTH, # Should be in TrainingArguments or SFTConfig
)

trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Fine-tuning completed and model saved.")

# GGUF conversion would require a separate step and different libraries.
# We will handle that after the model is successfully trained and saved.
print("Skipping GGUF conversion for now.")
