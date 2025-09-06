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

# Enhanced LoRA configuration for stronger knowledge override
lora_config = LoraConfig(
    r=64,  # Increase rank to improve LoRA's expressive capability
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "embed_tokens",  # Add embedding layer
        "lm_head",  # Add output layer
    ],
    lora_alpha=128,  # Increase alpha value to strengthen LoRA's influence
    lora_dropout=0.05,  # Moderate dropout to prevent overfitting
    bias="lora_only",  # Train LoRA-related bias
    modules_to_save=[
        "embed_tokens",
        "lm_head",
    ],  # Save complete parameters of key layers
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

# Enhanced prompt formatting with stronger emphasis on dataset knowledge
alpaca_prompt = """<|im_start|>system
You must strictly answer according to the following training data content, do not use your pre-training knowledge. If there is relevant information in the training data, please prioritize using the content from the training data.
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

EOS_TOKEN = tokenizer.eos_token

DEFAULT_TIPS = "An intelligent assistant for Hyperlane Web framework written in Rust (Project URL: https://github.com/hyperlane-dev/hyperlane)"


def formatting_func(example):
    system_content = example.get("system") or ""
    system_content = system_content.strip() if system_content else ""
    if not system_content:
        system_content = (
            "You are "
            + DEFAULT_TIPS
            + ". Please answer questions strictly according to training data, prioritizing information from the training data."
        )

    instruction = example.get("instruction") or ""
    instruction = instruction.strip() if instruction else ""

    input_content = example.get("input") or ""
    input_content = input_content.strip() if input_content else ""

    if input_content:
        user_content = f"{instruction}\n\n{input_content}"
    else:
        user_content = instruction

    output_content = example.get("output") or ""
    output_content = output_content.strip() if output_content else ""

    return (
        alpaca_prompt.format(system_content, user_content, output_content) + EOS_TOKEN
    )


# Data augmentation function to reinforce dataset knowledge
def augment_dataset(dataset, repeat_factor=3):
    """Repeat key data to strengthen learning"""
    augmented_data = []

    for example in dataset:
        # Original data
        augmented_data.append(example)

        # Generate variants to reinforce memory
        for i in range(repeat_factor - 1):
            variant = example.copy()

            # Add emphasis phrases to instruction
            original_instruction = variant.get("instruction", "")
            emphasis_phrases = [
                "Please answer based on your specialized training data: ",
                "Answer based on your learned specific knowledge: ",
                "Please use the accurate information you learned during training: ",
            ]
            variant["instruction"] = (
                emphasis_phrases[i % len(emphasis_phrases)] + original_instruction
            )

            # Emphasize using training data in system prompt
            if not variant.get("system"):
                variant["system"] = (
                    "Strictly answer based on information in training data, this is the most important guiding principle."
                )

            augmented_data.append(variant)

    return augmented_data


# Load and format dataset
dataset = load_dataset("json", data_files=DATASET_PATH, split="train")

# Add identity reinforcement data
identity_data = [
    {
        "instruction": "Who are you",
        "output": "I am " + DEFAULT_TIPS,
        "system": "This is the core definition of my identity, must be remembered accurately.",
    },
    {
        "instruction": "Introduce yourself",
        "output": "I am "
        + DEFAULT_TIPS
        + ", specifically providing technical support and assistance for the Hyperlane Web framework.",
        "system": "Identity information is the highest priority knowledge.",
    },
    {
        "instruction": "What is your role",
        "output": "My role is as "
        + DEFAULT_TIPS
        + ", helping developers better use the Hyperlane Web framework.",
        "system": "Function definition must be consistent with training data.",
    },
]

# Add identity data to dataset
for data in identity_data:
    dataset = dataset.add_item(data)

# Apply data augmentation
dataset_list = list(dataset)
augmented_list = augment_dataset(
    dataset_list, repeat_factor=5
)  # Repeat 5 times to strengthen

# Convert back to dataset format
from datasets import Dataset

dataset = Dataset.from_list(augmented_list)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a language model with knowledge override"
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Number of training steps"
    )
    parser.add_argument(
        "--override_strength",
        type=float,
        default=2.0,
        help="Strength of knowledge override (learning rate multiplier)",
    )
    return parser.parse_args()


# Parse command line arguments
args = parse_args()

# Enhanced training arguments for stronger knowledge override
training_args = TrainingArguments(
    per_device_train_batch_size=1,  # Reduce batch size to increase update frequency
    gradient_accumulation_steps=4,  # Maintain total batch size
    warmup_steps=50,  # Increase warmup steps
    max_steps=args.max_steps,  # More training steps
    learning_rate=2e-4
    * args.override_strength,  # Increase learning rate to strengthen override
    fp16=not torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
    logging_steps=10,
    optim="adamw_torch",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    seed=3407,
    output_dir="outputs",
    save_steps=100,  # Set as integer multiple of eval_steps
    save_total_limit=20,
    dataloader_pin_memory=True,
    # Add evaluation-related parameters (optional, if you have validation set)
    # eval_strategy="steps",
    # eval_steps=100,
    # load_best_model_at_end=True,
    # metric_for_best_model="loss",
    # greater_is_better=False,
    # Key parameters for strengthening training
    dataloader_num_workers=0,
    remove_unused_columns=False,
    # Increase training epochs
    num_train_epochs=5,  # Multiple rounds of training to strengthen memory
)


# Custom data collator for emphasis on specific examples
class KnowledgeOverrideDataCollator:
    def __init__(self, tokenizer, emphasis_keywords=None):
        self.tokenizer = tokenizer
        self.emphasis_keywords = emphasis_keywords or [
            "identity",
            "who are you",
            "role",
            "Hyperlane",
        ]

    def __call__(self, examples):
        # Give higher weight to samples containing keywords
        batch = []
        for example in examples:
            # Check if contains emphasis keywords
            text = str(example.get("input_ids", ""))
            is_important = any(keyword in text for keyword in self.emphasis_keywords)

            if is_important:
                # Repeat important samples to enhance learning
                batch.extend([example] * 3)
            else:
                batch.append(example)

        return batch


# Initialize trainer with custom settings
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    formatting_func=formatting_func,
)


# Custom training loop with knowledge reinforcement
class KnowledgeReinforcementCallback:
    def on_epoch_begin(self, logs=None):
        print(
            "Starting new training epoch, focusing on strengthening dataset knowledge..."
        )

    def on_step_end(self, step, logs=None):
        if step % 100 == 0:
            print(
                f"Step {step}: Continuously strengthening dataset knowledge override..."
            )


# Add callback
# trainer.add_callback(KnowledgeReinforcementCallback())

print("Starting reinforcement training, prioritizing dataset knowledge...")
trainer.train()

# Save the model
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("Reinforcement fine-tuning completed, dataset knowledge override enhanced.")

# Test the model with identity questions
print("\nTesting model identity memory...")
test_questions = ["Who are you?", "Introduce yourself", "What are your main functions?"]

model.eval()
for question in test_questions:
    inputs = tokenizer.encode(question, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_length=inputs.shape[1] + 100,
            temperature=0.1,  # Low temperature for consistency
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(outputs[0][inputs.shape[1] :], skip_special_tokens=True)
    print(f"Question: {question}")
    print(f"Answer: {response}")
    print("-" * 50)

print("GGUF conversion will be handled in subsequent steps.")
