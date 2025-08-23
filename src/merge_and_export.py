import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

# --- Configuration ---
# The base model ID, must be the same as in the training script.
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# The path where the final, merged model will be saved.
MERGED_MODEL_SAVE_PATH = "hyperlane-qwen2.5-coder-1.5b-merged"
# --- End Configuration ---


def get_latest_checkpoint_path(base_dir="outputs"):
    """
    Finds the path to the latest checkpoint directory within the base directory.
    """
    if not os.path.isdir(base_dir):
        raise ValueError(f"Directory not found: {base_dir}")

    checkpoint_dirs = [
        d
        for d in os.listdir(base_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not checkpoint_dirs:
        raise ValueError(f"No checkpoint directories found in '{base_dir}'")

    # Find the checkpoint with the highest number
    latest_checkpoint_dir = max(
        checkpoint_dirs, key=lambda d: int(re.search(r"checkpoint-(\d+)", d).group(1))
    )

    return os.path.join(base_dir, latest_checkpoint_dir)


def merge_model():
    """
    Merges the LoRA adapters with the base model and saves the result.
    """
    try:
        FINETUNED_MODEL_PATH = get_latest_checkpoint_path()
        print(f"Found latest checkpoint: {FINETUNED_MODEL_PATH}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    print(f"Loading base model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)

    print(f"Loading LoRA adapters from: {FINETUNED_MODEL_PATH}")
    # Load the LoRA model
    model_with_lora = PeftModel.from_pretrained(base_model, FINETUNED_MODEL_PATH)

    print("Merging LoRA adapters into the base model...")
    # Merge the adapters into the base model
    merged_model = model_with_lora.merge_and_unload()

    print(f"Saving the merged model to: {MERGED_MODEL_SAVE_PATH}")
    # Create directory if it doesn't exist
    if not os.path.exists(MERGED_MODEL_SAVE_PATH):
        os.makedirs(MERGED_MODEL_SAVE_PATH)

    # Save the merged model and tokenizer
    merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

    print("\n\033[92mSuccessfully merged and saved the model!\033[0m")
    print(f"Next step: Convert the model in '{MERGED_MODEL_SAVE_PATH}' to GGUF format.")


if __name__ == "__main__":
    merge_model()
