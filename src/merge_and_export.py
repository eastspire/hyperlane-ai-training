import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re
import shutil

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


def check_checkpoint_type(checkpoint_path):
    """
    Determine if the checkpoint contains LoRA adapters or a full model.
    """
    has_adapter_config = os.path.exists(
        os.path.join(checkpoint_path, "adapter_config.json")
    )
    has_adapter_model = os.path.exists(
        os.path.join(checkpoint_path, "adapter_model.safetensors")
    ) or os.path.exists(os.path.join(checkpoint_path, "adapter_model.bin"))
    has_full_model = os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))

    if has_adapter_config and has_adapter_model:
        return "lora"
    elif has_full_model:
        return "full_model"
    else:
        return "unknown"


def copy_full_model(source_path, dest_path):
    """
    Copy the full model from checkpoint directory to the destination.
    """
    print(f"Copying full model from {source_path} to {dest_path}")

    # Create destination directory
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)

    # List of files to copy (all files in the checkpoint except training-specific ones)
    exclude_files = {
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
    }

    for file_name in os.listdir(source_path):
        if file_name not in exclude_files:
            source_file = os.path.join(source_path, file_name)
            dest_file = os.path.join(dest_path, file_name)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, dest_file)
                print(f"  Copied: {file_name}")


def merge_lora_model(checkpoint_path):
    """
    Merge LoRA adapters with the base model.
    """
    print(f"Loading base model: {BASE_MODEL_ID}")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return False

    print(f"Loading LoRA adapters from: {checkpoint_path}")
    try:
        absolute_path = os.path.abspath(checkpoint_path)
        model_with_lora = PeftModel.from_pretrained(
            base_model, absolute_path, local_files_only=True
        )
    except Exception as e:
        print(f"Error loading LoRA adapters: {e}")
        return False

    print("Merging LoRA adapters into the base model...")
    try:
        merged_model = model_with_lora.merge_and_unload()
    except Exception as e:
        print(f"Error merging model: {e}")
        return False

    print(f"Saving the merged model to: {MERGED_MODEL_SAVE_PATH}")
    try:
        if not os.path.exists(MERGED_MODEL_SAVE_PATH):
            os.makedirs(MERGED_MODEL_SAVE_PATH)

        merged_model.save_pretrained(MERGED_MODEL_SAVE_PATH)
        tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)
        return True
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return False


def process_model():
    """
    Process the model based on checkpoint type (LoRA or full model).
    """
    try:
        checkpoint_path = get_latest_checkpoint_path()
        print(f"Found latest checkpoint: {checkpoint_path}")
    except ValueError as e:
        print(f"Error: {e}")
        return

    checkpoint_type = check_checkpoint_type(checkpoint_path)
    print(f"Checkpoint type detected: {checkpoint_type}")

    if checkpoint_type == "lora":
        print("Processing LoRA checkpoint...")
        success = merge_lora_model(checkpoint_path)
    elif checkpoint_type == "full_model":
        print("Processing full model checkpoint...")
        try:
            copy_full_model(checkpoint_path, MERGED_MODEL_SAVE_PATH)
            success = True
        except Exception as e:
            print(f"Error copying full model: {e}")
            success = False
    else:
        print("Error: Cannot determine checkpoint type.")
        print(
            "Expected either LoRA files (adapter_config.json, adapter_model.*) or full model (model.safetensors)"
        )
        return

    if success:
        print("\n\033[92mSuccessfully processed and saved the model!\033[0m")
        print(f"Model saved to: {MERGED_MODEL_SAVE_PATH}")
        print(
            f"Next step: Convert the model in '{MERGED_MODEL_SAVE_PATH}' to GGUF format."
        )
    else:
        print("\n\033[91mFailed to process the model.\033[0m")


if __name__ == "__main__":
    process_model()
