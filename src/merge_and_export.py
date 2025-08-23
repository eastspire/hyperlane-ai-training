import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re
import shutil
import json

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
    def extract_checkpoint_number(d):
        match = re.search(r"checkpoint-(\d+)", d)
        if match:
            return int(match.group(1))
        return 0

    latest_checkpoint_dir = max(checkpoint_dirs, key=extract_checkpoint_number)
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
    has_full_model = (
        os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
        or os.path.exists(os.path.join(checkpoint_path, "pytorch_model.bin"))
        or any(
            f.startswith("model-") and f.endswith(".safetensors")
            for f in os.listdir(checkpoint_path)
        )
    )

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
    if os.path.exists(dest_path):
        print(f"Destination directory {dest_path} already exists. Removing it.")
        shutil.rmtree(dest_path)
    os.makedirs(dest_path)

    # List of files to copy (all files in the checkpoint except training-specific ones)
    exclude_files = {
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
        "trainer_state.json",
        "pytorch_model.bin.index.json",  # Sometimes present in checkpoints
    }

    copied_count = 0
    for file_name in os.listdir(source_path):
        if file_name not in exclude_files:
            source_file = os.path.join(source_path, file_name)
            dest_file = os.path.join(dest_path, file_name)
            if os.path.isfile(source_file):
                try:
                    shutil.copy2(source_file, dest_file)
                    print(f"  Copied: {file_name}")
                    copied_count += 1
                except Exception as e:
                    print(f"  Error copying {file_name}: {e}")
            elif os.path.isdir(source_file):
                # Copy directories as well (like tokenizer_config directories)
                try:
                    shutil.copytree(source_file, dest_file)
                    print(f"  Copied directory: {file_name}")
                    copied_count += 1
                except Exception as e:
                    print(f"  Error copying directory {file_name}: {e}")

    if copied_count == 0:
        raise RuntimeError("No files were copied. Check the source directory.")

    print(f"Successfully copied {copied_count} items.")


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

        # Ensure tokenizer has a pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    except Exception as e:
        print(f"Error loading base model: {e}")
        return False

    print(f"Loading LoRA adapters from: {checkpoint_path}")
    try:
        # Verify adapter files exist
        adapter_config_path = os.path.join(checkpoint_path, "adapter_config.json")
        if not os.path.exists(adapter_config_path):
            raise FileNotFoundError("adapter_config.json not found")

        # Check adapter config for compatibility
        with open(adapter_config_path, "r") as f:
            adapter_config = json.load(f)
            print(
                f"Adapter target modules: {adapter_config.get('target_modules', 'unknown')}"
            )
            print(f"Adapter rank: {adapter_config.get('r', 'unknown')}")

        absolute_path = os.path.abspath(checkpoint_path)
        model_with_lora = PeftModel.from_pretrained(
            base_model, absolute_path, local_files_only=True, torch_dtype=torch.float16
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
        # Clean up existing directory
        if os.path.exists(MERGED_MODEL_SAVE_PATH):
            print(
                f"Destination directory {MERGED_MODEL_SAVE_PATH} already exists. Removing it."
            )
            shutil.rmtree(MERGED_MODEL_SAVE_PATH)
        os.makedirs(MERGED_MODEL_SAVE_PATH)

        # Save model and tokenizer
        merged_model.save_pretrained(
            MERGED_MODEL_SAVE_PATH,
            safe_serialization=True,  # Use safetensors format
            max_shard_size="2GB",  # Control shard size
        )
        tokenizer.save_pretrained(MERGED_MODEL_SAVE_PATH)

        # Copy additional files from base model if needed
        try:
            base_model_files = [
                "config.json",
                "generation_config.json",
                "tokenizer_config.json",
            ]
            for file_name in base_model_files:
                source_file = os.path.join(checkpoint_path, file_name)
                if os.path.exists(source_file):
                    dest_file = os.path.join(MERGED_MODEL_SAVE_PATH, file_name)
                    if not os.path.exists(
                        dest_file
                    ):  # Don't overwrite if already exists
                        shutil.copy2(source_file, dest_file)
                        print(f"  Copied additional file: {file_name}")
        except Exception as e:
            print(f"Warning: Could not copy additional files: {e}")

        return True
    except Exception as e:
        print(f"Error saving merged model: {e}")
        return False


def verify_merged_model(model_path):
    """
    Verify that the merged model can be loaded successfully.
    """
    print(f"Verifying merged model at: {model_path}")
    try:
        # Try to load the merged model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="cpu",  # Use CPU to avoid GPU memory issues
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Model loaded successfully with {num_params:,} parameters")

        # Quick test
        test_input = tokenizer.encode("def hello_world():", return_tensors="pt")
        with torch.no_grad():
            output = model.generate(
                test_input,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        test_output = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Test generation: {test_output}")

        return True
    except Exception as e:
        print(f"Error verifying model: {e}")
        return False


def process_model():
    """
    Process the model based on checkpoint type (LoRA or full model).
    """
    try:
        checkpoint_path = get_latest_checkpoint_path()
        print(f"Found latest checkpoint: {checkpoint_path}")

        # List checkpoint contents for debugging
        print(f"Checkpoint contents: {os.listdir(checkpoint_path)}")

    except ValueError as e:
        print(f"Error: {e}")
        return

    checkpoint_type = check_checkpoint_type(checkpoint_path)
    print(f"Checkpoint type detected: {checkpoint_type}")

    success = False

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
        print("Expected either:")
        print("  - LoRA files: adapter_config.json + adapter_model.safetensors/bin")
        print("  - Full model: model.safetensors or pytorch_model.bin")
        print(
            f"Found files: {os.listdir(checkpoint_path) if os.path.exists(checkpoint_path) else 'Directory not found'}"
        )
        return

    if success:
        print("\n\033[92m=== MODEL PROCESSING SUCCESSFUL ===\033[0m")
        print(f"Model saved to: {MERGED_MODEL_SAVE_PATH}")

        # Verify the merged model
        if verify_merged_model(MERGED_MODEL_SAVE_PATH):
            print("\n\033[92mModel verification passed!\033[0m")
        else:
            print(
                "\n\033[93mWarning: Model verification failed. Check the merged model manually.\033[0m"
            )

        print(f"\nNext steps:")
        print(
            f"1. Test the model: python -c \"from transformers import AutoModelForCausalLM; model = AutoModelForCausalLM.from_pretrained('{MERGED_MODEL_SAVE_PATH}')\""
        )
        print(f"2. Convert to GGUF format if needed")
        print(f"3. Upload or deploy your model")
    else:
        print("\n\033[91mFailed to process the model.\033[0m")
        print("Check the error messages above for details.")


if __name__ == "__main__":
    # Add some system info for debugging
    print(
        f"Python version: {torch.__version__ if 'torch' in globals() else 'torch not loaded'}"
    )
    print(f"Working directory: {os.getcwd()}")
    print("=" * 50)

    process_model()
