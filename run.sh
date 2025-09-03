
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Default max_steps value
max_steps=-1

# Check if the first argument is 'dev'
if [ "$1" = "dev" ]; then
    max_steps=1
    echo "Development mode enabled: max_steps=$max_steps"
fi

# Remove existing outputs and models
rm -rf ./outputs
rm -rf ./Qwen3-4B-Instruct-2507
rm -rf ./Qwen3-4B-Instruct-2507-merged

# 1. Install dependencies
# We reinstall all dependencies to ensure the environment is correct.
echo "Installing Python dependencies..."

pip install "torch>=2.3.0" transformers datasets trl peft accelerate hf_xet gguf mistral_common

# 2. Generate the dataset
echo "Generating the dataset..."
python generate_markdown.py

# 3. Run fine-tuning
echo "Running fine-tuning script..."
python finetune.py --max_steps $max_steps

# 4. Merge the LoRA adapter
echo "Merging LoRA adapter with the base model..."
python merge_model.py

# 5. Convert the model to GGUF format
echo "Converting the merged model to GGUF format..."
python convert_hf_to_gguf.py Qwen3-4B-Instruct-2507-merged --outfile Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf

echo "All tasks completed successfully!"
echo "The final GGUF model is located at: Qwen3-4B-Instruct-2507/Qwen3-4B-Instruct-2507.gguf"
