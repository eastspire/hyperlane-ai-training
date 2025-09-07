#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

python -m venv ./venv
source ./venv/Scripts/activate

# Default max_steps value
max_steps=1000

# Load environment variables from .env file
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found"
fi

# Check if the first argument is 'dev'
if [ "$1" = "dev" ]; then
    max_steps=10
    echo "Development mode enabled: max_steps=$max_steps"
fi

# Remove existing outputs and models
rm -rf ./outputs
rm -rf "./$MERGED_MODEL_DIR"
rm -rf "./$OUTPUT_DIR"

# 1. Install dependencies
# We reinstall all dependencies to ensure the environment is correct.
echo "Installing Python dependencies..."

pip install "torch>=2.3.0" transformers datasets trl peft accelerate hf_xet gguf mistral_common dotenv

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
python llama.cpp/convert_hf_to_gguf.py "$MERGED_MODEL_DIR" --outfile "$OUTPUT_DIR/$OUTPUT_DIR.gguf"

# 6. Analyze training arguments
echo "Analyzing training arguments..."
python analyze_training_args.py

# 7. Test merged model inference
echo "Testing merged model inference..."
python inference_merged.py

echo "All tasks completed successfully!"
echo "The final GGUF model is located at: $OUTPUT_DIR/$OUTPUT_DIR.gguf"