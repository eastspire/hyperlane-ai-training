#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

python -m venv ./venv

# Check if venv directory exists
if [ ! -d "./venv" ]; then
    echo "Error: venv directory not found. Please create a virtual environment first."
    exit 1
fi

# Detect operating system
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows system (Git Bash)
    if [ -f "./venv/Scripts/activate" ]; then
        source ./venv/Scripts/activate
        echo "Virtual environment activated (Windows)."
    else
        echo "Error: Cannot find Windows activation script."
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS system
    if [ -f "./venv/bin/activate" ]; then
        source ./venv/bin/activate
        echo "Virtual environment activated (macOS)."
    else
        echo "Error: Cannot find macOS activation script."
        exit 1
    fi
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux system
    if [ -f "./venv/bin/activate" ]; then
        source ./venv/bin/activate
        echo "Virtual environment activated (Linux)."
    else
        echo "Error: Cannot find Linux activation script."
        exit 1
    fi
else
    # Try both ways
    if [ -f "./venv/bin/activate" ]; then
        source ./venv/bin/activate
        echo "Virtual environment activated (Unix-like system)."
    elif [ -f "./venv/Scripts/activate" ]; then
        source ./venv/Scripts/activate
        echo "Virtual environment activated (Windows system)."
    else
        echo "Error: Cannot find activation script for your system."
        exit 1
    fi
fi

# Verify that the virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: Virtual environment may not be activated properly."
else
    echo "Virtual environment path: $VIRTUAL_ENV"
fi

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
git clone https://github.com/ggml-org/llama.cpp
python llama.cpp/convert_hf_to_gguf.py "$MERGED_MODEL_DIR" --outfile "$OUTPUT_DIR/$OUTPUT_DIR.gguf"

# 6. Analyze training arguments
echo "Analyzing training arguments..."
python analyze_training_args.py

# 7. Test merged model inference
echo "Testing merged model inference..."
python inference_merged.py

echo "All tasks completed successfully!"
echo "The final GGUF model is located at: $OUTPUT_DIR/$OUTPUT_DIR.gguf"
