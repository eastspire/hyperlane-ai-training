
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Remove existing outputs and models
rm -rf ./outputs
rm -rf ./deepseek-coder-1.3b-instruct
rm -rf ./deepseek-coder-1.3b-instruct-merged

# 1. Install dependencies
# We reinstall all dependencies to ensure the environment is correct.
echo "Installing Python dependencies..."

pip install "torch>=2.3.0" transformers datasets trl peft accelerate hf_xet gguf mistral_common

# 2. Generate the dataset
echo "Generating the dataset..."
python generate_markdown.py

# 3. Run fine-tuning
echo "Running fine-tuning script..."
python finetune.py

# 4. Merge the LoRA adapter
echo "Merging LoRA adapter with the base model..."
python merge_model.py

# 5. Download the GGUF conversion script
# We remove the old script if it exists to ensure we have the latest version.
rm -f convert_hf_to_gguf.py
echo "Downloading GGUF conversion script..."
curl -o convert_hf_to_gguf.py https://raw.githubusercontent.com/ggml-org/llama.cpp/master/convert_hf_to_gguf.py

# 6. Convert the model to GGUF format
echo "Converting the merged model to GGUF format..."
python convert_hf_to_gguf.py deepseek-coder-1.3b-instruct-merged --outfile deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct.gguf

echo "All tasks completed successfully!"
echo "The final GGUF model is located at: deepseek-coder-1.3b-instruct/deepseek-coder-1.3b-instruct.gguf"
