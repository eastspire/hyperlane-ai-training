# Hyperlane AI Training Project

This project provides a complete pipeline for fine-tuning language models and converting them to GGUF format for efficient inference.

## Project Overview

The pipeline includes the following steps:

1. Environment setup with Python virtual environment
2. Dependency installation
3. Dataset generation
4. Model fine-tuning with LoRA adapters
5. Merging LoRA adapters with the base model
6. Converting the merged model to GGUF format
7. Analyzing training arguments

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Setup and Usage

### 1. Clone the Repository

```bash
git clone <repository-url>
cd hyperlane-ai-training
```

### 2. Run the Training Pipeline

Execute the main script to run the complete pipeline:

```bash
./run.sh
```

This will:

- Create and activate a Python virtual environment
- Install all required dependencies
- Generate the dataset
- Fine-tune the model
- Merge the LoRA adapter with the base model
- Convert the merged model to GGUF format
- Analyze training arguments

### 3. Development Mode

For faster iteration during development, you can run the pipeline in development mode which limits the number of training steps:

```bash
./run.sh dev
```

## Configuration

The project can be configured using a `.env` file in the root directory. The following environment variables are available:

- `MERGED_MODEL_DIR`: Directory for the merged model (default: "merged_model")
- `OUTPUT_DIR`: Directory for the output files (default: "output")

Example `.env` file:

```
MERGED_MODEL_DIR=my_merged_model
OUTPUT_DIR=my_output
```

## Project Structure

- `run.sh`: Main execution script
- `generate_markdown.py`: Script to generate training dataset
- `finetune.py`: Model fine-tuning script
- `merge_model.py`: Script to merge LoRA adapters with the base model
- `convert_hf_to_gguf.py`: Script to convert models to GGUF format
- `analyze_training_args.py`: Script to analyze and log training arguments
- `dataset/`: Directory containing the training dataset

## Dependencies

The project requires the following Python packages:

- torch (>=2.3.0)
- transformers
- datasets
- trl
- peft
- accelerate
- hf_xet
- gguf
- mistral_common
- dotenv

## Output

After successful execution, the final GGUF model will be located at: `$OUTPUT_DIR/$OUTPUT_DIR.gguf`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## Contact

For any inquiries, please reach out to the author at [root@ltpp.vip](mailto:root@ltpp.vip).
