# Hyperlane Code & Docs AI Assistant Training Project

This project provides a complete, automated pipeline to fine-tune a `Qwen/Qwen2-1.5B-Instruct` language model. The goal is to create an AI assistant that has deep knowledge of the Hyperlane, crates.dev, and ltpp-docs ecosystems.

The entire process, from data acquisition to final model conversion, is handled by a single master script: `run_all.py`.

---

## Features

- **Automated Data Acquisition**: Automatically clones all public repositories from the `crates-dev` and `hyperlane-dev` GitHub organizations, in addition to the `ltpp-docs` repository.
- **AI-Powered Data Generation**: Uses the base Qwen2 model to read all downloaded source code and documentation, automatically generating high-quality Question-Answer pairs.
- **Instruction Fine-Tuning**: Trains the model on the generated Q&A pairs, specifically teaching it to answer questions based on the provided knowledge.
- **End-to-End Automation**: A single command executes the entire pipeline: data cloning, QA generation, training, model merging, and conversion to GGUF for LM Studio.

---

## Project Structure

The project is organized as follows:

- `run_all.py`: The main script to execute the entire pipeline.
- `src/`: Contains all the Python scripts for the different pipeline stages.
  - `acquire_data.py`: Clones repositories for training data.
  - `prepare_data.py`: Prepares the training data.
  - `train.py`: Fine-tunes the base model.
  - `merge_and_export.py`: Merges the LoRA adapters.
  - `gguf_converter.py`: Converts the model to GGUF format.
- `llama.cpp/`: A submodule for GGUF conversion.
- `outputs/`: Contains training checkpoints and logs.
- `training_sources/`: Stores the cloned source code and documents for training.
- `README.md`: This file.

---

## Prerequisites

1.  **Python 3.9+**
2.  **Git**: Must be installed and accessible from the command line.
3.  **Significant Disk Space**: The data acquisition step will download many repositories, requiring several gigabytes of storage.
4.  **Stable Internet Connection**: Required for downloading repositories and the base model.

---

## How to Run

The entire process is managed by the `run_all.py` script. Follow these steps:

### 1. Setup the Environment

First, set up a dedicated Python virtual environment. Open your terminal (PowerShell, CMD, etc.) in this project's root directory (`ai-training`).

```bash
# Create the virtual environment
python -m venv src/venv

# Activate the virtual environment
# On Windows:
.\src\venv\Scripts\activate
# On macOS/Linux:
# source src/venv/bin/activate
```

### 2. Install Dependencies

Install all necessary Python libraries from the `requirements.txt` file.

```bash
# Ensure your venv is active
pip install -r src/requirements.txt
```

### 3. Execute the Pipeline

Run the master script. This will trigger the entire end-to-end process.

**Warning:** This process will take a very long time, potentially many hours, depending on your internet speed and CPU performance. It is fully automated, so you can let it run in the background.

```bash
python run_all.py
```

---

## The Pipeline Explained

The `run_all.py` script executes the following steps in order:

1.  **`acquire_data.py`**: Clones or updates all specified GitHub repositories into the `training_sources` directory.
2.  **`prepare_data.py`**: Scans all downloaded files (code and markdown) and uses the base AI model to generate a `training_data.jsonl` file filled with Q&A pairs.
3.  **`train.py`**: Fine-tunes the Qwen2 model on the generated Q&A data for 3 full cycles (epochs).
4.  **`merge_and_export.py`**: Merges the fine-tuning data (LoRA adapters) back into the base model.
5.  **`gguf_converter.py`**: Converts the final, merged model into the GGUF format required by LM Studio.

## Final Output

Upon successful completion, you will find the final, ready-to-use model file in this directory:

- **`hyperlane-qwen2.5-coder-1.5b-instruct.gguf`**

Simply copy this file to your LM Studio models folder, and you can begin chatting with your custom-trained AI assistant.
