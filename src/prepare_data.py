import os
import glob
import json
from tqdm import tqdm

# --- Configuration ---
SOURCE_PATH = "./training_sources"
FILE_PATTERNS = ["**/*.md", "**/*.rs", "**/*.toml", "**/*.py", "**/*.js", "**/*.ts"]
OUTPUT_FILE = "./training_data.jsonl"
# --- End Configuration ---


def create_direct_text_training_data():
    """
    Processes all specified files and saves them directly
    to a JSONL file for training. Each file becomes one JSONL entry.
    """
    print("Starting direct text data preparation...")

    all_files = []
    print(f"Searching for files in: {os.path.abspath(SOURCE_PATH)}")
    for pattern in FILE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(SOURCE_PATH, pattern), recursive=True))

    print(f"Found {len(all_files)} total files to process.")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as in_file:
                    content = in_file.read()
                    if not content.strip():  # Skip empty files
                        continue
                    f.write(json.dumps({"text": content}) + "\n")
            except Exception as e:
                print(f"Skipping file {file_path} due to error: {e}")
                continue

    print(
        f"\n\033[92mSuccessfully created direct text training data at: {os.path.abspath(OUTPUT_FILE)}\033[0m"
    )


if __name__ == "__main__":
    create_direct_text_training_data()
