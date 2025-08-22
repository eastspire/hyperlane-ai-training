import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

# --- Configuration ---
# The script now looks in the 'training_sources' directory.
SOURCE_PATH = "./training_sources"
# We now include markdown, Rust, TOML, and other common code files.
FILE_PATTERNS = ["readme/README.md"]
OUTPUT_FILE = "./training_data.jsonl"
BASE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

QA_GENERATION_PROMPT = """
Based on the following text from a technical document or source code file, generate exactly two high-quality, distinct question-and-answer pairs. The questions should be technical and specific, something a developer might ask. The answers must be found directly in the text. Provide the output as a valid JSON list of objects, where each object has a "question" and "answer" key.

--- Text ---
{document_text}

--- JSON Output ---
"""


def generate_qa_pairs(model, tokenizer, text_chunk):
    cleaned_text = re.sub(r"\s\s+", " ", text_chunk).strip()
    if len(cleaned_text) < 100:
        return []
    prompt = QA_GENERATION_PROMPT.format(document_text=cleaned_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Corrected regex to find a JSON list, and added error handling for parsing.
        json_match = re.search(r"(\[.*?\])", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                return []  # Return empty if the extracted part is not valid JSON
        return []
    except Exception:
        return []


def create_qa_training_data():
    print("Starting QA-based data preparation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_files = []
    print(f"Searching for files in: {os.path.abspath(SOURCE_PATH)}")
    for pattern in FILE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(SOURCE_PATH, pattern), recursive=True))

    print(f"Found {len(all_files)} total files to process.")

    files_to_process = all_files
    print(f"Processing {len(files_to_process)} files for development.")

    training_prompt_format = "Question: {question}\nAnswer: {answer}"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for i, file_path in enumerate(files_to_process):
            print(f"Processing file {i+1}/{len(files_to_process)}: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as md_file:
                    content = md_file.read()
                    qa_pairs_generated = 0
                    for chunk_index, j in enumerate(range(0, len(content), 2000)):
                        print(f"  - Processing chunk {chunk_index + 1}...")
                        chunk = content[j : j + 2000]
                        qa_pairs = generate_qa_pairs(model, tokenizer, chunk)
                        for pair in qa_pairs:
                            if "question" in pair and "answer" in pair:
                                f.write(
                                    json.dumps(
                                        {"text": training_prompt_format.format(**pair)}
                                    )
                                    + "\n"
                                )
                                qa_pairs_generated += 1
                    print(f"  - Generated {qa_pairs_generated} QA pairs for this file.")
            except Exception as e:
                print(f"  - Error processing file {file_path}: {e}")
                continue

    print(
        f"\n\033[92mSuccessfully created Q&A training data at: {os.path.abspath(OUTPUT_FILE)}\033[0m"
    )


if __name__ == "__main__":
    create_qa_training_data()
