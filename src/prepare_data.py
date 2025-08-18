import os
import glob
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re

# --- Configuration ---
# The script now looks in the 'training_sources' directory.
SOURCE_PATH = './training_sources'
# We now include markdown, Rust, TOML, and other common code files.
FILE_PATTERNS = ["**/*.md", "**/*.rs", "**/*.toml", "**/*.py", "**/*.js", "**/*.ts"]
OUTPUT_FILE = './training_data.jsonl'
BASE_MODEL_ID = "Qwen/Qwen2-1.5B-Instruct"

QA_GENERATION_PROMPT = """
Based on the following text from a technical document or source code file, generate exactly two high-quality, distinct question-and-answer pairs. The questions should be technical and specific, something a developer might ask. The answers must be found directly in the text. Provide the output as a valid JSON list of objects, where each object has a "question" and an "answer" key.

--- Text ---
{document_text}

--- JSON Output ---
"""

def generate_qa_pairs(model, tokenizer, text_chunk):
    cleaned_text = re.sub(r'\s\s+', ' ', text_chunk).strip()
    if len(cleaned_text) < 100: return []
    prompt = QA_GENERATION_PROMPT.format(document_text=cleaned_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    try:
        outputs = model.generate(
            **inputs, max_new_tokens=512, eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id, do_sample=True, temperature=0.7, top_p=0.9
        )
        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        json_match = re.search(r'[\[\s*\{.*\}\s*\]]', response_text, re.DOTALL)
        if json_match: return json.loads(json_match.group(0))
        return []
    except Exception: return []

def create_qa_training_data():
    print("Starting QA-based data preparation...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, torch_dtype=torch.float32, device_map="cpu", trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    all_files = []
    print(f"Searching for files in: {os.path.abspath(SOURCE_PATH)}")
    for pattern in FILE_PATTERNS:
        all_files.extend(glob.glob(os.path.join(SOURCE_PATH, pattern), recursive=True))
    
    print(f"Found {len(all_files)} total files to process.")
    training_prompt_format = "Question: {question}\nAnswer: {answer}"

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                with open(file_path, 'r', encoding='utf-8') as md_file:
                    content = md_file.read()
                    for i in range(0, len(content), 2000):
                        chunk = content[i:i+2000]
                        qa_pairs = generate_qa_pairs(model, tokenizer, chunk)
                        for pair in qa_pairs:
                            if 'question' in pair and 'answer' in pair:
                                f.write(json.dumps({"text": training_prompt_format.format(**pair)}) + '\n')
            except Exception: continue

    print(f"\n\033[92mSuccessfully created Q&A training data at: {os.path.abspath(OUTPUT_FILE)}\033[0m")

if __name__ == '__main__':
    create_qa_training_data()