import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME")
OUTPUT_DIR = os.getenv("OUTPUT_DIR")
MERGED_MODEL_DIR = os.getenv("MERGED_MODEL_DIR")

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load PEFT model and merge
model = PeftModel.from_pretrained(base_model, OUTPUT_DIR, trust_remote_code=True)
model = model.merge_and_unload()

# Save merged model and tokenizer
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"Model merged and saved to {MERGED_MODEL_DIR}")
