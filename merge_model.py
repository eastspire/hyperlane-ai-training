import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration from environment variables
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME")
ADAPTER_MODEL_DIR = os.getenv("ADAPTER_MODEL_DIR")
MERGED_MODEL_DIR = os.getenv("MERGED_MODEL_DIR")

# Load base model and tokenizer
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

# Load PEFT model and merge
model = PeftModel.from_pretrained(base_model, ADAPTER_MODEL_DIR)
model = model.merge_and_unload()

# Save merged model and tokenizer
model.save_pretrained(MERGED_MODEL_DIR)
tokenizer.save_pretrained(MERGED_MODEL_DIR)

print(f"Model merged and saved to {MERGED_MODEL_DIR}")
