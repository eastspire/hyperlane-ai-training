import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
BASE_MODEL_NAME = "qwen/qwen2-1.5b"
ADAPTER_MODEL_DIR = "qwen3-4b"
MERGED_MODEL_DIR = "qwen3-4b-merged"

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
