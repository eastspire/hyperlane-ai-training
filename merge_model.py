import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Constants
BASE_MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
ADAPTER_MODEL_DIR = "deepseek-coder-1.3b-instruct"
MERGED_MODEL_DIR = "deepseek-coder-1.3b-instruct-merged"

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
