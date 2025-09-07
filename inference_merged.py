import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from dotenv import load_dotenv
import traceback

# Load environment variables
load_dotenv()

# Get configuration from environment variables
MERGED_MODEL_DIR = os.getenv("MERGED_MODEL_DIR")


def load_model():
    """Load merged model"""
    try:
        print(f"Loading merged model from {MERGED_MODEL_DIR}")
        # Load merged model
        model = AutoModelForCausalLM.from_pretrained(
            MERGED_MODEL_DIR,
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
                else torch.float32
            ),
            device_map="auto",
            trust_remote_code=True,
        )
        print("Merged model loaded successfully")

        # Load tokenizer
        print(f"Loading tokenizer from {MERGED_MODEL_DIR}")
        tokenizer = AutoTokenizer.from_pretrained(
            MERGED_MODEL_DIR, trust_remote_code=True
        )
        print("Tokenizer loaded successfully")

        # Set pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return None, None


def generate_response(model, tokenizer, prompt):
    """Generate model response"""
    try:
        # Prepare input
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=500,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        traceback.print_exc()
        return ""


def main():
    """Main function"""
    # Load model and tokenizer
    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        print("Failed to load model or tokenizer")
        return

    # Set prompt template
    alpaca_prompt = """<|im_start|>system
{}<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""

    # Question
    question = "Who are you?"
    system_prompt = (
        "You are an AI assistant for the Hyperlane Web framework written in Rust."
    )

    # Build complete prompt
    prompt = alpaca_prompt.format(system_prompt, question)

    # Generate response
    print(f"Question: {question}")
    response = generate_response(model, tokenizer, prompt)
    print(f"Answer: {response}")


if __name__ == "__main__":
    main()
