import os
import subprocess
import sys

# --- Configuration ---
# Path to the llama.cpp repository, assuming it's inside the current project directory
LLAMA_CPP_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "llama.cpp")
)
MODEL_TO_CONVERT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "hyperlane-qwen2-merged-model")
)
OUTPUT_GGUF_FILE = os.path.join(
    "outputs", "gguf", "hyperlane-qwen2.5-coder-1.5b-instruct.gguf"
)


def run_command(command, cwd):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: `{' '.join(command)}` in `{cwd}` ---")
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate()

        print(stdout)
        if process.returncode != 0:
            print(f"--- ERROR ---")
            print(stderr)
            sys.exit(f"Command failed with exit code {process.returncode}")
        print("--- SUCCESS ---")
    except FileNotFoundError:
        print("--- ERROR ---")
        print(
            f"Command not found: {command[0]}. Make sure it is installed and in your PATH."
        )
        sys.exit(1)
    except Exception as e:
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        print(str(e))
        sys.exit(1)


def main():
    # Check if the required directories exist
    if not os.path.isdir(MODEL_TO_CONVERT_PATH):
        print(f"Error: Merged model directory not found at '{MODEL_TO_CONVERT_PATH}'")
        print("Please run 'python merge_and_export.py' first.")
        sys.exit(1)

    if not os.path.isdir(LLAMA_CPP_PATH):
        print(f"Error: llama.cpp directory not found at '{LLAMA_CPP_PATH}'")
        print(
            "Please ensure the llama.cpp folder is inside the 'ai-training' directory."
        )
        sys.exit(1)

    print(f"Found llama.cpp at: {LLAMA_CPP_PATH}")

    python_executable = sys.executable
    print("Installing/checking minimal dependencies for GGUF conversion...")
    run_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "numpy",
            "sentencepiece",
            "protobuf",
            "mistral_common",
        ],
        LLAMA_CPP_PATH,
    )

    convert_script_path = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    # Output the file directly into the ai-training directory for easier access
    output_file_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", OUTPUT_GGUF_FILE)
    )
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    python_executable = sys.executable

    print("Starting GGUF conversion...")
    conversion_command = [
        python_executable,
        convert_script_path,
        MODEL_TO_CONVERT_PATH,
        "--outfile",
        output_file_path,
    ]
    run_command(conversion_command, LLAMA_CPP_PATH)

    print(f"\n\n\033[92mConversion complete! Your GGUF file is ready at:\033[0m")
    print(output_file_path)


if __name__ == "__main__":
    main()
