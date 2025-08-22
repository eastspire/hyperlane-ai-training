import os
import subprocess
import sys

# --- Configuration ---
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LLAMA_CPP_PATH = os.path.join(PROJECT_ROOT, "llama.cpp")
MODEL_TO_CONVERT_PATH = os.path.join(
    PROJECT_ROOT, "hyperlane-qwen2.5-coder-1.5b-merged"
)
OUTPUT_GGUF_FILE = os.path.join(
    PROJECT_ROOT, "outputs", "gguf", "hyperlane-qwen2.5-coder-1.5b-instruct.gguf"
)
GIT_REPO = "https://github.com/ggml-org/llama.cpp.git"


def run_command(command, cwd=None):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: `{' '.join(command)}` in `{cwd or os.getcwd()}` ---")
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


def ensure_llama_cpp():
    """Clone llama.cpp if it doesn't exist."""
    if not os.path.isdir(LLAMA_CPP_PATH):
        print(f"llama.cpp not found at '{LLAMA_CPP_PATH}'. Cloning repository...")
        run_command(["git", "clone", GIT_REPO, LLAMA_CPP_PATH], cwd=PROJECT_ROOT)
    else:
        print(f"Found existing llama.cpp at: {LLAMA_CPP_PATH}")


def main():
    ensure_llama_cpp()

    if not os.path.isdir(MODEL_TO_CONVERT_PATH):
        print(f"Error: Merged model directory not found at '{MODEL_TO_CONVERT_PATH}'")
        print("Please run 'merge_and_export.py' first.")
        sys.exit(1)

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
            "torch",
            "gguf",
        ],
        LLAMA_CPP_PATH,
    )

    convert_script_path = os.path.join(LLAMA_CPP_PATH, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script_path):
        print(
            f"Error: The conversion script 'convert_hf_to_gguf.py' was not found in '{LLAMA_CPP_PATH}'."
        )
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_GGUF_FILE), exist_ok=True)

    print("Starting GGUF conversion...")
    conversion_command = [
        python_executable,
        convert_script_path,
        MODEL_TO_CONVERT_PATH,
        "--outfile",
        OUTPUT_GGUF_FILE,
    ]
    run_command(conversion_command, LLAMA_CPP_PATH)

    print(f"\n\n\033[92mConversion complete! Your GGUF file is ready at:\033[0m")
    print(OUTPUT_GGUF_FILE)


if __name__ == "__main__":
    main()
