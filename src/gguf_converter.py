import os
import subprocess
import sys
import time

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


def run_command(command, cwd=None, capture_output=False):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: `{' '.join(command)}` in `{cwd or os.getcwd()}` ---")
    try:
        if capture_output:
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
        else:
            # For real-time output
            process = subprocess.Popen(
                command,
                cwd=cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            stdout, stderr = process.communicate()

        if capture_output:
            print(stdout)
            if stderr:
                print("STDERR:", stderr)

        if process.returncode != 0:
            print(f"--- ERROR ---")
            if capture_output and stderr:
                print(stderr)
            return False, stdout, stderr

        print("--- SUCCESS ---")
        return True, stdout, stderr

    except FileNotFoundError:
        print("--- ERROR ---")
        print(
            f"Command not found: {command[0]}. Make sure it is installed and in your PATH."
        )
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        print("--- AN UNEXPECTED ERROR OCCURRED ---")
        print(str(e))
        return False, "", str(e)


def ensure_llama_cpp():
    """Clone llama.cpp if it doesn't exist."""
    if not os.path.isdir(LLAMA_CPP_PATH):
        print(f"llama.cpp not found at '{LLAMA_CPP_PATH}'. Cloning repository...")
        success, _, _ = run_command(
            ["git", "clone", GIT_REPO, LLAMA_CPP_PATH], cwd=PROJECT_ROOT
        )
        if not success:
            sys.exit(1)
    else:
        print(f"Found existing llama.cpp at: {LLAMA_CPP_PATH}")


def check_model_files():
    """Check if the model files exist and are valid."""
    print(f"\n--- Checking model files in: {MODEL_TO_CONVERT_PATH} ---")

    if not os.path.isdir(MODEL_TO_CONVERT_PATH):
        print(f"Error: Model directory not found: {MODEL_TO_CONVERT_PATH}")
        return False

    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]

    # Check required files
    for file in required_files:
        file_path = os.path.join(MODEL_TO_CONVERT_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ Found {file} ({size} bytes)")
        else:
            print(f"✗ Missing required file: {file}")
            return False

    # Check for model files
    model_file_found = False
    for file in model_files:
        file_path = os.path.join(MODEL_TO_CONVERT_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ Found {file} ({size} bytes)")
            model_file_found = True
            break

    if not model_file_found:
        print(
            "✗ No model files found (looking for model.safetensors or pytorch_model.bin)"
        )
        return False

    print("--- All required files found ---")
    return True


def find_conversion_script():
    """Find the correct conversion script."""
    possible_scripts = ["convert_hf_to_gguf.py", "convert.py", "convert-hf-to-gguf.py"]

    for script in possible_scripts:
        script_path = os.path.join(LLAMA_CPP_PATH, script)
        if os.path.exists(script_path):
            print(f"Found conversion script: {script}")
            return script_path

    print("Error: No conversion script found. Available files in llama.cpp:")
    try:
        for file in os.listdir(LLAMA_CPP_PATH):
            if file.endswith(".py") and "convert" in file.lower():
                print(f"  - {file}")
    except:
        print("  Could not list files")

    return None


def check_output_directory():
    """Ensure output directory exists and is writable."""
    output_dir = os.path.dirname(OUTPUT_GGUF_FILE)
    print(f"\n--- Checking output directory: {output_dir} ---")

    try:
        os.makedirs(output_dir, exist_ok=True)

        # Test write permissions
        test_file = os.path.join(output_dir, "test_write.tmp")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)

        print(f"✓ Output directory is ready and writable")
        return True
    except Exception as e:
        print(f"✗ Error with output directory: {e}")
        return False


def main():
    print("=== GGUF Conversion Script ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_TO_CONVERT_PATH}")
    print(f"Output file: {OUTPUT_GGUF_FILE}")

    # Step 1: Ensure llama.cpp exists
    ensure_llama_cpp()

    # Step 2: Check model files
    if not check_model_files():
        print("Please run 'merge_and_export.py' first to create the merged model.")
        sys.exit(1)

    # Step 3: Check output directory
    if not check_output_directory():
        sys.exit(1)

    # Step 4: Find conversion script
    convert_script_path = find_conversion_script()
    if not convert_script_path:
        sys.exit(1)

    # Step 5: Install dependencies
    python_executable = sys.executable
    print(f"\n--- Installing dependencies using: {python_executable} ---")
    success, _, _ = run_command(
        [
            python_executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "numpy",
            "sentencepiece",
            "torch",
            "gguf",
            "transformers",
        ],
        LLAMA_CPP_PATH,
        capture_output=True,
    )
    if not success:
        print("Warning: Failed to install some dependencies, but continuing...")

    # Step 6: Remove existing output file if it exists
    if os.path.exists(OUTPUT_GGUF_FILE):
        print(f"Removing existing output file: {OUTPUT_GGUF_FILE}")
        os.remove(OUTPUT_GGUF_FILE)

    # Step 7: Run conversion
    print(f"\n--- Starting GGUF conversion ---")
    print(f"Script: {convert_script_path}")
    print(f"Model: {MODEL_TO_CONVERT_PATH}")
    print(f"Output: {OUTPUT_GGUF_FILE}")

    conversion_command = [
        python_executable,
        convert_script_path,
        MODEL_TO_CONVERT_PATH,
        "--outfile",
        OUTPUT_GGUF_FILE,
    ]

    # Run with real-time output
    success, stdout, stderr = run_command(conversion_command, LLAMA_CPP_PATH)

    # Step 8: Verify output
    print(f"\n--- Verifying conversion results ---")

    if os.path.exists(OUTPUT_GGUF_FILE):
        file_size = os.path.getsize(OUTPUT_GGUF_FILE)
        print(f"✓ GGUF file created successfully!")
        print(f"  Path: {OUTPUT_GGUF_FILE}")
        print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

        # Check if file size is reasonable (should be > 1MB for a real model)
        if file_size < 1024 * 1024:  # Less than 1MB
            print("⚠ Warning: File size seems too small for a model")

    else:
        print(f"✗ GGUF file was not created!")
        print(f"Expected location: {OUTPUT_GGUF_FILE}")

        # Check if file was created elsewhere
        possible_locations = [
            os.path.join(LLAMA_CPP_PATH, "hyperlane-qwen2.5-coder-1.5b-instruct.gguf"),
            os.path.join(
                MODEL_TO_CONVERT_PATH, "hyperlane-qwen2.5-coder-1.5b-instruct.gguf"
            ),
            os.path.join(os.getcwd(), "hyperlane-qwen2.5-coder-1.5b-instruct.gguf"),
        ]

        print("Checking alternative locations:")
        for location in possible_locations:
            if os.path.exists(location):
                size = os.path.getsize(location)
                print(f"  ✓ Found at: {location} ({size:,} bytes)")
            else:
                print(f"  ✗ Not at: {location}")

        sys.exit(1)

    print(f"\n\033[92m=== Conversion Complete! ===\033[0m")
    print(f"Your GGUF model is ready at: {OUTPUT_GGUF_FILE}")


if __name__ == "__main__":
    main()
