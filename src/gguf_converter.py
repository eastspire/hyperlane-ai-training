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
GIT_REPO = "https://github.com/eastspire/llama.cpp.git"


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

            # Print output for debugging
            if stdout.strip():
                print("STDOUT:", stdout)
            if stderr.strip():
                print("STDERR:", stderr)
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

        if process.returncode != 0:
            print(f"--- COMMAND FAILED (exit code: {process.returncode}) ---")
            if capture_output and stderr:
                print("Error details:", stderr)
            return False, stdout, stderr

        print("--- COMMAND SUCCESSFUL ---")
        return True, stdout, stderr

    except FileNotFoundError:
        print("--- ERROR ---")
        print(
            f"Command not found: {command[0]}. Make sure it is installed and in your PATH."
        )
        return False, "", f"Command not found: {command[0]}"
    except Exception as e:
        print("--- UNEXPECTED ERROR ---")
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
            print("Failed to clone llama.cpp repository.")
            print("You can manually clone it using:")
            print(f"  git clone {GIT_REPO} {LLAMA_CPP_PATH}")
            sys.exit(1)
        print(f"Successfully cloned llama.cpp to: {LLAMA_CPP_PATH}")
    else:
        print(f"Found existing llama.cpp at: {LLAMA_CPP_PATH}")

        # Check if it's a git repository and try to update
        git_dir = os.path.join(LLAMA_CPP_PATH, ".git")
        if os.path.exists(git_dir):
            print("Attempting to update llama.cpp to latest version...")
            success, _, _ = run_command(
                ["git", "pull"], cwd=LLAMA_CPP_PATH, capture_output=True
            )
            if success:
                print("llama.cpp updated successfully")
            else:
                print("Warning: Could not update llama.cpp, using existing version")


def check_model_files():
    """Check if the model files exist and are valid."""
    print(f"\n--- Checking model files in: {MODEL_TO_CONVERT_PATH} ---")

    if not os.path.isdir(MODEL_TO_CONVERT_PATH):
        print(f"Error: Model directory not found: {MODEL_TO_CONVERT_PATH}")
        return False

    # List all files in the model directory for debugging
    try:
        files_in_dir = os.listdir(MODEL_TO_CONVERT_PATH)
        print(f"Files in model directory: {files_in_dir}")
    except Exception as e:
        print(f"Could not list files in model directory: {e}")
        return False

    required_files = ["config.json"]
    model_files = ["model.safetensors", "pytorch_model.bin"]

    # Also check for sharded model files
    sharded_files = [
        f for f in files_in_dir if f.startswith("model-") and f.endswith(".safetensors")
    ]

    # Check required files
    for file in required_files:
        file_path = os.path.join(MODEL_TO_CONVERT_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ Found {file} ({size:,} bytes)")
        else:
            print(f"✗ Missing required file: {file}")
            return False

    # Check for model files
    model_file_found = False

    # Check single model files first
    for file in model_files:
        file_path = os.path.join(MODEL_TO_CONVERT_PATH, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✓ Found {file} ({size:,} bytes)")
            model_file_found = True
            break

    # Check for sharded model files
    if not model_file_found and sharded_files:
        total_size = 0
        for file in sharded_files:
            file_path = os.path.join(MODEL_TO_CONVERT_PATH, file)
            size = os.path.getsize(file_path)
            total_size += size
            print(f"✓ Found {file} ({size:,} bytes)")

        if sharded_files:
            print(
                f"✓ Found {len(sharded_files)} sharded model files (total: {total_size:,} bytes)"
            )
            model_file_found = True

    if not model_file_found:
        print(
            "✗ No model files found (looking for model.safetensors, pytorch_model.bin, or model-*.safetensors)"
        )
        return False

    print("--- All required files found ---")
    return True


def fix_mistral_import_issue():
    """Try to fix the mistral_common import issue."""
    print("Attempting to fix mistral_common import issue...")

    try:
        # Try to uninstall mistral_common if it's causing issues
        success, _, _ = run_command(
            [sys.executable, "-m", "pip", "uninstall", "-y", "mistral_common"],
            capture_output=True,
        )

        if success:
            print("✓ Uninstalled problematic mistral_common package")
            return True
        else:
            print("Could not uninstall mistral_common package")

    except Exception as e:
        print(f"Error trying to fix mistral_common issue: {e}")

    return False


def find_conversion_script():
    """Find the correct conversion script."""
    possible_scripts = [
        "convert_hf_to_gguf.py",
        "convert.py",
        "convert-hf-to-gguf.py",
        "convert_hf_to_gguf_update.py",
    ]

    for script in possible_scripts:
        script_path = os.path.join(LLAMA_CPP_PATH, script)
        if os.path.exists(script_path):
            print(f"Found conversion script: {script}")
            return script_path

    print("Error: No conversion script found. Available Python files in llama.cpp:")
    try:
        python_files = []
        for file in os.listdir(LLAMA_CPP_PATH):
            if file.endswith(".py"):
                python_files.append(file)

        if python_files:
            for file in sorted(python_files):
                print(f"  - {file}")
        else:
            print("  No Python files found")

        # Check if there are any convert scripts with different names
        convert_files = [f for f in python_files if "convert" in f.lower()]
        if convert_files:
            print(f"\nConversion-related scripts found:")
            for file in convert_files:
                print(f"  - {file}")

    except Exception as e:
        print(f"  Could not list files: {e}")

    return None


def check_python_dependencies():
    """Check if required Python packages are available."""
    print(f"\n--- Checking Python dependencies ---")

    required_packages = {
        "numpy": "numpy",
        "torch": "torch",
        "transformers": "transformers",
        "gguf": "gguf",
    }

    # Check for problematic packages that might cause issues
    problematic_packages = {"mistral_common": "mistral_common"}

    missing_packages = []

    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"✓ {package_name} is available")
        except ImportError:
            print(f"✗ {package_name} is missing")
            missing_packages.append(package_name)

    # Check for problematic packages
    for package_name, import_name in problematic_packages.items():
        try:
            module = __import__(import_name)
            version = getattr(module, "__version__", "unknown")
            print(
                f"⚠ {package_name} is installed (version: {version}) - may cause import issues"
            )
        except ImportError:
            print(f"✓ {package_name} not installed (good - avoids import conflicts)")

    if missing_packages:
        print(f"\n⚠ Warning: Missing packages: {', '.join(missing_packages)}")
        print("Please install them using:")
        print(f"  pip install {' '.join(missing_packages)}")

        user_input = input("\nDo you want to continue anyway? (y/N): ").lower().strip()
        if user_input != "y" and user_input != "yes":
            print("Aborting conversion.")
            return False
    else:
        print("✓ All required packages are available")

    return True


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


def estimate_conversion_time(model_path):
    """Estimate conversion time based on model size."""
    try:
        total_size = 0
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file.endswith((".safetensors", ".bin")):
                    file_path = os.path.join(root, file)
                    total_size += os.path.getsize(file_path)

        size_gb = total_size / (1024**3)
        # Rough estimate: ~1-2 minutes per GB
        estimated_minutes = size_gb * 1.5

        print(f"Model size: {size_gb:.2f} GB")
        print(f"Estimated conversion time: {estimated_minutes:.1f} minutes")

    except Exception as e:
        print(f"Could not estimate conversion time: {e}")


def main():
    print("=== GGUF Conversion Script ===")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model path: {MODEL_TO_CONVERT_PATH}")
    print(f"Output file: {OUTPUT_GGUF_FILE}")

    # Step 1: Ensure llama.cpp exists
    ensure_llama_cpp()

    # Step 2: Check model files
    if not check_model_files():
        print("\nPlease ensure the merged model exists at the specified path.")
        print("You may need to run the merge script first.")
        sys.exit(1)

    # Step 3: Check Python dependencies
    if not check_python_dependencies():
        sys.exit(1)

    # Step 4: Check output directory
    if not check_output_directory():
        sys.exit(1)

    # Step 5: Find conversion script
    convert_script_path = find_conversion_script()
    if not convert_script_path:
        print("\nPlease check your llama.cpp installation and try again.")
        sys.exit(1)

    # Step 6: Estimate conversion time
    estimate_conversion_time(MODEL_TO_CONVERT_PATH)

    # Step 7: Remove existing output file if it exists
    if os.path.exists(OUTPUT_GGUF_FILE):
        print(f"Removing existing output file: {OUTPUT_GGUF_FILE}")
        try:
            os.remove(OUTPUT_GGUF_FILE)
        except Exception as e:
            print(f"Warning: Could not remove existing file: {e}")

    # Step 8: Try alternative conversion approaches if main script fails
    print(f"\n--- Starting GGUF conversion ---")
    print(f"Script: {convert_script_path}")
    print(f"Model: {MODEL_TO_CONVERT_PATH}")
    print(f"Output: {OUTPUT_GGUF_FILE}")
    print("This may take several minutes depending on model size...")

    python_executable = sys.executable
    conversion_success = False

    # Try different conversion approaches
    conversion_attempts = [
        {
            "name": "Standard conversion with f16",
            "command": [
                python_executable,
                convert_script_path,
                MODEL_TO_CONVERT_PATH,
                "--outfile",
                OUTPUT_GGUF_FILE,
                "--outtype",
                "f16",
            ],
        },
        {
            "name": "Conversion without outtype parameter",
            "command": [
                python_executable,
                convert_script_path,
                MODEL_TO_CONVERT_PATH,
                "--outfile",
                OUTPUT_GGUF_FILE,
            ],
        },
        {
            "name": "Conversion with q8_0 quantization",
            "command": [
                python_executable,
                convert_script_path,
                MODEL_TO_CONVERT_PATH,
                "--outfile",
                OUTPUT_GGUF_FILE,
                "--outtype",
                "q8_0",
            ],
        },
    ]

    for attempt in conversion_attempts:
        print(f"\n--- Attempting: {attempt['name']} ---")

        start_time = time.time()
        success, stdout, stderr = run_command(attempt["command"], LLAMA_CPP_PATH)
        end_time = time.time()

        conversion_time = end_time - start_time
        print(
            f"Attempt took: {conversion_time:.1f} seconds ({conversion_time/60:.1f} minutes)"
        )

        if success and os.path.exists(OUTPUT_GGUF_FILE):
            conversion_success = True
            print(f"✓ Conversion successful with: {attempt['name']}")
            break
        else:
            print(f"✗ Failed: {attempt['name']}")
            if (
                "mistral_common" in str(stderr).lower()
                or "tokenizerversion" in str(stderr).lower()
            ):
                print("Detected mistral_common import issue, trying to fix...")
                if fix_mistral_import_issue():
                    print("Retrying after fixing mistral_common...")
                    start_time = time.time()
                    success, stdout, stderr = run_command(
                        attempt["command"], LLAMA_CPP_PATH
                    )
                    end_time = time.time()
                    conversion_time = end_time - start_time

                    if success and os.path.exists(OUTPUT_GGUF_FILE):
                        conversion_success = True
                        print(f"✓ Conversion successful after fixing mistral_common")
                        break

            # Clean up any partial files
            if os.path.exists(OUTPUT_GGUF_FILE):
                try:
                    os.remove(OUTPUT_GGUF_FILE)
                    print("Cleaned up partial output file")
                except:
                    pass

    if not conversion_success:
        print("\n--- All conversion attempts failed ---")
        print("Common solutions:")
        print("1. Try uninstalling mistral_common: pip uninstall mistral_common")
        print("2. Update llama.cpp: cd llama.cpp && git pull")
        print("3. Try different conversion script versions")
        print("4. Check model file format compatibility")

    # Step 9: Verify output
    print(f"\n--- Verifying conversion results ---")

    if os.path.exists(OUTPUT_GGUF_FILE):
        file_size = os.path.getsize(OUTPUT_GGUF_FILE)
        print(f"✓ GGUF file created successfully!")
        print(f"  Path: {OUTPUT_GGUF_FILE}")
        print(f"  Size: {file_size:,} bytes ({file_size / 1024 / 1024:.2f} MB)")

        # Check if file size is reasonable (should be > 10MB for a real model)
        if file_size < 10 * 1024 * 1024:  # Less than 10MB
            print("⚠ Warning: File size seems too small for a model")
        else:
            print("✓ File size looks reasonable")

        # Try to validate GGUF file format
        try:
            with open(OUTPUT_GGUF_FILE, "rb") as f:
                magic = f.read(4)
                if magic == b"GGUF":
                    print("✓ GGUF magic bytes verified")
                else:
                    print(f"⚠ Warning: Unexpected magic bytes: {magic}")
        except Exception as e:
            print(f"Could not validate GGUF format: {e}")

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
            os.path.join(LLAMA_CPP_PATH, os.path.basename(OUTPUT_GGUF_FILE)),
        ]

        print("\nChecking alternative locations:")
        file_found = False
        for location in possible_locations:
            if os.path.exists(location):
                size = os.path.getsize(location)
                print(f"  ✓ Found at: {location} ({size:,} bytes)")
                print(f"    You may need to move it to: {OUTPUT_GGUF_FILE}")
                file_found = True
            else:
                print(f"  ✗ Not at: {location}")

        if not file_found:
            print(
                "\nConversion appears to have failed. Check the error messages above."
            )
            sys.exit(1)

    print(
        f"\n\033[92m=== Conversion Complete! ===\033[0m"
        if conversion_success
        else f"\n\033[91m=== Conversion Failed ===\033[0m"
    )
    if os.path.exists(OUTPUT_GGUF_FILE) and conversion_success:
        print(f"Your GGUF model is ready at: {OUTPUT_GGUF_FILE}")
        print(f"\nNext steps:")
        print(f"1. Test the model with llama.cpp:")
        print(f"   cd {LLAMA_CPP_PATH}")
        print(f'   ./main -m {OUTPUT_GGUF_FILE} -p "Hello, world!"')
        print(f"2. Use the model in your applications")
    else:
        print("Conversion failed. Please check the error messages above.")
        print("\nTroubleshooting steps:")
        print("1. Ensure all Python dependencies are correctly installed")
        print("2. Try: pip uninstall mistral_common")
        print("3. Update llama.cpp: cd llama.cpp && git pull")
        print("4. Check model file integrity")
        sys.exit(1)


if __name__ == "__main__":
    main()
