import subprocess
import sys
import os

def run_step(script_name):
    """Runs a python script as a step and exits if it fails."""
    print(f"\n{'='*60}")
    print(f"--- Running Step: {script_name} ---")
    print(f"{ '='*60}\n")
    python_executable = sys.executable
    process = subprocess.Popen(
        [python_executable, script_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    for line in iter(process.stdout.readline, ''):
        print(line, end='')
    process.wait()
    if process.returncode != 0:
        print(f"\n--- STEP FAILED: {script_name} ---", file=sys.stderr)
        sys.exit(f"Script '{script_name}' failed with exit code {process.returncode}")
    print(f"\n--- Step Succeeded: {script_name} ---")

def main():
    """Runs the entire training and conversion pipeline."""
    pipeline_scripts = [
        "src/acquire_data.py",
        "src/prepare_data.py",
        "src/train.py",
        "src/merge_and_export.py",
        "src/gguf_converter.py"
    ]
    
    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"Error: Required script '{script}' not found.", file=sys.stderr)
            sys.exit(1)
            
    for script in pipeline_scripts:
        run_step(script)
        
    print(f"\n\n{'='*60}")
    print("--- ENTIRE PIPELINE COMPLETED SUCCESSFULLY! ---")
    print("Your final GGUF model is ready.")
    print(f"{ '='*60}\n")

if __name__ == "__main__":
    main()
