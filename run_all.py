import subprocess
import sys
import os
import shutil


def run_step(script_name, logs_dir):
    """Runs a python script as a step and exits if it fails."""
    print(f"\n{'='*60}")
    print(f"--- Running Step: {script_name} ---")
    print(f"{ '='*60}\n")
    python_executable = sys.executable
    log_file_path = os.path.join(
        logs_dir, f"{os.path.basename(script_name).replace('.py', '.log')}"
    )
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [python_executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            log_file.write(line)
        process.wait()
    if process.returncode != 0:
        print(f"\n--- STEP FAILED: {script_name} ---", file=sys.stderr)
        print(f"See log file for details: {log_file_path}", file=sys.stderr)
        sys.exit(f"Script '{script_name}' failed with exit code {process.returncode}")
    print(f"\n--- Step Succeeded: {script_name} ---")


def main():
    """Runs the entire training and conversion pipeline."""
    # --- Cleanup old files ---
    print("--- Deleting old training files and outputs ---")

    # Delete the outputs directory
    if os.path.isdir("outputs"):
        shutil.rmtree("outputs")
        print("Deleted directory: outputs")

    # Delete the logs directory
    if os.path.isdir("logs"):
        shutil.rmtree("logs")
        print("Deleted directory: logs")

    # Delete the training_data.jsonl file
    training_data_path = "training_data.jsonl"
    if os.path.exists(training_data_path):
        os.remove(training_data_path)
        print(f"Deleted file: {training_data_path}")

    print("--- Cleanup complete ---")

    logs_dir = "logs"
    if not os.path.isdir(logs_dir):
        os.makedirs(logs_dir)

    pipeline_scripts = [
        "src/acquire_data.py",
        "src/prepare_data.py",
        "src/train.py",
        "src/merge_and_export.py",
        "src/gguf_converter.py",
    ]

    for script in pipeline_scripts:
        if not os.path.exists(script):
            print(f"Error: Required script '{script}' not found.", file=sys.stderr)
            sys.exit(1)

    for script in pipeline_scripts:
        run_step(script, logs_dir)

    print(f"\n\n{'='*60}")
    print("--- ENTIRE PIPELINE COMPLETED SUCCESSFULLY! ---")
    print("Your final GGUF model is ready.")
    print(f"{ '='*60}\n")


if __name__ == "__main__":
    main()
