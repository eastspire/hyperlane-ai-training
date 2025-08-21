import subprocess
import sys
import os
import shutil


def run_step(script_name, logs_dir):
    """Runs a python script as a step and exits if it fails."""
    print(f"\n{'='*60}")
    print(f"--- Running Step: {script_name} ---")
    log_file_path = os.path.join(
        logs_dir, f"{os.path.basename(script_name).replace('.py', '.log')}"
    )
    print(f"--- Log file: {log_file_path} ---")
    print(f"{ '='*60}\n")
    python_executable = sys.executable
    with open(log_file_path, "w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [python_executable, script_name],
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
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
    pid_file = "run_all.pid"

    # --- Handle STOP command ---
    if len(sys.argv) > 1 and sys.argv[1].lower() == "stop":
        if not os.path.exists(pid_file):
            print("PID file not found. Nothing to stop.")
            sys.exit(0)

        try:
            with open(pid_file, "r") as f:
                pid = int(f.read().strip())
        except (IOError, ValueError) as e:
            print(f"Error reading PID file: {e}")
            sys.exit(1)

        print(f"--- Attempting to stop process with PID: {pid} ---")
        try:
            # Use taskkill on Windows to forcefully terminate the process
            result = subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                print(f"--- Process {pid} stopped successfully. ---")
            else:
                if "not found" in result.stderr.lower():
                    print(
                        f"--- Process {pid} was not found. It may have already finished. ---"
                    )
                else:
                    print(f"--- Failed to stop process {pid}. ---")
                    print(f"taskkill stderr: {result.stderr.strip()}")

        except FileNotFoundError:
            print("`taskkill` command not found. Are you on Windows?")
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            if os.path.exists(pid_file):
                os.remove(pid_file)
        sys.exit(0)

    # --- Handle START command (default) ---
    if os.name == "nt" and os.environ.get("_IS_BACKGROUND_PROCESS") != "1":
        if os.path.exists(pid_file):
            print("PID file exists. A process might already be running.")
            print(
                "If you are sure no process is running, delete 'run_all.pid' and try again."
            )
            print("To stop a running process, use: python run_all.py stop")
            sys.exit(1)

        print("--- Launching the pipeline in the background. ---")
        env = os.environ.copy()
        env["_IS_BACKGROUND_PROCESS"] = "1"

        p = subprocess.Popen(
            [sys.executable, __file__],
            env=env,
            creationflags=0x08000008,  # DETACHED_PROCESS | CREATE_NO_WINDOW
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )

        try:
            with open(pid_file, "w") as f:
                f.write(str(p.pid))
        except IOError as e:
            print(f"Fatal: Could not write PID file: {e}")
            p.kill()  # Kill the process we just started
            sys.exit(1)

        print(f"--- Pipeline is now running in the background with PID: {p.pid}. ---")
        print("--- To stop it, run: python run_all.py stop ---")

    else:
        # This is the background process, or we are not on Windows.
        try:
            main()
        finally:
            # Clean up the PID file on normal completion
            if os.name == "nt" and os.environ.get("_IS_BACKGROUND_PROCESS") == "1":
                if os.path.exists(pid_file):
                    os.remove(pid_file)
