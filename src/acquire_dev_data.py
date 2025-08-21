import os
import subprocess
import requests
import shutil

# --- Configuration ---
# The main directory to hold all our training source files
SOURCES_DIR = "./training_sources"
# The original docs repository
PRIMARY_REPO = {
    "url": "https://github.com/eastspire/ltpp-docs.git",
    "name": "ltpp-docs",
}


def run_command(command, cwd="."):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: {' '.join(command)} in {cwd} ---")
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
    if process.returncode != 0:
        print(f"--- ERROR ---\n{stderr}")
        return False
    print("--- SUCCESS ---")
    return True


def main():
    """Clones all necessary repositories into a structured directory."""
    if not os.path.isdir(SOURCES_DIR):
        os.makedirs(SOURCES_DIR)

    # 1. Clone the primary repository directly into the sources dir
    repo_path = os.path.join(SOURCES_DIR, PRIMARY_REPO["name"])
    if os.path.exists(repo_path):
        print(f"Pulling latest changes for {PRIMARY_REPO['name']}...")
        run_command(["git", "pull"], cwd=repo_path)
    else:
        print(f"Cloning {PRIMARY_REPO['name']}...")
        run_command(["git", "clone", PRIMARY_REPO["url"], repo_path])

    # 2. Copy the README.md file
    readme_src_path = "./README.md"
    readme_dest_dir = os.path.join(SOURCES_DIR, "readme")
    if not os.path.isdir(readme_dest_dir):
        os.makedirs(readme_dest_dir)
    readme_dest_path = os.path.join(readme_dest_dir, "README.md")
    print(f"Copying {readme_src_path} to {readme_dest_path}...")
    shutil.copy(readme_src_path, readme_dest_path)
    print("--- SUCCESS ---")


    print(
        "\n\033[92mData acquisition complete! All repositories are organized in the '{SOURCES_DIR}' directory.\033[0m"
    )


if __name__ == "__main__":
    main()
