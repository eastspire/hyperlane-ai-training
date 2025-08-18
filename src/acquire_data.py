import os
import subprocess
import requests
import shutil
import sys

# --- Configuration ---
# The main directory to hold all our training source files
SOURCES_DIR = "../training_sources"
# The original docs repository
PRIMARY_REPO = {"url": "https://github.com/eastspire/ltpp-docs.git", "name": "ltpp-docs"}
# The organizations from which to clone all public repositories
ORGS_TO_CLONE = ["crates-dev", "hyperlane-dev"]

def run_command(command, cwd="."):
    """Runs a command and checks for errors."""
    print(f"\n--- Running command: {' '.join(command)} in {cwd} ---")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"--- ERROR ---\n{stderr}")
        return False
    print("--- SUCCESS ---")
    return True

def get_repo_urls_from_org(org_name):
    """Fetches all public repository clone URLs for a given GitHub organization."""
    print(f"Fetching repository list for organization: {org_name}...")
    repo_urls = []
    page = 1
    while True:
        api_url = f"https://api.github.com/orgs/{org_name}/repos?type=public&page={page}&per_page=100"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if not data or not isinstance(data, list):
                break
            for repo in data:
                if isinstance(repo, dict) and 'clone_url' in repo:
                    repo_urls.append(repo['clone_url'])
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repos for {org_name}: {e}")
            break
    print(f"Found {len(repo_urls)} repositories for {org_name}.")
    return repo_urls

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

    # 2. Clone all repos from the specified organizations into subdirectories
    for org in ORGS_TO_CLONE:
        org_dir = os.path.join(SOURCES_DIR, org)
        if not os.path.isdir(org_dir):
            os.makedirs(org_dir)
        
        repo_urls = get_repo_urls_from_org(org)
        print(f"\nStarting to process {len(repo_urls)} repositories for {org}...")
        for repo_url in repo_urls:
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            repo_path = os.path.join(org_dir, repo_name)
            if os.path.exists(repo_path):
                print(f"Repository {repo_name} already exists. Skipping clone.")
            else:
                print(f"Cloning {repo_name} into {org} folder...")
                run_command(["git", "clone", repo_url, repo_path])

    print("\n\033[92mData acquisition complete! All repositories are organized in the '{SOURCES_DIR}' directory.\033[0m")

if __name__ == "__main__":
    main()