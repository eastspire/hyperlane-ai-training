import os
import subprocess
import requests
import concurrent.futures
from functools import partial

# --- Configuration ---
# The main directory to hold all our training source files
SOURCES_DIR = "./training_sources"
# The original docs repository
PRIMARY_REPO = {
    "url": "https://github.com/eastspire/ltpp-docs.git",
    "name": "ltpp-docs",
}
# The organizations from which to clone all public repositories
ORGS_TO_CLONE = ["crates-dev", "hyperlane-dev"]
# Number of threads to use for cloning/pulling
MAX_WORKERS = 10


def run_command(command, cwd=".", quiet=False):
    """Runs a command and checks for errors."""
    if not quiet:
        print(f"--- Running: {' '.join(command)} in {cwd}")
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
        stdout, stderr = process.communicate(timeout=300)  # 5-minute timeout
        if process.returncode != 0:
            # Print error only if it fails to keep logs clean
            print(f"--- ERROR running {' '.join(command)} in {cwd} ---\n{stderr.strip()}")
            return False
        if not quiet:
            print(f"--- SUCCESS: {' '.join(command)} in {cwd}")
        return True
    except subprocess.TimeoutExpired:
        print(f"--- TIMEOUT ERROR running {' '.join(command)} in {cwd} ---")
        # process.kill() will be handled by Popen's context manager if we use it,
        # but here we ensure it's terminated.
        process.kill()
        return False
    except Exception as e:
        print(f"--- EXCEPTION in run_command: {e} ---")
        return False



def get_repo_urls_from_org(org_name):
    """Fetches all public repository clone URLs for a given GitHub organization."""
    print(f"Fetching repository list for organization: {org_name}...")
    repo_urls = []
    page = 1
    while True:
        api_url = f"https://api.github.com/orgs/{org_name}/repos?type=public&page={page}&per_page=100"
        try:
            response = requests.get(api_url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if not data or not isinstance(data, list):
                break
            for repo in data:
                if isinstance(repo, dict) and "clone_url" in repo:
                    repo_urls.append(repo["clone_url"])
            page += 1
        except requests.exceptions.RequestException as e:
            print(f"Error fetching repos for {org_name}: {e}")
            break
    print(f"Found {len(repo_urls)} repositories for {org_name}.")
    return repo_urls


def process_repo(repo_url, *, org_dir):
    """Clones or pulls a single repository."""
    repo_name = repo_url.split("/")[-1].replace(".git", "")
    repo_path = os.path.join(org_dir, repo_name)
    if os.path.exists(repo_path):
        # print(f"Pulling latest changes for {repo_name}...")
        run_command(["git", "pull"], cwd=repo_path, quiet=True)
    else:
        # print(f"Cloning {repo_name}...")
        run_command(["git", "clone", "--depth", "1", repo_url, repo_path], quiet=True)


def main():
    """Clones all necessary repositories into a structured directory."""
    if not os.path.isdir(SOURCES_DIR):
        os.makedirs(SOURCES_DIR)

    # 1. Clone the primary repository directly into the sources dir (sequentially)
    repo_path = os.path.join(SOURCES_DIR, PRIMARY_REPO["name"])
    print(f"\n--- Processing primary repository: {PRIMARY_REPO['name']} ---")
    if os.path.exists(repo_path):
        run_command(["git", "pull"], cwd=repo_path)
    else:
        run_command(["git", "clone", PRIMARY_REPO["url"], repo_path])

    # 2. Clone all repos from the specified organizations in parallel
    for org in ORGS_TO_CLONE:
        org_dir = os.path.join(SOURCES_DIR, org)
        if not os.path.isdir(org_dir):
            os.makedirs(org_dir)

        repo_urls = get_repo_urls_from_org(org)
        # Use functools.partial to create a function with org_dir already set
        task_fn = partial(process_repo, org_dir=org_dir)

        if repo_urls:
            print(
                f"\n--- Starting parallel processing for {len(repo_urls)} repositories in {org} ---"
            )
            # Use a with statement to ensure threads are cleaned up promptly.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:
                # Using map for cleaner execution and to apply the function to each item.
                # The results are implicitly handled.
                list(
                    executor.map(task_fn, repo_urls)
                )  # list() to ensure all tasks complete

    print(
        f"\n\033[92mData acquisition complete! All repositories are up-to-date in the '{SOURCES_DIR}' directory.\033[0m"
    )


if __name__ == "__main__":
    main()