#!/bin/bash

user_dir="source"

mkdir -p "$user_dir"

clone_or_pull() {
  local repo_url=$1
  local repo_name=$2
  local target_dir="$user_dir/$repo_name"

  if [ -d "$target_dir" ]; then
    echo "Repository $repo_name already exists, pulling updates..."
    cd "$target_dir" || { echo "Failed to enter directory: $target_dir"; return 1; }
    if git pull --rebase; then
      echo "Update successful: $repo_name"
    else
      echo "Update failed: $repo_name"
    fi
    cd - >/dev/null || exit
  else
    echo "Cloning: $repo_name"
    if git clone "$repo_url" "$target_dir"; then
      echo "Successfully cloned: $repo_name"
    else
      echo "Clone failed: $repo_url"
      return 1
    fi
  fi
}

echo "Cloning repositories under hyperlane-dev organization..."
curl -s "https://api.github.com/orgs/hyperlane-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "Cloning repositories under crates-dev organization..."
curl -s "https://api.github.com/orgs/crates-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "Cloning repositories under eastspire organization..."
curl -s "https://api.github.com/orgs/eastspire/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "Cloning eastspire/ltpp-docs..."
clone_or_pull "https://github.com/eastspire/ltpp-docs" "ltpp-docs"

echo "All repositories cloned/updated successfully!"
