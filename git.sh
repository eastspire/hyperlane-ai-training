#!/bin/bash

user_dir="source"

mkdir -p "$user_dir"

clone_or_pull() {
  local repo_url=$1
  local repo_name=$2
  local target_dir="$user_dir/$repo_name"

  if [ -d "$target_dir" ]; then
    echo "仓库 $repo_name 已存在，正在拉取更新..."
    cd "$target_dir" || { echo "无法进入目录: $target_dir"; return 1; }
    if git pull --rebase; then
      echo "更新成功: $repo_name"
    else
      echo "更新失败: $repo_name"
    fi
    cd - >/dev/null || exit
  else
    echo "正在克隆: $repo_name"
    if git clone "$repo_url" "$target_dir"; then
      echo "成功克隆: $repo_name"
    else
      echo "克隆失败: $repo_url"
      return 1
    fi
  fi
}

echo "克隆 hyperlane-dev 组织下的仓库..."
curl -s "https://api.github.com/orgs/hyperlane-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "克隆 crates-dev 组织下的仓库..."
curl -s "https://api.github.com/orgs/crates-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "克隆 eastspire 组织下的仓库..."
curl -s "https://api.github.com/orgs/eastspire/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

echo "克隆 eastspire/ltpp-docs..."
clone_or_pull "https://github.com/eastspire/ltpp-docs" "ltpp-docs"

echo "所有仓库克隆/更新完成！"
