#!/bin/bash

# 定义用户名作为中间目录
user_dir="source"

# 创建用户名目录
mkdir -p "$user_dir"

# 函数：克隆或拉取仓库
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

# 克隆 hyperlane-dev 组织下的所有仓库
echo "克隆 hyperlane-dev 组织下的仓库..."
curl -s "https://api.github.com/orgs/hyperlane-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

# 克隆 crates-dev 组织下的所有仓库
echo "克隆 crates-dev 组织下的仓库..."
curl -s "https://api.github.com/orgs/crates-dev/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

# 克隆 eastspire 组织下的所有仓库
echo "克隆 eastspire 组织下的仓库..."
curl -s "https://api.github.com/orgs/eastspire/repos?per_page=100" |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "\(.*\)"/\1/' |
  while read -r repo; do
    repo_name=$(basename "$repo" .git)
    clone_or_pull "$repo" "$repo_name"
  done

# 单独克隆 ltpp-docs（可选）
echo "克隆 eastspire/ltpp-docs..."
clone_or_pull "https://github.com/eastspire/ltpp-docs" "ltpp-docs"

echo "所有仓库克隆/更新完成！"