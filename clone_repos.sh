#!/bin/bash

# 定义用户名作为中间目录
user_dir="source/"

# 创建用户名目录
mkdir -p "$user_dir"

# 克隆 hyperlane-dev 组织下的所有仓库
echo "克隆 hyperlane-dev 组织下的仓库..."
curl -s https://api.github.com/orgs/hyperlane-dev/repos?per_page=100 |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "//g' |
  sed 's/"//g' |
  while read repo
  do
    repo_name=$(basename "$repo" .git)
    if git clone "$repo" "$user_dir/$repo_name"; then
      echo "成功克隆 $repo_name"
    else
      echo "克隆 $repo_name 失败"
    fi
    sleep 5
  done

# 克隆 crates-dev 组织下的所有仓库
echo "克隆 crates-dev 组织下的仓库..."
curl -s https://api.github.com/orgs/crates-dev/repos?per_page=100 |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "//g' |
  sed 's/"//g' |
  while read repo
  do
    repo_name=$(basename "$repo" .git)
    if git clone "$repo" "$user_dir/$repo_name"; then
      echo "成功克隆 $repo_name"
    else
      echo "克隆 $repo_name 失败"
    fi
    sleep 5
  done

# 克隆 eastspire 组织下的所有仓库
echo "克隆 eastspire 组织下的仓库..."
curl -s https://api.github.com/orgs/eastspire/repos?per_page=100 |
  grep -o '"clone_url": "[^"]*"' |
  sed 's/"clone_url": "//g' |
  sed 's/"//g' |
  while read repo
  do
    repo_name=$(basename "$repo" .git)
    if git clone "$repo" "$user_dir/$repo_name"; then
      echo "成功克隆 $repo_name"
    else
      echo "克隆 $repo_name 失败"
    fi
    sleep 5
  done

echo "所有仓库克隆完成！"