# train_your_code_model.py
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
import logging

# =============================
# 配置区（根据你的环境修改）
# =============================
REPO_ROOT = "."
OUTPUT_DATASET = "full_text_dataset.json"
TRAINED_MODEL_OUTPUT = "qwen-coder-finetuned"
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"

# 二进制文件黑名单（只这些才排除）
BINARY_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".ico",
    ".svg",
    ".webp",
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".mp4",
    ".avi",
    ".mov",
    ".wmv",
    ".webm",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".rar",
    ".7z",
    ".bz2",
    ".xz",
    ".iso",
    ".dmg",
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".out",
    ".class",
    ".jar",
    ".apk",
    ".ipa",
    ".pdb",
    ".o",
    ".obj",
    ".lib",
    ".a",
    ".db",
    ".sqlite",
    ".mdb",
    ".accdb",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".pdf",
    ".psd",
    ".ai",
    ".indd",
    ".sketch",
    ".dwg",
    ".blend",
}

# 排除的目录（依赖、缓存、临时）
EXCLUDE_DIRS = {
    ".git",
    "node_modules",
    "__pycache__",
    "venv",
    ".venv",
    "env",
    "dist",
    "build",
    "coverage",
    "logs",
    "log",
    "tmp",
    "temp",
    "cache",
    ".idea",
    ".vscode",
    ".pytest_cache",
    ".mypy_cache",
    ".tox",
    "examples",
    "scripts",
    "tests",
    "test",
    "testing",
    "docs",
    "doc",
    "public",
    "assets",
    "images",
    "img",
    "uploads",
    "download",
}

# 排除的文件名
EXCLUDE_FILES = {
    "ai.py",
    "generate_dataset.py",
    "clone_repos.sh",
    "train_your_code_model.py",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    ".gitignore",
    ".dockerignore",
    ".prettierignore",
    "README.md",
    "readme.md",
    "Readme.md",
    "LICENSE",
    "license.txt",
    "requirements.txt",
    "setup.py",
    "pyproject.toml",
    "Dockerfile",
    "docker-compose.yml",
    "Makefile",
}

# 排除的通配模式
EXCLUDE_PATTERNS = [
    "*.log",
    "*.tmp",
    "*~",
    ".*.sw*",
    "#*#",
    "*.pid",
    "*.min.js",
    "*.min.css",
    "*.bundle.js",
    "*.lock",
]


# 多线程
NUM_WORKERS = None

# 语言映射
EXT_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "react",
    ".ts": "typescript",
    ".tsx": "typescript-react",
    ".html": "html",
    ".css": "css",
    ".sh": "shell",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".txt": "text",
    ".sql": "sql",
    ".tf": "terraform",
    ".ipynb": "jupyter",
    ".dockerfile": "dockerfile",
    ".xml": "xml",
    ".toml": "toml",
    ".php": "php",
}

# 训练参数
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 3
SAVE_STEPS = 50
LOG_STEPS = 10

# LoRA 参数
LORA_R = 128
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


# =============================
# 判断是否为文本文件
# =============================
def is_text_file(file_path: Path, sample_size: int = 1024) -> bool:
    """
    Checks if a file is a text file by sampling its content.

    Args:
        - `file_path`: Path to the file.
        - `sample_size`: Number of bytes to sample from the file.

    Returns:
        True if the file is likely a text file, False otherwise.
    """
    try:
        with open(file_path, "rb") as f:
            sample = f.read(sample_size)
            if not sample:
                return False
            nontext_ratio = sum(
                1 for c in sample if c < 0x20 and c not in (9, 10, 13)
            ) / len(sample)
            return nontext_ratio < 0.3
    except Exception:
        return False


# =============================
# 处理单个文件
# =============================
def process_file(file_path: Path, repo_root: Path):
    """
    Process a single file to extract content and metadata.

    Args:
        file_path (Path): Path to the file.
        repo_root (Path): Path to the repository root.

    Returns:
        dict | None: A dictionary with the following keys if processed successfully:
            - `text` (str): File content.
            - `file_path` (str): Relative path from repo root.
            - `language` (str): Detected language based on extension.
            - `size` (int): Length of the text content.
        Returns None if the file is excluded or cannot be processed.
    """
    try:
        ext = file_path.suffix.lower()
        if ext in BINARY_EXTENSIONS:
            return None
        if any(part.lower() in EXCLUDE_DIRS for part in file_path.parts):
            return None
        if file_path.name.lower() in EXCLUDE_FILES:
            return None
        if any(file_path.match(p) for p in EXCLUDE_PATTERNS):
            return None
        if file_path.stat().st_size == 0:
            return None
        if not is_text_file(file_path):
            return None

        encodings = ["utf-8", "utf-8-sig", "latin1", "cp1252", "gbk"]
        content = None
        for enc in encodings:
            try:
                with open(file_path, "r", encoding=enc, errors="ignore") as f:
                    content = f.read().strip()
                break
            except Exception:
                continue

        return {
            "text": content,
            "file_path": str(file_path.relative_to(repo_root)),
            "language": EXT_TO_LANGUAGE.get(ext, ext.replace(".", "") or "unknown"),
            "size": len(content),
        }
    except Exception:
        return None


def create_alpaca_entry(item):
    """
    Create an Alpaca format entry from processed file item.

    Args:
        item (dict): Processed file item with text, file_path, language, size

    Returns:
        dict: Alpaca format entry
    """
    file_path = item["file_path"]
    language = item["language"]
    content = item["text"]

    # 根据文件类型生成不同的instruction
    if language in ["python", "py"]:
        instruction = f"请解释这个Python代码文件的功能和实现逻辑"
        system = (
            "你是一个专业的Python代码分析师，能够详细解释代码的功能、结构和实现细节。"
        )
    elif language in ["javascript", "js", "ts", "typescript"]:
        instruction = f"请分析这个JavaScript/TypeScript代码的功能和设计模式"
        system = "你是一个前端开发专家，擅长分析JavaScript和TypeScript代码的设计模式和最佳实践。"
    elif language in ["java"]:
        instruction = f"请解析这个Java代码的结构和功能实现"
        system = "你是一个Java开发专家，能够深入分析Java代码的面向对象设计和功能实现。"
    elif language in ["cpp", "c++", "c"]:
        instruction = f"请分析这个C/C++代码的算法和数据结构实现"
        system = "你是一个系统级编程专家，精通C/C++的内存管理、算法优化和系统设计。"
    elif language in ["html"]:
        instruction = f"请分析这个HTML文件的结构和语义"
        system = "你是一个前端开发专家，擅长HTML语义化和Web标准。"
    elif language in ["css"]:
        instruction = f"请解释这个CSS样式文件的设计思路和布局方案"
        system = "你是一个UI/UX设计师，精通CSS布局、动画和响应式设计。"
    elif language in ["markdown", "md"]:
        instruction = f"请总结这个Markdown文档的主要内容和结构"
        system = "你是一个技术文档专家，能够准确提取和总结文档的核心信息。"
    elif language in ["json"]:
        instruction = f"请解释这个JSON配置文件的结构和用途"
        system = "你是一个系统配置专家，能够解释各种配置文件的作用和最佳实践。"
    elif language in ["yaml", "yml"]:
        instruction = f"请分析这个YAML配置文件的配置项和用途"
        system = "你是一个DevOps工程师，精通各种配置文件格式和部署配置。"
    else:
        instruction = f"请分析这个{language}文件的内容和功能"
        system = f"你是一个资深的{language}开发专家，能够深入分析代码结构和实现逻辑。"

    # 生成针对具体文件的instruction
    instruction = f"{instruction}：{file_path}"

    return {
        "instruction": instruction,
        "input": f"文件路径: {file_path}\n语言类型: {language}\n文件大小: {item['size']} 字符\n\n文件内容:\n```{language}\n{content}\n```",
        "output": f"这是一个{language}文件，位于 `{file_path}`。文件包含 {item['size']} 个字符的代码内容。\n\n基于文件内容的分析，该文件主要功能包括：\n\n1. **文件结构**: 该文件采用了标准的{language}语法结构\n2. **主要功能**: 需要根据具体代码内容进行详细分析\n3. **技术特点**: 使用了{language}的相关特性和最佳实践\n4. **代码质量**: 代码结构清晰，符合{language}的编码规范\n\n建议进一步分析具体的函数、类或模块实现来了解更详细的功能逻辑。",
        "system": system,
        "history": [],
    }


# =============================
# 生成数据集
# =============================
def generate_dataset():
    """
    Generate an Alpaca format dataset from repository files.

    Args:
        None

    Returns:
        list: A list of dictionaries in Alpaca format with the following keys:
            - `instruction` (str): User instruction or question.
            - `input` (str): Context information including file path, language, and content.
            - `output` (str): Model response analyzing the code.
            - `system` (str): System prompt or role setting.
            - `history` (list): Historical dialogue, empty for new conversations.
    """
    repo_path = Path(REPO_ROOT).resolve()
    print(f"🔍 扫描仓库: {repo_path}")

    all_files = [f for f in repo_path.rglob("*") if f.is_file()]
    print(f"✅ 发现 {len(all_files)} 个文件，开始处理...")

    # 处理文件内容
    results = []
    with ThreadPoolExecutor(
        max_workers=NUM_WORKERS or (os.cpu_count() or 1) * 2
    ) as exec:
        futures = {exec.submit(process_file, fp, repo_path): fp for fp in all_files}
        for future in as_completed(futures):
            item = future.result()
            if item:
                results.append(item)

    # 去重
    seen = set()
    unique = []
    for item in results:
        h = hashlib.md5(item["text"].encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            unique.append(item)

    # 转换为Alpaca格式
    print(f"🔄 转换为Alpaca格式...")
    alpaca_dataset = []
    for item in unique:
        try:
            alpaca_entry = create_alpaca_entry(item)
            alpaca_dataset.append(alpaca_entry)
        except Exception as e:
            print(f"⚠️  处理文件 {item.get('file_path', 'unknown')} 时出错: {e}")
            continue

    # 保存为标准JSON格式
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"🎉 Alpaca格式数据集生成完成: {OUTPUT_DATASET} (共 {len(alpaca_dataset)} 条训练样本)"
    )

    # 输出数据集统计信息
    language_stats = {}
    for entry in alpaca_dataset:
        # 从input中提取语言信息
        input_text = entry["input"]
        if "语言类型:" in input_text:
            lang = input_text.split("语言类型:")[1].split("\n")[0].strip()
            language_stats[lang] = language_stats.get(lang, 0) + 1

    print(f"\n📊 数据集语言分布:")
    for lang, count in sorted(language_stats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {lang}: {count} 个文件")

    return alpaca_dataset


def generate_enhanced_alpaca_dataset():
    """
    Generate enhanced Alpaca dataset with multiple instruction variations per file.

    Returns:
        list: Enhanced Alpaca format dataset with multiple training samples per file.
    """
    # 首先生成基础数据集
    basic_dataset = generate_dataset()

    # 为每个文件生成多种instruction变化
    enhanced_dataset = []

    instruction_templates = {
        "analysis": [
            "请详细分析这个代码文件的功能和实现原理",
            "分析这个文件中的核心算法和数据结构",
            "请解释这个代码的设计模式和架构思路",
        ],
        "explanation": [
            "请用通俗易懂的语言解释这段代码的作用",
            "这个代码文件实现了什么功能？请详细说明",
            "请逐行解释这个代码的执行逻辑",
        ],
        "optimization": [
            "请评估这个代码的性能并提出优化建议",
            "这个代码有哪些可以改进的地方？",
            "从代码质量角度分析这个文件的优缺点",
        ],
        "documentation": [
            "请为这个代码文件生成详细的技术文档",
            "如何为这个代码编写单元测试？",
            "请总结这个文件的API接口和使用方法",
        ],
    }

    for item in basic_dataset:
        # 保留原始条目
        enhanced_dataset.append(item)

        # 为每种模板生成额外的训练样本
        for template_type, templates in instruction_templates.items():
            for template in templates:
                enhanced_entry = item.copy()
                file_path = item["input"].split("文件路径:")[1].split("\n")[0].strip()
                enhanced_entry["instruction"] = f"{template}：{file_path}"

                # 根据不同的instruction类型调整system prompt
                if template_type == "optimization":
                    enhanced_entry["system"] = (
                        "你是一个代码优化专家，能够识别性能瓶颈并提供具体的优化建议。"
                    )
                elif template_type == "documentation":
                    enhanced_entry["system"] = (
                        "你是一个技术文档专家，能够编写清晰、准确的技术文档和测试用例。"
                    )
                elif template_type == "explanation":
                    enhanced_entry["system"] = (
                        "你是一个编程导师，擅长用简单易懂的方式解释复杂的代码逻辑。"
                    )

                enhanced_dataset.append(enhanced_entry)

    # 保存增强版数据集
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"🚀 增强版Alpaca数据集生成完成: {OUTPUT_DATASET} (共 {len(enhanced_dataset)} 条训练样本)"
    )
    return enhanced_dataset


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 步骤1：生成数据集
    generate_dataset()
