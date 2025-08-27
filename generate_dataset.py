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
# =============================
# 处理单个文件
# =============================


def get_file_language(file_path: Path, content: str = None):
    """
    智能检测文件语言类型

    Args:
        file_path (Path): 文件路径
        content (str): 文件内容（可选）

    Returns:
        str: 检测到的语言类型
    """
    ext = file_path.suffix.lower()
    filename = file_path.name.lower()

    # 首先尝试扩展名匹配
    if ext in EXT_TO_LANGUAGE:
        return EXT_TO_LANGUAGE[ext]

    # 特殊文件名检测
    special_files = {
        "license": "license",
        "readme": "markdown",
        "changelog": "markdown",
        "dockerfile": "dockerfile",
        "makefile": "makefile",
        "jenkinsfile": "groovy",
        "vagrantfile": "ruby",
        "gemfile": "ruby",
        "requirements.txt": "text",
        "package.json": "json",
        "composer.json": "json",
        "cargo.toml": "toml",
        "pyproject.toml": "toml",
        ".gitignore": "gitignore",
        ".gitattributes": "gitattributes",
        ".env": "env",
        ".dockerignore": "dockerignore",
    }

    for pattern, lang in special_files.items():
        if pattern in filename:
            return lang

    # 基于文件内容的检测（如果提供了内容）
    if content:
        content_lower = content.lower().strip()

        # LICENSE文件检测
        if any(
            keyword in content_lower
            for keyword in ["license", "copyright", "permission is hereby granted"]
        ):
            return "license"

        # Shell脚本检测
        if content.startswith("#!/bin/bash") or content.startswith("#!/bin/sh"):
            return "shell"

        # Python脚本检测
        if content.startswith("#!/usr/bin/env python") or content.startswith(
            "#!/usr/bin/python"
        ):
            return "python"

        # HTML检测
        if content_lower.startswith("<!doctype html") or "<html" in content_lower:
            return "html"

        # XML检测
        if content.startswith("<?xml"):
            return "xml"

        # JSON检测
        if (content.startswith("{") and content.endswith("}")) or (
            content.startswith("[") and content.endswith("]")
        ):
            try:
                import json

                json.loads(content)
                return "json"
            except:
                pass

    # 如果都无法识别，返回文件扩展名或unknown
    return ext.replace(".", "") if ext else "text"


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
            "language": get_file_language(file_path, content),
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

    # 根据文件类型生成不同的instruction和system
    if language in ["python", "py"]:
        instruction = "请详细分析这个Python代码文件的功能、结构和实现逻辑"
        system = "你是一个专业的Python代码分析师，能够深入理解代码的功能、架构设计和实现细节。你会从代码结构、算法逻辑、设计模式等多个角度进行全面分析。"
    elif language in ["javascript", "js", "ts", "typescript"]:
        instruction = "请分析这个JavaScript/TypeScript代码的功能实现和设计模式"
        system = "你是一个前端开发专家，擅长分析JavaScript和TypeScript代码的设计模式、最佳实践和性能优化。"
    elif language in ["rs"]:
        instruction = "请分析这个Rust代码的功能实现和设计模式"
        system = (
            "你是一个Rust开发专家，擅长分析Rust代码的设计模式、最佳实践和性能优化。"
        )
    elif language in ["java"]:
        instruction = "请解析这个Java代码的面向对象设计和功能实现"
        system = (
            "你是一个Java开发专家，精通Java的面向对象编程、设计模式和企业级应用开发。"
        )
    elif language in ["cpp", "c++", "c"]:
        instruction = "请分析这个C/C++代码的算法实现和系统设计"
        system = "你是一个系统级编程专家，精通C/C++的内存管理、算法优化、数据结构和系统设计。"
    elif language in ["html"]:
        instruction = "请分析这个HTML文件的结构、语义和设计规范"
        system = "你是一个前端开发专家，精通HTML语义化、Web标准和用户体验设计。"
    elif language in ["css"]:
        instruction = "请解释这个CSS样式文件的设计思路和布局实现"
        system = "你是一个UI/UX设计师，精通CSS布局、响应式设计、动画效果和现代CSS特性。"
    elif language in ["markdown", "md"]:
        instruction = "请总结这个Markdown文档的主要内容、结构和要点"
        system = (
            "你是一个技术文档专家，擅长提取文档核心信息、分析文档结构和总结关键要点。"
        )
    elif language in ["json"]:
        instruction = "请解释这个JSON文件的数据结构、配置项和使用场景"
        system = "你是一个系统配置专家，能够解析各种配置文件格式，理解配置项的作用和最佳实践。"
    elif language in ["yaml", "yml"]:
        instruction = "请分析这个YAML配置文件的结构、配置项和应用场景"
        system = "你是一个DevOps工程师，精通各种配置文件格式、部署配置和自动化运维。"
    elif language == "license":
        instruction = "请分析这个开源许可证文件的内容和法律条款"
        system = "你是一个开源许可证专家，熟悉各种开源许可证的条款、限制和使用场景。"
    elif language == "dockerfile":
        instruction = "请分析这个Dockerfile的构建逻辑和容器配置"
        system = "你是一个DevOps工程师，精通Docker容器技术、镜像构建和部署优化。"
    elif language in ["shell", "bash"]:
        instruction = "请分析这个Shell脚本的功能逻辑和系统操作"
        system = "你是一个系统管理员，精通Shell脚本编程、系统运维和自动化任务。"
    elif language in ["xml"]:
        instruction = "请分析这个XML文件的结构和数据组织方式"
        system = "你是一个数据结构专家，熟悉XML格式、数据建模和结构化数据处理。"
    elif language in ["gitignore"]:
        instruction = "请解释这个Git忽略文件的配置规则和作用"
        system = "你是一个版本控制专家，精通Git工作流、代码管理和项目配置。"
    else:
        instruction = f"请分析这个{language}文件的内容结构和主要功能"
        system = f"你是一个资深的{language}开发专家，能够深入分析代码结构、实现逻辑和技术特点。"

    content_preview = content
    size_note = f"文件大小: {item['size']} 字符"

    # input包含文件的上下文信息和内容
    input_text = f"""文件: {file_path}
语言: {language}
{size_note}

代码内容:
```{language}
{content_preview}
```"""

    # 生成更真实的output，基于实际代码内容
    output = generate_realistic_analysis(content, language, file_path, item["size"])

    return {
        "instruction": instruction,
        "input": input_text,
        "output": output,
        "system": system,
        "history": [],
    }


def generate_realistic_analysis(content, language, file_path, size):
    """
    生成更真实的代码分析回答
    """
    # 简单的代码分析逻辑
    lines = content.split("\n")
    non_empty_lines = [line for line in lines if line.strip()]

    # 检测一些基本特征
    has_functions = any(
        "def " in line or "function " in line or "func " in line for line in lines
    )
    has_classes = any("class " in line for line in lines)
    has_imports = any(
        line.strip().startswith(("import ", "from ", "#include", "require(", "const "))
        for line in lines
    )
    has_comments = any(
        line.strip().startswith(("#", "//", "/*", "<!--")) for line in lines
    )

    analysis = f"这是一个{language}文件，位于 `{file_path}`，包含{size}个字符，共{len(lines)}行代码。\n\n"

    # 结构分析
    analysis += "**代码结构分析:**\n"
    if has_imports:
        analysis += "- 包含模块导入/引用语句，说明代码依赖其他模块或库\n"
    if has_classes:
        analysis += "- 定义了类结构，采用面向对象编程方式\n"
    if has_functions:
        analysis += "- 包含函数定义，代码模块化程度较好\n"
    if has_comments:
        analysis += "- 有注释说明，代码可读性较好\n"

    # 功能分析
    analysis += "\n**主要功能特点:**\n"

    if language in ["python", "py"]:
        if "def " in content:
            func_count = content.count("def ")
            analysis += f"- 定义了{func_count}个函数，实现特定的功能逻辑\n"
        if "class " in content:
            class_count = content.count("class ")
            analysis += f"- 包含{class_count}个类定义，采用面向对象设计\n"
        if "import " in content or "from " in content:
            analysis += "- 使用了外部库依赖，扩展了功能实现\n"

    elif language in ["javascript", "js"]:
        if "function" in content or "=>" in content:
            analysis += "- 包含JavaScript函数定义，实现交互逻辑\n"
        if "const " in content or "let " in content:
            analysis += "- 使用现代JavaScript语法，代码规范性较好\n"
        if "async" in content or "await" in content:
            analysis += "- 采用异步编程模式，处理异步操作\n"

    elif language in ["html"]:
        if "<script" in content:
            analysis += "- 包含JavaScript脚本，具有交互功能\n"
        if "<style" in content or "css" in content:
            analysis += "- 包含样式定义，注重页面外观设计\n"
        if "<!DOCTYPE" in content:
            analysis += "- 使用标准HTML5文档类型声明\n"

    elif language in ["css"]:
        selector_count = content.count("{")
        analysis += f"- 包含约{selector_count}个CSS规则，定义页面样式\n"
        if "@media" in content:
            analysis += "- 使用媒体查询，支持响应式设计\n"
        if "animation" in content or "transition" in content:
            analysis += "- 包含动画效果，提升用户体验\n"

    # 代码质量评估
    analysis += "\n**代码质量评估:**\n"
    comment_ratio = sum(
        1 for line in lines if line.strip().startswith(("#", "//", "/*", "<!--"))
    ) / max(len(non_empty_lines), 1)

    if comment_ratio > 0.1:
        analysis += "- 注释较为充分，代码可维护性好\n"
    elif comment_ratio > 0.05:
        analysis += "- 有适量注释，基本满足可读性要求\n"
    else:
        analysis += "- 注释相对较少，建议增加必要的说明\n"

    if len(non_empty_lines) < 50:
        analysis += "- 代码较为简洁，结构清晰\n"
    elif len(non_empty_lines) < 200:
        analysis += "- 代码规模适中，逻辑相对完整\n"
    else:
        analysis += "- 代码规模较大，功能相对复杂\n"

    analysis += f"\n这个{language}文件展现了良好的编程实践，建议结合具体业务需求进一步优化和完善。"

    return analysis


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

    # 保存为标准JSON格式（不是JSONL）
    output_file = OUTPUT_DATASET.replace(".jsonl", "_alpaca.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(alpaca_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"🎉 Alpaca格式数据集生成完成: {output_file} (共 {len(alpaca_dataset)} 条训练样本)"
    )

    # 输出数据集统计信息
    language_stats = {}
    for entry in alpaca_dataset:
        # 从input中提取语言信息
        input_text = entry["input"]
        if "语言类型:" in input_text:
            lang = input_text.split("语言类型:")[1].split("\n")[0].strip()
            language_stats[lang] = language_stats.get(lang, 0) + 1

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
    output_file = OUTPUT_DATASET.replace(".jsonl", "_alpaca_enhanced.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(enhanced_dataset, f, ensure_ascii=False, indent=2)

    print(
        f"🚀 增强版Alpaca数据集生成完成: {output_file} (共 {len(enhanced_dataset)} 条训练样本)"
    )
    return enhanced_dataset


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 步骤1：生成数据集
    generate_dataset()
