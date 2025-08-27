# train_your_code_model.py
import json
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# 检查 GPU 是否可用 (只执行一次)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    is_amd_gpu = False
elif torch.device("cuda:0").type == "cuda":
    device = torch.device("cuda:0")
    print("AMD GPU is available. Using GPU.")
    is_amd_gpu = True
else:
    device = torch.device("cpu")
    print("No GPU available, using CPU.")
    is_amd_gpu = False

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
import os
import logging

# =============================
# 配置区（根据你的环境修改）
# =============================
REPO_ROOT = "."  # 修改为你的仓库路径，如: "/path/to/your/repo"
OUTPUT_DATASET = "full_text_dataset.jsonl"
TRAINED_MODEL_OUTPUT = "qwen-coder-finetuned"  # 训练后模型保存路径
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B"  # 支持中文/英文代码

# 二进制文件黑名单（只这些才排除）
BINARY_EXTENSIONS = {
    # 图像
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tiff",
    ".ico",
    ".svg",
    ".webp",
    # 音视频
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
    # 压缩包
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
    # 可执行文件
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
    # 数据库
    ".db",
    ".sqlite",
    ".mdb",
    ".accdb",
    # 文档（二进制）
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".pdf",
    # 设计文件
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
    "train_your_code_model.py",  # 排除自己
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

# 文件大小限制
MAX_FILE_SIZE_BYTES = 1024 * 1024  # 1MB
MIN_CONTENT_LENGTH = 1  # 至少1字符

# 多线程
NUM_WORKERS = None  # 自动

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
        if (
            file_path.stat().st_size == 0
            or file_path.stat().st_size > MAX_FILE_SIZE_BYTES
        ):
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
        if not content or len(content) < MIN_CONTENT_LENGTH:
            return None

        return {
            "text": content,
            "file_path": str(file_path.relative_to(repo_root)),
            "language": EXT_TO_LANGUAGE.get(ext, ext.replace(".", "") or "unknown"),
            "size": len(content),
        }
    except Exception:
        return None


# =============================
# 生成数据集
# =============================
def generate_dataset():
    repo_path = Path(REPO_ROOT).resolve()
    print(f"🔍 扫描仓库: {repo_path}")

    all_files = [f for f in repo_path.rglob("*") if f.is_file()]
    print(f"✅ 发现 {len(all_files)} 个文件，开始处理...")

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

    # 保存
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for item in unique:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"🎉 数据集生成完成: {OUTPUT_DATASET} (共 {len(unique)} 条文本文件)")
    return unique


# =============================
# 微调模型
# =============================
def fine_tune():
    print("📚 加载数据集...")
    dataset = Dataset.from_json(OUTPUT_DATASET)

    print("🔧 加载 Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize(examples):
        return tokenizer(
            examples["text"], truncation=True, max_length=MAX_SEQ_LENGTH, padding=False
        )

    print("🔄 分词中...")
    tokenized_ds = dataset.map(
        tokenize, batched=True, remove_columns=["text", "file_path", "language", "size"]
    )

    print("🚀 加载 Qwen-Coder-1.5B (4-bit 量化)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # AMD GPU 推荐 bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,  # ✅ 使用 4-bit
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    # LoRA
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 训练参数
    training_args = TrainingArguments(
        output_dir=TRAINED_MODEL_OUTPUT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=LOG_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,  # ✅ 改成 bfloat16
        fp16=False,  # ❌ 关闭 fp16
        max_grad_norm=0.3,
        gradient_checkpointing=True,
        report_to="none",
        eval_strategy="no",
        dataloader_pin_memory=False,
        optim="adamw_torch",
        dataloader_num_workers=os.cpu_count() or 4,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds.with_format("torch"),
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        tokenizer=tokenizer,
    )

    print("🔥 开始微调...")
    trainer.train()

    print("💾 保存模型...")
    model.save_pretrained(TRAINED_MODEL_OUTPUT)
    tokenizer.save_pretrained(TRAINED_MODEL_OUTPUT)
    print(f"🎉 训练完成！模型已保存到: {TRAINED_MODEL_OUTPUT}")


# =============================
# 主程序
# =============================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 步骤1：生成数据集
    generate_dataset()

    # 步骤2：微调模型
    fine_tune()

    print("✨ 你的专属代码模型已训练完成！")
    print(f"使用方式：")
    print(f"  from transformers import AutoModelForCausalLM, AutoTokenizer")
    print(f'  model = AutoModelForCausalLM.from_pretrained("{TRAINED_MODEL_OUTPUT}")')
    print(f'  tokenizer = AutoTokenizer.from_pretrained("{TRAINED_MODEL_OUTPUT}")')
