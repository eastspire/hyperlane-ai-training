import os
import json
import hashlib
import mimetypes
from datetime import datetime
from pathlib import Path
import base64
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import time


class ThreadSafeFileProcessor:
    def __init__(self, source_dir="source", output_file="dataset.md", max_workers=None):
        self.source_dir = source_dir
        self.output_file = output_file
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

        # 线程安全的数据结构
        self.dataset = []
        self.dataset_lock = threading.Lock()
        self.progress_lock = threading.Lock()

        # 进度跟踪
        self.processed_count = 0
        self.total_files = 0
        self.start_time = None

        # 支持的文件扩展名
        self.text_extensions = {
            ".txt",
            ".md",
            ".py",
            ".js",
            ".html",
            ".css",
            ".json",
            ".xml",
            ".csv",
            ".sql",
            ".yml",
            ".yaml",
            ".ini",
            ".cfg",
            ".conf",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".php",
            ".rb",
            ".go",
            ".rs",
            ".sh",
            ".bat",
            ".ps1",
            ".dockerfile",
            ".gitignore",
            ".env",
            ".log",
            ".readme",
            ".license",
            ".makefile",
            ".cmake",
        }

        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".bmp",
            ".tiff",
            ".webp",
            ".svg",
        }

        # 性能统计
        self.stats = {
            "files_per_second": 0,
            "total_processing_time": 0,
            "thread_stats": {},
        }

    def get_file_hash(self, file_path):
        """计算文件的MD5哈希值 - 优化版本"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"

    def read_text_file(self, file_path):
        """读取文本文件内容 - 优化版本"""
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, buffering=8192) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return f"读取错误: {str(e)}"
        return "无法解码文件内容"

    def encode_binary_file(self, file_path):
        """将二进制文件编码为base64 - 优化版本"""
        try:
            with open(file_path, "rb") as f:
                file_size = os.path.getsize(file_path)
                if file_size > 10 * 1024 * 1024:  # 大于10MB
                    return "文件过大，跳过编码"
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            return f"编码错误: {str(e)}"

    def get_file_info(self, file_path):
        """获取文件基本信息"""
        try:
            stat = os.stat(file_path)
            return {
                "size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "mime_type": mimetypes.guess_type(file_path)[0] or "unknown",
            }
        except Exception as e:
            return {"size": 0, "error": str(e)}

    def process_single_file(self, file_path, file_id):
        """处理单个文件 - 线程安全版本"""
        thread_id = threading.current_thread().ident

        if thread_id not in self.stats["thread_stats"]:
            self.stats["thread_stats"][thread_id] = {"processed": 0, "errors": 0}

        try:
            file_path = Path(file_path)
            relative_path = file_path.relative_to(self.source_dir)
            extension = file_path.suffix.lower()

            file_info = self.get_file_info(file_path)

            file_data = {
                "id": file_id,
                "filename": file_path.name,
                "path": str(relative_path),
                "full_path": str(file_path),
                "extension": extension,
                "hash": self.get_file_hash(file_path),
                "file_info": file_info,
                "processed_time": datetime.now().isoformat(),
                "thread_id": thread_id,
            }

            if extension in self.text_extensions:
                file_data["type"] = "text"
                content = self.read_text_file(file_path)
                file_data["content"] = content
                file_data["encoding"] = "utf-8"
                file_data["preview"] = (
                    content[:500] + "..." if len(content) > 500 else content
                )

            elif extension in self.image_extensions:
                file_data["type"] = "image"
                size = file_info.get("size", 0)
                if size < 5 * 1024 * 1024:
                    file_data["content"] = self.encode_binary_file(file_path)
                    file_data["encoding"] = "base64"
                else:
                    file_data["content"] = "图像文件过大，仅保存元数据"
                    file_data["encoding"] = "none"
                file_data["preview"] = f"📷 图像文件 ({size} B)"

            else:
                file_data["type"] = "binary"
                size = file_info.get("size", 0)
                if size < 1024 * 1024:
                    file_data["content"] = self.encode_binary_file(file_path)
                    file_data["encoding"] = "base64"
                else:
                    file_data["content"] = "文件过大，仅保存元数据"
                    file_data["encoding"] = "none"
                file_data["preview"] = f"📁 二进制文件 ({size} B)"

            self.stats["thread_stats"][thread_id]["processed"] += 1
            return file_data

        except Exception as e:
            self.stats["thread_stats"][thread_id]["errors"] += 1
            return {
                "id": file_id,
                "filename": (
                    file_path.name if hasattr(file_path, "name") else str(file_path)
                ),
                "path": str(file_path),
                "error": str(e),
                "processed_time": datetime.now().isoformat(),
                "thread_id": thread_id,
            }

    def update_progress(self, file_path=""):
        """更新进度 - 线程安全"""
        with self.progress_lock:
            self.processed_count += 1
            if (
                self.processed_count % 10 == 0
                or self.processed_count == self.total_files
            ):
                elapsed = time.time() - self.start_time
                fps = self.processed_count / elapsed if elapsed > 0 else 0
                progress = (self.processed_count / self.total_files) * 100

                print(
                    f"\r进度: {self.processed_count}/{self.total_files} "
                    f"({progress:.1f}%) - {fps:.1f} files/s - "
                    f"用时: {elapsed:.1f}s",
                    end="",
                    flush=True,
                )

    def collect_all_files(self):
        """收集所有文件路径"""
        all_files = []
        if not os.path.exists(self.source_dir):
            print(f"错误: 源目录 '{self.source_dir}' 不存在")
            return all_files

        print(f"正在扫描目录: {self.source_dir}")
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        self.total_files = len(all_files)
        print(f"发现 {self.total_files} 个文件")
        return all_files

    def process_directory_multithread(self):
        """多线程处理目录"""
        all_files = self.collect_all_files()
        if not all_files:
            return False

        print(f"开始多线程处理 (工作线程数: {self.max_workers})")
        print("-" * 60)

        self.start_time = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_single_file, file_path, idx + 1): file_path
                for idx, file_path in enumerate(all_files)
            }

            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    with self.dataset_lock:
                        self.dataset.append(result)
                    self.update_progress(file_path)
                except Exception as e:
                    print(f"\n处理文件失败 {file_path}: {e}")

        print("\n" + "=" * 60)
        total_time = time.time() - self.start_time
        self.stats["total_processing_time"] = total_time
        self.stats["files_per_second"] = (
            self.total_files / total_time if total_time > 0 else 0
        )

        print(f"处理完成!")
        print(f"总用时: {total_time:.2f} 秒")
        print(f"处理速度: {self.stats['files_per_second']:.2f} 文件/秒")
        print(f"活跃线程数: {len(self.stats['thread_stats'])}")

        return True

    def generate_markdown_report(self):
        """生成完整的 Markdown 报告"""
        total_files = len(self.dataset)
        errors = sum(1 for f in self.dataset if "error" in f)
        success = total_files - errors

        file_types = {}
        extensions = {}
        total_size = 0

        for item in self.dataset:
            if "error" in item:
                continue
            ftype = item.get("type", "unknown")
            ext = item.get("extension", "未知")
            file_types[ftype] = file_types.get(ftype, 0) + 1
            extensions[ext] = extensions.get(ext, 0) + 1
            if "file_info" in item and "size" in item["file_info"]:
                total_size += item["file_info"]["size"]

        def human_readable_size(size):
            for unit in ["B", "KB", "MB", "GB"]:
                if size < 1024.0:
                    return f"{size:.2f} {unit}"
                size /= 1024.0
            return f"{size:.2f} TB"

        # 创建输出目录（如果不存在）
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as md:
            md.write(f"# 📁 AI 数据集报告\n\n")
            md.write(
                f"> 生成时间: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`\n\n"
            )

            md.write("## 📊 摘要信息\n\n")
            md.write("| 项目 | 值 |\n")
            md.write("|------|-----|\n")
            md.write(f"| 总文件数 | {total_files} |\n")
            md.write(f"| 成功处理 | {success} |\n")
            md.write(f"| 处理失败 | {errors} |\n")
            md.write(f"| 总大小 | {human_readable_size(total_size)} |\n")
            md.write(f"| 处理耗时 | {self.stats['total_processing_time']:.2f} 秒 |\n")
            md.write(f"| 处理速度 | {self.stats['files_per_second']:.2f} 文件/秒 |\n")
            md.write(f"| 线程数 | {self.max_workers} |\n\n")

            md.write("## 🧩 文件类型分布\n\n")
            md.write("| 类型 | 数量 |\n")
            md.write("|------|------|\n")
            for t, c in file_types.items():
                md.write(f"| `{t}` | {c} |\n")
            md.write("\n")

            md.write("## 🔧 扩展名统计\n\n")
            md.write("| 扩展名 | 数量 |\n")
            md.write("|--------|------|\n")
            for ext, cnt in sorted(extensions.items(), key=lambda x: -x[1]):
                md.write(f"| `{ext}` | {cnt} |\n")
            md.write("\n")

            md.write("## ⚙️ 线程性能统计\n\n")
            md.write("| 线程ID | 处理文件数 | 错误数 |\n")
            md.write("|--------|-----------|--------|\n")
            for tid, stat in self.stats["thread_stats"].items():
                md.write(f"| `{tid}` | {stat['processed']} | {stat['errors']} |\n")
            md.write("\n")

            md.write("## 📄 详细文件列表\n\n")
            md.write("| ID | 文件名 | 路径 | 类型 | 大小 | 修改时间 | 预览 |\n")
            md.write("|----|--------|------|------|------|----------|-------|\n")

            for item in sorted(self.dataset, key=lambda x: x.get("id", 0)):
                filename = item["filename"]
                path = item["path"]
                ftype = item.get("type", "unknown")
                preview = item.get("preview", "")
                size = item["file_info"].get("size", 0)
                mtime = item["file_info"].get("modified_time", "N/A")

                md.write(
                    f"| `{item['id']}` "
                    f"| `{filename}` "
                    f"| `{path}` "
                    f"| `{ftype}` "
                    f"| `{size:,} B` "
                    f"| `{mtime.split('T')[0]}` "
                    f"| {preview.replace('|', '\\|')} |\n"
                )

            md.write("\n")

            md.write("## 🔍 文件内容详情\n\n")
            for item in sorted(self.dataset, key=lambda x: x.get("id", 0)):
                if "error" in item:
                    continue

                md.write(f"### 📄 文件 #{item['id']} - `{item['filename']}`\n\n")
                md.write(f"- **路径**: `{item['path']}`\n")
                md.write(f"- **类型**: `{item['type']}`\n")
                md.write(f"- **大小**: `{item['file_info']['size']:,} B`\n")
                md.write(f"- **修改时间**: `{item['file_info']['modified_time']}`\n")
                md.write(f"- **编码**: `{item.get('encoding', 'N/A')}`\n\n")

                content = item.get("content", "")
                if item["type"] == "text":
                    md.write("#### 内容预览\n\n")
                    md.write("```txt\n")
                    md.write(
                        (content[:2000] + "...\n")
                        if len(content) > 2000
                        else content + "\n"
                    )
                    md.write("```\n\n")

        print(f"\n✅ Markdown 报告已生成: {self.output_file}")

    def run(self):
        """运行完整流程"""
        if self.process_directory_multithread():
            self.generate_markdown_report()
            return True
        return False


def main():
    """主函数"""
    source_directory = "./source"  # 源目录
    output_md = "./dataset/dataset.md"  # 输出为 .md
    max_workers = None  # 自动设置线程数

    processor = ThreadSafeFileProcessor(
        source_dir=source_directory,
        output_file=output_md,
        max_workers=max_workers,
    )

    print("=== 📝 多线程AI数据集生成器 (Markdown 输出版) ===")
    print(f"源目录: {source_directory}")
    print(f"输出文件: {output_md}")
    print(f"最大工作线程数: {processor.max_workers}")
    print(f"CPU核心数: {os.cpu_count()}")
    print("=" * 60)

    success = processor.run()

    if success:
        print("\n🎉 数据集报告已成功生成为 Markdown 文件！")
    else:
        print("\n❌ 处理失败，请检查源目录是否存在。")


if __name__ == "__main__":
    main()
