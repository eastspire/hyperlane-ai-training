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
    def __init__(
        self, source_dir="source", output_file="dataset.json", max_workers=None
    ):
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
                # 使用更大的缓冲区提升性能
                for chunk in iter(lambda: f.read(8192), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"

    def read_text_file(self, file_path):
        """读取文本文件内容 - 优化版本"""
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]

        # 先尝试最常用的编码
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
                # 对于大文件，分块读取
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

        # 记录线程统计
        if thread_id not in self.stats["thread_stats"]:
            self.stats["thread_stats"][thread_id] = {"processed": 0, "errors": 0}

        try:
            file_path = Path(file_path)
            relative_path = file_path.relative_to(self.source_dir)
            extension = file_path.suffix.lower()

            # 基础文件信息
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

            # 根据文件类型处理内容
            if extension in self.text_extensions:
                file_data["type"] = "text"
                file_data["content"] = self.read_text_file(file_path)
                file_data["encoding"] = "utf-8"

            elif extension in self.image_extensions:
                file_data["type"] = "image"
                if file_info.get("size", 0) < 5 * 1024 * 1024:  # 小于5MB
                    file_data["content"] = self.encode_binary_file(file_path)
                    file_data["encoding"] = "base64"
                else:
                    file_data["content"] = "图像文件过大，仅保存元数据"
                    file_data["encoding"] = "none"

            else:
                file_data["type"] = "binary"
                if file_info.get("size", 0) < 1024 * 1024:  # 小于1MB
                    file_data["content"] = self.encode_binary_file(file_path)
                    file_data["encoding"] = "base64"
                else:
                    file_data["content"] = "文件过大，仅保存元数据"
                    file_data["encoding"] = "none"

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

        # 使用ThreadPoolExecutor进行多线程处理
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.process_single_file, file_path, idx + 1): file_path
                for idx, file_path in enumerate(all_files)
            }

            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()

                    # 线程安全地添加到数据集
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

    def generate_summary(self):
        """生成数据集摘要信息"""
        total_files = len(self.dataset)
        file_types = {}
        total_size = 0
        extensions = {}
        errors = 0

        for item in self.dataset:
            # 统计错误
            if "error" in item:
                errors += 1
                continue

            # 统计文件类型
            file_type = item.get("type", "unknown")
            file_types[file_type] = file_types.get(file_type, 0) + 1

            # 统计文件扩展名
            ext = item.get("extension", "")
            extensions[ext] = extensions.get(ext, 0) + 1

            # 统计总大小
            if "file_info" in item and "size" in item["file_info"]:
                total_size += item["file_info"]["size"]

        # 线程性能统计
        thread_performance = {}
        for thread_id, stats in self.stats["thread_stats"].items():
            thread_performance[f"thread_{thread_id}"] = stats

        return {
            "total_files": total_files,
            "successful_files": total_files - errors,
            "failed_files": errors,
            "total_size": total_size,
            "total_size_human": self.human_readable_size(total_size),
            "file_types": file_types,
            "extensions": extensions,
            "performance": {
                "processing_time_seconds": self.stats["total_processing_time"],
                "files_per_second": self.stats["files_per_second"],
                "max_workers": self.max_workers,
                "thread_performance": thread_performance,
            },
            "generated_time": datetime.now().isoformat(),
        }

    def human_readable_size(self, size):
        """将字节转换为人类可读的格式"""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size < 1024.0:
                return f"{size:.2f} {unit}"
            size /= 1024.0
        return f"{size:.2f} PB"

    def save_dataset(self):
        """保存数据集到JSON文件"""
        print("正在生成摘要信息...")
        summary = self.generate_summary()

        # 按ID排序确保输出有序
        self.dataset.sort(key=lambda x: x.get("id", 0))

        final_dataset = {"summary": summary, "files": self.dataset}

        print("正在保存JSON文件...")
        try:
            with open(self.output_file, "w", encoding="utf-8") as f:
                json.dump(final_dataset, f, ensure_ascii=False, indent=2)

            print(f"\n✅ 数据集已保存到: {self.output_file}")
            print(f"📊 摘要信息:")
            print(f"   总文件数: {summary['total_files']}")
            print(f"   成功处理: {summary['successful_files']}")
            print(f"   失败文件: {summary['failed_files']}")
            print(f"   总大小: {summary['total_size_human']}")
            print(
                f"   处理速度: {summary['performance']['files_per_second']:.2f} 文件/秒"
            )
            print(f"   文件类型分布: {summary['file_types']}")

            return True

        except Exception as e:
            print(f"❌ 保存数据集时出错: {e}")
            return False

    def run(self):
        """运行完整的处理流程"""
        if self.process_directory_multithread():
            return self.save_dataset()
        return False


def main():
    """主函数 - 支持参数配置"""
    # 配置参数
    source_directory = "./source"  # 源目录路径
    output_json = "./dataset.json"  # 输出JSON文件名
    max_workers = None  # None表示自动检测，也可以手动设置如16

    # 如果需要自定义线程数，取消下面的注释并设置数值
    # max_workers = 16  # 设置为你希望的线程数

    processor = ThreadSafeFileProcessor(
        source_dir=source_directory, output_file=output_json, max_workers=max_workers
    )

    print("=== 多线程AI数据集生成器 ===")
    print(f"源目录: {source_directory}")
    print(f"输出文件: {output_json}")
    print(f"最大工作线程数: {processor.max_workers}")
    print(f"CPU核心数: {os.cpu_count()}")
    print("=" * 50)

    success = processor.run()

    if success:
        print("\n🎉 数据集生成完成!")
    else:
        print("\n❌ 数据集生成失败!")


if __name__ == "__main__":
    main()
