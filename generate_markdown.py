import os
import mimetypes
from datetime import datetime
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


class ThreadSafeFileProcessor:
    def __init__(self, source_dir="source", output_file="dataset.md", max_workers=None):
        self.source_dir = source_dir
        self.output_file = output_file
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)

        # Thread-safe data structures
        self.dataset = []
        self.dataset_lock = threading.Lock()
        self.progress_lock = threading.Lock()

        # Progress tracking
        self.processed_count = 0
        self.total_files = 0
        self.start_time = None

        # Supported file extensions
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

        # Extension to Markdown code language mapping
        self.extension_to_lang = {
            ".py": "python",
            ".js": "javascript",
            ".html": "html",
            ".css": "css",
            ".json": "json",
            ".xml": "xml",
            ".csv": "csv",
            ".sql": "sql",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".ini": "ini",
            ".cfg": "cfg",
            ".conf": "bash",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".h": "c",
            ".php": "php",
            ".rb": "ruby",
            ".go": "go",
            ".rs": "rust",
            ".sh": "bash",
            ".bat": "batch",
            ".ps1": "powershell",
            ".dockerfile": "dockerfile",
            ".gitignore": "gitignore",
            ".env": "env",
            ".log": "log",
            ".makefile": "makefile",
            ".cmake": "cmake",
            ".md": "markdown",
            ".txt": "text",
        }

        # Performance statistics
        self.stats = {
            "files_per_second": 0,
            "total_processing_time": 0,
            "thread_stats": {},
        }

    def read_text_file(self, file_path):
        """Read text file content - optimized version"""
        encodings = ["utf-8", "gbk", "gb2312", "latin1"]
        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding, buffering=8192) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                return f"Read error: {str(e)}"
        return "Unable to decode file content"

    def get_file_info(self, file_path):
        """Get basic file information"""
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
        """Process single file - thread-safe version"""
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
                "file_info": file_info,
                "processed_time": datetime.now().isoformat(),
                "thread_id": thread_id,
            }

            if extension in self.text_extensions:
                file_data["type"] = "text"
                content = self.read_text_file(file_path)
                file_data["content"] = content
                file_data["encoding"] = "utf-8"

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

    def update_progress(self):
        """Update progress - thread-safe"""
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
                    f"\rProgress: {self.processed_count}/{self.total_files} "
                    f"({progress:.1f}%) - {fps:.1f} files/s - "
                    f"Time elapsed: {elapsed:.1f}s",
                    end="",
                    flush=True,
                )

    def collect_all_files(self):
        """Collect all file paths"""
        all_files = []
        if not os.path.exists(self.source_dir):
            print(f"Error: Source directory '{self.source_dir}' does not exist")
            return all_files

        print(f"Scanning directory: {self.source_dir}")
        for root, dirs, files in os.walk(self.source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                all_files.append(file_path)

        self.total_files = len(all_files)
        print(f"Found {self.total_files} files")
        return all_files

    def process_directory_multithread(self):
        """Multithreaded directory processing"""
        all_files = self.collect_all_files()
        if not all_files:
            return False

        print(f"Starting multithreaded processing (worker threads: {self.max_workers})")
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
                    self.update_progress()
                except Exception as e:
                    print(f"\nFailed to process file {file_path}: {e}")

        print("\n" + "=" * 60)
        total_time = time.time() - self.start_time
        self.stats["total_processing_time"] = total_time
        self.stats["files_per_second"] = (
            self.total_files / total_time if total_time > 0 else 0
        )

        print(f"Processing completed!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Processing speed: {self.stats['files_per_second']:.2f} files/second")
        print(f"Active threads: {len(self.stats['thread_stats'])}")

        return True

    def generate_markdown_report(self):
        """Generate complete Markdown report"""
        # Create output directory (if it doesn't exist)
        Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_file, "w", encoding="utf-8") as md:
            md.write("## 🔍 File Content Details\n\n")
            for item in sorted(self.dataset, key=lambda x: x.get("id", 0)):
                if "error" in item:
                    continue

                extension = item["extension"]
                lang = self.extension_to_lang.get(extension, "")

                md.write(f"### 📄 File #{item['id']} - `{item['filename']}`\n\n")
                md.write(f"- **Path**: `{item['path']}`\n")
                md.write(f"- **Size**: `{item['file_info']['size']:,} B`\n")
                md.write(
                    f"- **Modified Time**: `{item['file_info']['modified_time']}`\n"
                )

                content = item.get("content", "")

                md.write("#### Content Preview\n\n")

                if lang:
                    # Wrap with corresponding language code block
                    md.write(f"```{lang}\n")
                    md.write(content)
                    md.write("\n```\n\n")
                else:
                    # Plain text or unknown type, write directly
                    md.write(content)
                    md.write("\n\n")

        print(f"\n✅ Markdown report generated: {self.output_file}")

    def run(self):
        """Run complete process"""
        if self.process_directory_multithread():
            self.generate_markdown_report()
            return True
        return False


def main():
    """Main function"""
    source_directory = "./source"  # Source directory
    output_md = "./dataset/dataset.md"  # Output as .md
    max_workers = None  # Automatically set thread count

    processor = ThreadSafeFileProcessor(
        source_dir=source_directory,
        output_file=output_md,
        max_workers=max_workers,
    )

    print("=== 📝 Multithreaded AIMarkdown Generator (Markdown Output Version) ===")
    print(f"Source directory: {source_directory}")
    print(f"Output file: {output_md}")
    print(f"Max worker threads: {processor.max_workers}")
    print(f"CPU cores: {os.cpu_count()}")
    print("=" * 60)

    success = processor.run()

    if success:
        print("\n🎉 Markdown report successfully generated as Markdown file!")
    else:
        print("\n❌ Processing failed, please check if source directory exists.")


if __name__ == "__main__":
    main()
