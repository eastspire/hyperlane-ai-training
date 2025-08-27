# export_gguf.py
import os

MERGED_MODEL = "qwen-coder-finetuned"
GGUF_OUTPUT = "qwen-coder-finetuned-q5.gguf"


def run_export():
    if not os.path.exists(MERGED_MODEL):
        print(f"❌ 模型不存在: {MERGED_MODEL}")
        print("请先运行 train_your_code_model.py 完成训练和合并")
        return

    cmd = (
        f"python llama.cpp/convert-hf-to-gguf.py {MERGED_MODEL} "
        f"--outfile {GGUF_OUTPUT} --q5_0"
    )
    print(f"🚀 执行: {cmd}")
    os.system(cmd)
    print(f"🎉 GGUF 模型已生成: {GGUF_OUTPUT}")


if __name__ == "__main__":
    run_export()
