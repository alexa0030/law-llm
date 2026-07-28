# -*- coding: utf-8 -*-
"""启动 Gradio Demo。

用法:
    python scripts/launch_demo.py \
        --index-dir data/faiss_index \
        --api-base http://localhost:8000/v1 \
        --port 7860
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from law_llm.app.gradio_app import launch


def parse_args():
    parser = argparse.ArgumentParser(description="启动 Gradio Demo")
    parser.add_argument(
        "--index-dir", type=str, default="data/faiss_index",
        help="FAISS 索引目录",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="BAAI/bge-small-zh-v1.5",
        help="embedding 模型",
    )
    parser.add_argument(
        "--model-mode", type=str, default="api",
        choices=["api", "local"],
        help="模型推理模式",
    )
    parser.add_argument(
        "--api-base", type=str, default="http://localhost:8000/v1",
        help="vLLM API 地址",
    )
    parser.add_argument(
        "--model-name", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="模型名称",
    )
    parser.add_argument(
        "--local-model-path", type=str, default="",
        help="本地模型路径",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="检索 Top-K",
    )
    parser.add_argument(
        "--use-reranker", action="store_true",
        help="启用重排序",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备",
    )
    parser.add_argument(
        "--host", type=str, default="0.0.0.0",
        help="监听地址",
    )
    parser.add_argument(
        "--port", type=int, default=7860,
        help="监听端口",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("启动 Gradio Demo")
    print("=" * 60)

    launch(
        host=args.host,
        port=args.port,
        index_dir=args.index_dir,
        embedding_model=args.embedding_model,
        model_mode=args.model_mode,
        api_base=args.api_base,
        model_name=args.model_name,
        local_model_path=args.local_model_path,
        top_k=args.top_k,
        use_reranker=args.use_reranker,
        device=args.device,
    )


if __name__ == "__main__":
    main()
