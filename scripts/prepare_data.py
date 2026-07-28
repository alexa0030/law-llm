# -*- coding: utf-8 -*-
"""数据准备流水线 —— 一键完成清洗、去重、采样、划分。

用法:
    python scripts/prepare_data.py \
        --legal-input data/raw/legal_raw.json \
        --general-input data/raw/general_raw.json \
        --output-dir data/processed \
        --embedding-model BAAI/bge-small-zh-v1.5 \
        --similarity-threshold 0.80 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 将 src 加入路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from law_llm.data.clean import clean_dataset
from law_llm.data.deduplicate import deduplicate_dataset
from law_llm.data.sample import sample_dataset
from law_llm.data.split import mix_and_split
from law_llm.data.report import build_report, save_report, print_report


def parse_args():
    parser = argparse.ArgumentParser(description="法律 SFT 数据准备流水线")
    parser.add_argument(
        "--legal-input", type=str, required=True,
        help="法律数据输入路径 (JSON)",
    )
    parser.add_argument(
        "--general-input", type=str, default="",
        help="通用数据输入路径 (JSON)，可选",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/processed",
        help="输出目录",
    )
    parser.add_argument(
        "--embedding-model", type=str, default="BAAI/bge-small-zh-v1.5",
        help="Sentence-BERT 模型",
    )
    parser.add_argument(
        "--similarity-threshold", type=float, default=0.80,
        help="语义去重相似度阈值",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3000,
        help="通用数据采样数量",
    )
    parser.add_argument(
        "--mix-ratio-legal", type=float, default=0.8,
        help="法律数据混合比例 (0-1)",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.8,
        help="训练集比例",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.1,
        help="验证集比例",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="计算设备 (cpu/cuda)",
    )
    parser.add_argument(
        "--skip-dedup", action="store_true",
        help="跳过去重步骤（快速测试用）",
    )
    parser.add_argument(
        "--skip-sample", action="store_true",
        help="跳过采样步骤",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    log_dir = out / "logs"
    log_dir.mkdir(exist_ok=True)

    report = build_report(
        original_count=0,
        format_error_count=0,
        rule_filtered_count=0,
        semantic_duplicate_count=0,
        final_count=0,
        seed=args.seed,
        config={
            "embedding_model": args.embedding_model,
            "similarity_threshold": args.similarity_threshold,
            "n_samples": args.n_samples,
            "mix_ratio_legal": args.mix_ratio_legal,
            "device": args.device,
            "skip_dedup": args.skip_dedup,
            "skip_sample": args.skip_sample,
        },
    )

    # ===================================================================
    # Step 1: 规则清洗
    # ===================================================================
    print("\n" + "=" * 60)
    print("Step 1: 规则清洗")
    print("=" * 60)

    # 法律数据清洗
    legal_clean_path = str(out / "legal_cleaned.json")
    legal_stats = clean_dataset(
        input_path=args.legal_input,
        output_path=legal_clean_path,
        log_path=str(log_dir / "legal_clean_log.json"),
    )
    print(f"  法律数据: {legal_stats}")

    report.original_count += legal_stats["original"]
    report.rule_filtered_count += legal_stats["filtered"]
    report.clean_reason_distribution.update(legal_stats.get("reason_distribution", {}))

    # 通用数据清洗（可选）
    general_clean_path = None
    if args.general_input and Path(args.general_input).exists():
        general_clean_path = str(out / "general_cleaned.json")
        general_stats = clean_dataset(
            input_path=args.general_input,
            output_path=general_clean_path,
            log_path=str(log_dir / "general_clean_log.json"),
        )
        print(f"  通用数据: {general_stats}")
        report.original_count += general_stats["original"]
        report.rule_filtered_count += general_stats["filtered"]

    # ===================================================================
    # Step 2: 语义去重
    # ===================================================================
    legal_dedup_path = str(out / "legal_deduped.json")

    if not args.skip_dedup:
        print("\n" + "=" * 60)
        print("Step 2: 语义去重")
        print("=" * 60)

        legal_dedup_stats = deduplicate_dataset(
            input_path=legal_clean_path,
            output_path=legal_dedup_path,
            log_path=str(log_dir / "legal_dedup_log.json"),
            embedding_model=args.embedding_model,
            similarity_threshold=args.similarity_threshold,
            device=args.device,
        )
        print(f"  法律数据去重: {legal_dedup_stats}")
        report.semantic_duplicate_count += legal_dedup_stats["exact_removed"]
        report.semantic_duplicate_count += legal_dedup_stats["semantic_removed"]
    else:
        print("\n  ⏭ 跳过去重步骤")
        legal_dedup_path = legal_clean_path

    # ===================================================================
    # Step 3: 通用数据采样
    # ===================================================================
    general_sampled_path = None
    if general_clean_path and not args.skip_sample:
        print("\n" + "=" * 60)
        print("Step 3: 通用数据代表性采样")
        print("=" * 60)

        general_sampled_path = str(out / "general_sampled.json")
        sample_stats = sample_dataset(
            input_path=general_clean_path,
            output_path=general_sampled_path,
            embedding_model=args.embedding_model,
            n_samples=args.n_samples,
            seed=args.seed,
            device=args.device,
        )
        print(f"  通用数据采样: {sample_stats}")
    else:
        print("\n  ⏭ 跳过采样步骤")

    # ===================================================================
    # Step 4: 混合与划分
    # ===================================================================
    print("\n" + "=" * 60)
    print("Step 4: 混合与划分")
    print("=" * 60)

    sources = [("legal", legal_dedup_path, args.mix_ratio_legal)]
    if general_sampled_path:
        sources.append(("general", general_sampled_path, 1 - args.mix_ratio_legal))

    mix_ratio = {
        "legal": args.mix_ratio_legal,
        "general": 1 - args.mix_ratio_legal,
    }

    split_stats = mix_and_split(
        sources=sources,
        output_dir=str(out / "splits"),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1 - args.train_ratio - args.val_ratio,
        seed=args.seed,
        mix_ratio=mix_ratio,
    )
    print(f"  混合划分: {split_stats}")

    # 更新报告
    report.final_count = split_stats["total"]
    report.train_count = split_stats["train"]
    report.val_count = split_stats["val"]
    report.test_count = split_stats["test"]

    # 保存报告
    report_path = str(out / "processing_report.json")
    save_report(report, report_path)
    print("\n" + "=" * 60)
    print_report(report)
    print(f"\n报告已保存至 {report_path}")
    print(f"处理后的数据已保存至 {out}")


if __name__ == "__main__":
    main()
