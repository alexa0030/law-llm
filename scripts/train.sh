#!/bin/bash
# -*- coding: utf-8 -*-
# LoRA 微调启动脚本 (LLaMA-Factory)
#
# 前提条件：
#   1. 已安装 LLaMA-Factory: pip install llamafactory
#   2. 已准备好数据: data/processed/splits/sft_train.json, sft_val.json
#   3. 已将数据目录链接到 LLaMA-Factory 的 data 目录
#
# 用法:
#   bash scripts/train.sh                          # 默认配置
#   bash scripts/train.sh --gpu 0                  # 指定 GPU
#   LORA_RANK=16 bash scripts/train.sh             # 覆盖 LoRA rank

set -euo pipefail

# =========================================================================
# 配置（可通过环境变量覆盖）
# =========================================================================
MODEL_NAME=${MODEL_NAME:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
DATA_DIR=${DATA_DIR:-"data/processed/splits"}
OUTPUT_DIR=${OUTPUT_DIR:-"outputs/sft_lora"}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.05}
LEARNING_RATE=${LEARNING_RATE:-5.0e-5}
NUM_EPOCHS=${NUM_EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-8}
CUTOFF_LEN=${CUTOFF_LEN:-4096}
GPU_IDS=${GPU_IDS:-"0"}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-100}

# =========================================================================
# 解析命令行参数
# =========================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu) GPU_IDS="$2"; shift 2 ;;
        --model) MODEL_NAME="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

echo "============================================================"
echo "LoRA 微调启动"
echo "============================================================"
echo "  模型:           $MODEL_NAME"
echo "  数据目录:       $DATA_DIR"
echo "  输出目录:       $OUTPUT_DIR"
echo "  LoRA Rank:      $LORA_RANK"
echo "  LoRA Alpha:     $LORA_ALPHA"
echo "  学习率:         $LEARNING_RATE"
echo "  训练轮数:       $NUM_EPOCHS"
echo "  批大小:         $BATCH_SIZE x $GRAD_ACCUM (有效: $((BATCH_SIZE * GRAD_ACCUM)))"
echo "  GPU:            $GPU_IDS"
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

# =========================================================================
# 启动训练
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$MODEL_NAME" \
    --finetuning_type lora \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --dataset_dir "$DATA_DIR" \
    --dataset law_sft_train \
    --template deepseek3 \
    --cutoff_len $CUTOFF_LEN \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --output_dir "$OUTPUT_DIR" \
    --plot_loss True \
    --bf16 True \
    --val_size 0.2 \
    --eval_strategy steps \
    --eval_steps 200 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --report_to tensorboard

echo ""
echo "============================================================"
echo "训练完成！"
echo "  最佳模型: $OUTPUT_DIR"
echo ""
echo "导出合并模型:"
echo "  llamafactory-cli export \\"
echo "    --model_name_or_path $MODEL_NAME \\"
echo "    --adapter_name_or_path $OUTPUT_DIR \\"
echo "    --export_dir ${OUTPUT_DIR}_merged \\"
echo "    --export_size 4 \\"
echo "    --export_legacy_format False"
echo "============================================================"
