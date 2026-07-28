#!/bin/bash
# -*- coding: utf-8 -*-
# vLLM 推理服务启动脚本
#
# 前提条件：
#   1. 已安装 vLLM: pip install vllm
#   2. 已有 SFT 微调后的模型权重（或使用基础模型）
#
# 用法:
#   bash scripts/serve_vllm.sh                              # 默认配置
#   MODEL_PATH=./outputs/sft_lora_merged bash scripts/serve_vllm.sh  # 指定模型路径

set -euo pipefail

# =========================================================================
# 配置
# =========================================================================
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"}
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.90}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
DTYPE=${DTYPE:-"bfloat16"}
GPU_IDS=${GPU_IDS:-"0"}

echo "============================================================"
echo "vLLM 推理服务启动"
echo "============================================================"
echo "  模型路径:           $MODEL_PATH"
echo "  端口:               $PORT"
echo "  GPU:                $GPU_IDS"
echo "  GPU 显存利用率:     $GPU_MEMORY_UTILIZATION"
echo "  最大序列长度:       $MAX_MODEL_LEN"
echo "  数据类型:           $DTYPE"
echo "============================================================"

# =========================================================================
# 启动 vLLM OpenAI 兼容服务
# =========================================================================
CUDA_VISIBLE_DEVICES=$GPU_IDS python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --dtype $DTYPE \
    --trust-remote-code \
    --served-model-name "law-llm" \
    --api-key "EMPTY"

# 启动后可通过以下方式访问:
#   curl http://localhost:${PORT}/v1/chat/completions \
#     -H "Content-Type: application/json" \
#     -H "Authorization: Bearer EMPTY" \
#     -d '{
#       "model": "law-llm",
#       "messages": [{"role": "user", "content": "什么是故意犯罪？"}]
#     }'
