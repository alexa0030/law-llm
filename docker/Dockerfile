# -*- coding: utf-8 -*-
# Dockerfile for law-llm
# 多阶段构建：base → app

# =========================================================================
# Stage 1: Base image with dependencies
# =========================================================================
FROM python:3.10-slim AS base

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# =========================================================================
# Stage 2: Application
# =========================================================================
FROM base AS app

WORKDIR /app

# 复制项目代码
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY pyproject.toml .
COPY README.md .

# 安装项目
RUN pip install -e .

# 复制法律文档和示例数据
COPY data/laws/ ./data/laws/
COPY data/eval/ ./data/eval/
COPY data/samples/ ./data/samples/

# 暴露端口
# 5000: Flask API
# 7860: Gradio Demo
EXPOSE 5000 7860

# 环境变量
ENV PYTHONPATH=/app/src
ENV LAW_LLM_INDEX_DIR=/app/data/faiss_index
ENV LAW_LLM_API_BASE=http://localhost:8000/v1

# 默认启动 Gradio Demo
CMD ["python", "scripts/launch_demo.py", \
     "--index-dir", "/app/data/faiss_index", \
     "--host", "0.0.0.0", "--port", "7860"]
