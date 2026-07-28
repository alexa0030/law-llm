# 系统架构

## 整体流程

```
原始数据                    法律文档
  │                          │
  ▼                          ▼
格式统一与规则清洗          文档解析
  │                          │
  ▼                          ▼
语义去重 (FAISS)          法条切分 (编/章/节/条)
  │                          │
  ▼                          ▼
代表性采样 (K-Center)     Embedding 建库 (FAISS)
  │                          │
  ▼                          │
数据集划分 (8:1:1)           │
  │                          │
  ▼                          │
LoRA 微调 (LLaMA-Factory)    │
  │                          │
  ▼                          │
模型导出 → vLLM 部署         │
  │                          │
  └──────────┬───────────────┘
             │
             ▼
      RAG 检索增强问答
             │
             ▼
    ┌────────┴────────┐
    │                 │
    ▼                 ▼
  Flask API      Gradio Demo
```

## 模块说明

### 1. 数据处理 (`src/law_llm/data/`)

| 模块 | 功能 |
|------|------|
| `schema.py` | 数据模型定义 (SFTSample, LawArticle, ProcessingReport 等) |
| `clean.py` | 规则清洗：格式校验、质量过滤、关键词筛选 |
| `deduplicate.py` | 语义去重：文本归一化 → 精确去重 → FAISS 近重复检测 |
| `sample.py` | 代表性采样：K-Center-Greedy 聚类 + 能力分类配额 |
| `split.py` | 数据集划分：分层随机划分 + 多源混合 |
| `report.py` | 处理报告生成 |

### 2. RAG 检索 (`src/law_llm/rag/`)

| 模块 | 功能 |
|------|------|
| `loader.py` | 法律文档加载，提取法律名称和生效日期 |
| `splitter.py` | 法条切分：Markdown 标题 → 正则法条 → 递归长文本 |
| `indexer.py` | FAISS 索引构建、保存、加载 |
| `retriever.py` | 向量检索 Top-k + 元数据过滤 |
| `reranker.py` | Cross-Encoder 重排序（可选） |
| `pipeline.py` | 端到端 RAG 流水线：检索 → 上下文构建 → LLM 生成 → 引用提取 |

### 3. 评测 (`src/law_llm/evaluation/`)

| 模块 | 功能 |
|------|------|
| `dataset.py` | 评测集格式定义、加载、内置示例集 |
| `metrics.py` | 多维度指标计算：检索、引用、正确性、拒答 |
| `citation_eval.py` | 法条引用专项评测 |
| `report.py` | 评测报告生成与对比表 |

### 4. 服务 (`src/law_llm/service/`)

| 模块 | 功能 |
|------|------|
| `schemas.py` | API 请求/响应模型 |
| `model_client.py` | LLM 推理客户端 (vLLM API / 本地 HuggingFace) |
| `api.py` | Flask API 服务 |

### 5. 应用 (`src/law_llm/app/`)

| 模块 | 功能 |
|------|------|
| `gradio_app.py` | Gradio 交互界面 |

## 技术选型

| 组件 | 选型 | 理由 |
|------|------|------|
| 基座模型 | DeepSeek-R1-Distill-Qwen-7B | 中文能力强，蒸馏后推理高效 |
| 微调框架 | LLaMA-Factory | 成熟稳定，配置驱动 |
| 微调方法 | LoRA | 显存友好，可插拔 |
| Embedding | BAAI/bge-small-zh-v1.5 | 中文法律向量效果好，模型小 |
| 向量检索 | FAISS | 工业级，支持百万级向量 |
| 推理服务 | vLLM | OpenAI 兼容，高吞吐 |
| API | Flask | 轻量，易部署 |
| 前端 | Gradio | 快速原型展示 |
