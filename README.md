# ⚖️ Law-LLM: 面向法律咨询场景的大语言模型微调与 RAG 系统

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 基于 DeepSeek-R1-Distill-Qwen-7B 与 LLaMA-Factory 构建法律领域 LoRA 微调流程，完成通用指令数据与法律咨询数据的格式统一、规则清洗、语义去重和代表性采样。针对法律文档设计按法律层级及法条切分的知识库构建方法，并基于 FAISS 实现检索增强问答。使用 vLLM 提供 OpenAI 兼容推理服务，通过 Flask 和 Gradio 完成 API 与交互界面。建立基础模型、SFT、RAG 和 SFT+RAG 的对照评测，分析法律结论正确性、法条引用准确性及模型幻觉情况。

---

## 目录

- [问题背景](#问题背景)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [数据处理](#数据处理)
- [LoRA 微调](#lora-微调)
- [RAG 建库](#rag-建库)
- [模型部署](#模型部署)
- [实验结果](#实验结果)
- [案例展示](#案例展示)
- [已知限制](#已知限制)
- [项目路线图](#项目路线图)
- [免责声明](#免责声明)

---

## 问题背景

法律咨询场景对大语言模型提出了特殊要求：

1. **准确性**：法律条文引用必须准确，不能编造不存在的法条
2. **可追溯**：回答应标注具体的法律名称和条文编号
3. **安全性**：超出知识范围的问题应拒答，而非给出误导性回答
4. **专业性**：需要理解法律层级（编→章→节→条）和法律术语

本项目通过 **SFT 微调 + RAG 检索增强** 的组合方案解决上述问题：

- **SFT** 让模型学会法律咨询的回答风格和引用格式
- **RAG** 确保引用的法条来自真实法律文档，减少幻觉

---

## 系统架构

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

详细架构说明见 [docs/architecture.md](docs/architecture.md)。

---

## 快速开始

### 环境要求

- Python ≥ 3.9
- PyTorch ≥ 2.0（训练和本地推理需要）
- GPU（训练需要，推理推荐）

### 安装

```bash
git clone https://github.com/alexa0030/GreenRAG.git
cd GreenRAG
pip install -e ".[dev]"
```

### 一键体验（无需 GPU）

```bash
# 1. 构建 FAISS 索引（使用内置法律文档）
python scripts/build_index.py --law-dir data/laws --output-dir data/faiss_index

# 2. 启动 Gradio Demo（仅检索模式，无需 LLM）
python scripts/launch_demo.py --index-dir data/faiss_index
```

打开浏览器访问 `http://localhost:7860` 即可体验法律问答。

### 完整部署（需要 GPU）

```bash
# 1. 准备数据
make data

# 2. LoRA 微调
make train

# 3. 导出合并模型
llamafactory-cli export \
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --adapter_name_or_path outputs/sft_lora \
    --export_dir outputs/sft_lora_merged

# 4. 启动 vLLM 推理服务
make serve

# 5. 构建 RAG 索引
make index

# 6. 启动 Gradio Demo
make demo

# 7. 运行评测
make eval
```

---

## 数据处理

### 数据来源

| 数据集 | 说明 | 用途 |
|--------|------|------|
| DISC-Law-SFT | 法律 SFT 数据集 | 法律咨询训练 |
| JEC-QA | 司法考试题库 | 法律考试训练 |
| alpaca_gpt4_data_zh | 中文通用指令数据 | 通用能力训练 |
| 法律法规文档 | 刑法、个人信息保护法等 | RAG 知识库 |

详细说明见 [docs/dataset.md](docs/dataset.md)。

### 处理流程

```bash
python scripts/prepare_data.py \
    --legal-input data/raw/legal_raw.json \
    --general-input data/raw/general_raw.json \
    --output-dir data/processed \
    --embedding-model BAAI/bge-small-zh-v1.5 \
    --similarity-threshold 0.80 \
    --n-samples 3000 \
    --mix-ratio-legal 0.8 \
    --seed 42
```

处理完成后生成报告：

```
===== 数据处理报告 =====
原始数据数量:      59314
格式错误数量:      126
规则过滤数量:      18203
语义重复数量:      3487
最终保留数量:      37498
训练/验证/测试:    29998 / 3750 / 3750
随机种子:          42
```

### 关键设计

- **语义去重**：Sentence-BERT 向量化 + FAISS 近重复检测，阈值 0.80
- **代表性采样**：K-Center-Greedy 聚类，保证采样子集在向量空间中分布均匀
- **能力分类**：按长文本理解、逻辑推理、结构化表达、歧义消解四维分类配额
- **可复现**：固定随机种子，全流程参数化，输出处理报告

---

## LoRA 微调

### 训练配置

配置文件位于 `configs/sft_lora.yaml`：

```yaml
model_name_or_path: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
finetuning_type: lora
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
learning_rate: 5.0e-5
num_train_epochs: 3
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
cutoff_len: 4096
bf16: true
```

### 启动训练

```bash
bash scripts/train.sh
# 或自定义参数
LORA_RANK=16 NUM_EPOCHS=5 bash scripts/train.sh
```

### 训练环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4090 24GB (或同等) |
| 训练时长 | ~4 小时 (3 epochs, 5K samples) |
| 显存占用 | ~18GB (LoRA, bf16, batch=2×8) |
| 框架 | LLaMA-Factory |

训练日志和损失曲线保存在 `outputs/sft_lora/`。

---

## RAG 建库

### 法律文档切分

采用三级切分策略：

1. **Markdown 标题切分**：提取法律名称、编、章、节
2. **法条正则切分**：`第[一二三四五六七八九十百千万零\d]+条`，支持中文数字
3. **长文本递归切分**：超过 chunk_size 时进一步切分

每个法条块保留完整元数据：

```json
{
  "law_name": "中华人民共和国个人信息保护法",
  "article": "第二十八条",
  "chapter": "第二章",
  "section": "第一节",
  "effective_date": "2021年8月20日",
  "source": "data/laws/行政法/个人信息保护法.md"
}
```

### 构建索引

```bash
python scripts/build_index.py \
    --law-dir data/laws \
    --output-dir data/faiss_index \
    --embedding-model BAAI/bge-small-zh-v1.5 \
    --chunk-size 512
```

### RAG 问答流程

```
用户问题
  ↓
向量检索 Top-5 法条 (FAISS)
  ↓
可选 Cross-Encoder 重排序
  ↓
构建带法条引用的提示词
  ↓
LLM 生成回答（约束只引用检索到的法条）
  ↓
提取引用并格式化输出
  ↓
无证据时拒答
```

---

## 模型部署

### vLLM 推理服务

```bash
bash scripts/serve_vllm.sh
# 服务启动在 http://localhost:8000
# OpenAI 兼容 API
```

### Flask API

```bash
python -c "
from law_llm.service.api import run_server
run_server(host='0.0.0.0', port=5000, index_dir='data/faiss_index')
"
```

API 接口：

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/query` | POST | 法律问答 |
| `/retrieve` | POST | 仅检索法条 |
| `/laws` | GET | 列出知识库法律 |

请求示例：

```bash
curl -X POST http://localhost:5000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "生物识别信息属于什么类型的个人信息？", "use_rag": true}'
```

### Gradio Demo

```bash
python scripts/launch_demo.py --index-dir data/faiss_index
```

---

## 实验结果

### 评测体系

| 维度 | 指标 |
|------|------|
| 检索效果 | Recall@1, Recall@3, Recall@5, MRR |
| 法条引用 | Citation Precision, Recall, F1, 幻觉率 |
| 答案正确性 | 关键词匹配率, 结论判定正确率, LLM Judge |
| 拒答能力 | 不可回答问题拒答准确率 |

### 消融实验

四组对比实验（base / SFT / RAG / SFT+RAG）：

```bash
python scripts/evaluate.py --experiment base --output-dir outputs/eval_results
python scripts/evaluate.py --experiment sft --output-dir outputs/eval_results
python scripts/evaluate.py --experiment base+rag --output-dir outputs/eval_results
python scripts/evaluate.py --experiment sft+rag --output-dir outputs/eval_results
```

> **注意**：具体数值取决于评测集大小和标注质量。在自建 10 条示例评测集上，SFT+RAG 的检索 Recall@5 和法条引用 F1 表现最优。建议构建 50+ 条人工标注评测集以获得更可靠的结论。

详细评测说明见 [docs/evaluation.md](docs/evaluation.md)。

---

## 案例展示

### 示例 1：法条检索问答

**问题**：生物识别信息属于什么类型的个人信息？

**回答**：

> 根据《中华人民共和国个人信息保护法》第二十八条，生物识别信息属于敏感个人信息。
>
> 敏感个人信息是一旦泄露或者非法使用，容易导致自然人的人格尊严受到侵害或者人身、财产安全受到危害的个人信息，包括生物识别、宗教信仰、特定身份、医疗健康、金融账户、行踪轨迹等信息，以及不满十四周岁未成年人的个人信息。
>
> **引用依据：**
> [1]《中华人民共和国个人信息保护法》第二十八条

### 示例 2：拒答机制

**问题**：中国去年的GDP增长率是多少？

**回答**：

> 抱歉，根据现有法律知识库，未检索到与您问题相关的法律条文。建议您咨询相关专业部门获取准确信息。

---

## 已知限制

1. **法律覆盖范围有限**：当前知识库仅包含刑法和个人信息保护法等少量法律，未覆盖全部现行法律法规
2. **评测集规模较小**：内置示例集仅 10 条，不足以支撑统计显著的结论
3. **无法律版本管理**：未实现法律修订版本和生效日期的严格管理
4. **检索为精确匹配**：使用 FAISS Flat 索引，未针对超大规模法律库优化
5. **单一语言**：仅支持中文法律问答
6. **非法律意见**：系统输出仅供参考，不构成正式法律意见

---

## 项目路线图

- [x] 数据清洗与格式统一
- [x] 语义去重（FAISS）
- [x] K-Center-Greedy 代表性采样
- [x] LoRA 微调配置
- [x] 法律文档按层级切分
- [x] FAISS 索引构建
- [x] RAG 检索增强问答
- [x] Flask API + Gradio Demo
- [x] 多维度评测体系
- [x] 四组消融实验框架
- [ ] 扩充法律知识库（民法、行政法、劳动法等）
- [ ] 50+ 条人工标注评测集
- [ ] 法律版本和生效日期管理
- [ ] SFT vs RAG 详细消融分析
- [ ] Docker 一键部署
- [ ] CI/CD 自动化测试
- [ ] 结构化日志和请求追踪

---

## 免责声明

> ⚠️ **本项目用于技术研究和模型能力验证，生成结果不构成法律意见。**
>
> 法律问题具有高度专业性，实际法律事务请咨询持有执业资格的律师。本项目不对任何基于系统输出做出的决定承担责任。

---

## License

[MIT License](LICENSE)
