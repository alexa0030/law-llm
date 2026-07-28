# 评测体系

## 评测维度

本项目建立了多维度的评测体系，覆盖检索、引用、正确性和拒答四个层面。

### 1. 检索效果

| 指标 | 说明 |
|------|------|
| Recall@1 | Top-1 检索结果中包含正确法条的比例 |
| Recall@3 | Top-3 检索结果中包含正确法条的比例 |
| Recall@5 | Top-5 检索结果中包含正确法条的比例 |
| MRR | 平均倒数排名 (Mean Reciprocal Rank) |

### 2. 法条引用

| 指标 | 说明 |
|------|------|
| Citation Precision | 回答中引用的法条有多少是正确的 |
| Citation Recall | 应该引用的法条有多少被引用了 |
| Citation F1 | Precision 和 Recall 的调和平均 |
| Hallucination Rate | 引用了不在检索结果中的法条比例（法条幻觉率） |

### 3. 答案正确性

| 指标 | 说明 |
|------|------|
| Exact Match | 精确匹配率 |
| Keyword Match | 关键词匹配率（ground truth 关键信息覆盖率） |
| LLM Judge | LLM 评判分数 (0-1) |
| Conclusion Accuracy | 结论判定正确率 |

### 4. 拒答能力

| 指标 | 说明 |
|------|------|
| Refusal Accuracy | 不可回答问题的拒答准确率 |

## 对比实验

支持四组消融实验：

| 实验 | 使用 RAG | 使用 SFT | 目的 |
|------|---------|---------|------|
| `base` | ✗ | ✗ | 基线模型 |
| `sft` | ✗ | ✓ | 微调效果 |
| `base+rag` | ✓ | ✗ | RAG 效果 |
| `sft+rag` | ✓ | ✓ | 联合效果 |

运行命令：

```bash
# 分别运行四组实验
python scripts/evaluate.py --experiment base --output-dir outputs/eval_results
python scripts/evaluate.py --experiment sft --output-dir outputs/eval_results
python scripts/evaluate.py --experiment base+rag --output-dir outputs/eval_results
python scripts/evaluate.py --experiment sft+rag --output-dir outputs/eval_results
```

## 评测集

- 内置 10 条示例评测集（`src/law_llm/evaluation/dataset.py`）
- 支持自定义评测集，格式见 `data/eval/eval_dataset.json`
- 评测集包含：问题、标准答案、相关法条标注、类别（含不可回答类）

## 评测结果格式

每次评测生成两个文件：

1. `{experiment}_report.json` — 评测指标汇总
2. `{experiment}_predictions.json` — 每条样本的详细预测结果

## 注意事项

- 评测集中的问题不应出现在训练集中
- "结论判定正确率"基于关键词匹配，不是严格的人工标注准确率
- LLM Judge 为可选项，需要额外配置 LLM
- 评测结果仅供参考，不构成法律意见
