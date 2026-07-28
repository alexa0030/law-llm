.PHONY: help install dev test clean data index train serve demo eval docker

help: ## 显示帮助信息
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## 安装依赖
	pip install -e .

dev: ## 安装开发依赖
	pip install -e ".[dev]"

test: ## 运行测试
	pytest tests/ -v

data: ## 准备数据
	python scripts/prepare_data.py \
		--legal-input data/raw/legal_raw.json \
		--general-input data/raw/general_raw.json \
		--output-dir data/processed \
		--seed 42

index: ## 构建 FAISS 索引
	python scripts/build_index.py \
		--law-dir data/laws \
		--output-dir data/faiss_index

train: ## LoRA 微调
	bash scripts/train.sh

serve: ## 启动 vLLM 推理服务
	bash scripts/serve_vllm.sh

demo: ## 启动 Gradio Demo
	python scripts/launch_demo.py \
		--index-dir data/faiss_index \
		--api-base http://localhost:8000/v1

eval: ## 运行评测
	python scripts/evaluate.py \
		--eval-dataset data/eval/eval_dataset.json \
		--index-dir data/faiss_index \
		--experiment sft+rag \
		--output-dir outputs/eval_results

docker: ## Docker 构建
	docker build -t law-llm -f docker/Dockerfile .

docker-up: ## Docker Compose 启动
	docker-compose -f docker/docker-compose.yml up -d

clean: ## 清理缓存
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .ruff_cache
