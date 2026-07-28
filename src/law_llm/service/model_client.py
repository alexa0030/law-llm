# -*- coding: utf-8 -*-
"""LLM 模型客户端 —— 支持 vLLM (OpenAI 兼容) 和直接 HuggingFace 推理。"""

from __future__ import annotations

import os
from typing import Any


class ModelClient:
    """LLM 推理客户端。

    支持两种模式：
    1. vLLM / OpenAI 兼容 API（推荐）：通过 HTTP 调用
    2. 直接 HuggingFace 推理：本地加载模型
    """

    def __init__(
        self,
        mode: str = "api",  # "api" or "local"
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        local_model_path: str = "",
        device: str = "auto",
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> None:
        self.mode = mode
        self.api_base = api_base
        self.api_key = api_key
        self.model_name = model_name
        self.local_model_path = local_model_path
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = None
        self._tokenizer = None
        self._model = None

    def _ensure_api_client(self) -> None:
        """延迟初始化 OpenAI 客户端。"""
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(
                base_url=self.api_base,
                api_key=self.api_key,
            )

    def _ensure_local_model(self) -> None:
        """延迟加载本地模型。"""
        if self._model is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = self.local_model_path or self.model_name
            self._tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            self._model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self._model.eval()

    def chat(self, messages: list[dict[str, str]], **kwargs) -> str:
        """调用 LLM 生成回复。

        Args:
            messages: OpenAI 格式的消息列表
            **kwargs: 额外生成参数

        Returns:
            生成的文本
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)

        if self.mode == "api":
            return self._chat_api(messages, temperature, max_tokens)
        else:
            return self._chat_local(messages, temperature, max_tokens)

    def _chat_api(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """通过 OpenAI 兼容 API 调用 vLLM。"""
        self._ensure_api_client()
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _chat_local(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """本地模型推理。"""
        self._ensure_local_model()
        import torch

        # 构建提示词
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(text, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # 只取新生成的部分
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    def is_ready(self) -> bool:
        """检查模型是否可用。"""
        if self.mode == "api":
            try:
                self._ensure_api_client()
                self._client.models.list()
                return True
            except Exception:
                return False
        else:
            return self._model is not None or os.path.exists(self.local_model_path)
