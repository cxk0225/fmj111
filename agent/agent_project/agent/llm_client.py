"""LLM API 调用封装

支持所有 OpenAI 兼容接口（OpenAI / DeepSeek / 通义千问 / 硅基流动等）。
"""
from typing import List, Dict, Optional, Generator

from openai import OpenAI

from config.api_config import llm_config


class LLMClient:
    """大模型 API 客户端"""

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = None,
    ):
        self.api_key = api_key or llm_config.API_KEY
        self.base_url = base_url or llm_config.BASE_URL
        self.model = model or llm_config.MODEL
        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> str:
        """
        普通对话（非流式）。
        返回完整回复文本。
        """
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """
        流式对话。
        逐 chunk 产生文本片段，可用于前端打字机效果。
        """
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    def count_tokens(self, text: str) -> int:
        """估算 token 数量（通过字符简单估算，兼容中英文）"""
        # 中英文混合粗略估算：1 token ≈ 1.5 个中文字符 ≈ 0.25 个英文词
        try:
            import tiktoken
            enc = tiktoken.encoding_for_model(self.model)
            return len(enc.encode(text))
        except Exception:
            # fallback: 字符数 / 2 的粗略估算
            return len(text) // 2
