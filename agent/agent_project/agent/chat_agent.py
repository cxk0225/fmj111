"""对话 Agent

支持两种模式：
  1. 纯对话模式 — 直接调用 LLM，无 RAG
  2. RAG 增强模式 — 检索文档 + LLM 生成
"""
from typing import List, Dict, Optional

from agent.llm_client import LLMClient
from rag.rag_chain import RAGChain

# 纯对话模式 System Prompt
CHAT_SYSTEM_PROMPT = """你是一个智能问答助手。
请友好、准确地回答用户的问题。如遇到不确定的信息，请如实告知用户。
用中文回答，语言简洁明了。
"""


class ChatAgent:
    """对话 Agent"""

    def __init__(
        self,
        llm_client: LLMClient,
        rag_chain: Optional[RAGChain] = None,
        max_history: int = 20,
    ):
        self.llm = llm_client
        self.rag = rag_chain
        self.max_history = max_history
        self.history: List[Dict[str, str]] = []

    def _trim_history(self):
        """控制历史长度，避免超出上下文"""
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def chat(self, message: str) -> str:
        """纯对话模式"""
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        response = self.llm.chat(messages, stream=False)

        # 更新历史
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": response})
        self._trim_history()

        return response

    def chat_stream(self, message: str):
        """纯对话模式（流式）"""
        messages = [{"role": "system", "content": CHAT_SYSTEM_PROMPT}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": message})

        full_response = ""
        for chunk in self.llm.chat_stream(messages):
            full_response += chunk
            yield chunk

        # 流式结束后更新历史
        self.history.append({"role": "user", "content": message})
        self.history.append({"role": "assistant", "content": full_response})
        self._trim_history()

    def rag_query(self, question: str, top_k: int = 5):
        """RAG 增强问答"""
        if self.rag is None:
            return {
                "answer": "RAG 模块未初始化，请先上传文档。",
                "sources": [],
                "raw_chunks": [],
            }
        return self.rag.query(question, top_k=top_k)

    def clear_history(self):
        """清空对话历史"""
        self.history = []

    def set_history(self, history: List[Dict[str, str]]):
        """设置对话历史（用于 UI 恢复）"""
        self.history = history[-self.max_history:]
