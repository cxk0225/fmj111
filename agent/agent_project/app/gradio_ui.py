"""Gradio 用户界面

三个面板：
  1. 纯对话 — 直接与 LLM 对话
  2. RAG 问答 — 上传文档后基于文档问答
  3. 文档管理 — 查看/删除已上传文档
"""
import os
import tempfile
from pathlib import Path
from typing import List, Tuple

import gradio as gr

from agent.llm_client import LLMClient
from agent.chat_agent import ChatAgent
from rag.vector_store import VectorStore
from rag.document_loader import DocumentLoader
from rag.text_splitter import NLPSemanticSplitter
from rag.retriever import Retriever, TfidfRetriever, highlight_keywords
from rag.rag_chain import RAGChain


class AgentUI:
    """Gradio 应用主控制器"""

    def __init__(self):
        # 初始化各模块
        self.llm = LLMClient()
        self.vector_store = VectorStore()
        self.tfidf = TfidfRetriever()
        self.retriever = Retriever(self.vector_store, self.tfidf)
        self.rag_chain = RAGChain(self.retriever, self.llm)
        self.chat_agent = ChatAgent(self.llm, self.rag_chain)
        self.splitter = NLPSemanticSplitter()

        # 构建 UI
        self.app = self._build_app()

    def _build_app(self):
        with gr.Blocks(
            title="AI 智能助手",
            theme=gr.themes.Soft(),
            css="""
            .source-box { border-left: 3px solid #4a90d9; padding: 8px 12px; margin: 4px 0; background: #f5f8ff; border-radius: 0 6px 6px 0; }
            .highlight { background: #fff3cd; padding: 0 2px; border-radius: 2px; }
            """
        ) as app:
            gr.Markdown("# 🧠 AI 智能助手")
            gr.Markdown("基于大模型 API + RAG 技术，支持纯对话与文档增强问答。")

            with gr.Tabs():
                # ===== Tab 1: 纯对话 =====
                with gr.TabItem("💬 纯对话"):
                    self._build_chat_tab()

                # ===== Tab 2: RAG 问答 =====
                with gr.TabItem("📚 RAG 智能问答"):
                    self._build_rag_tab()

                # ===== Tab 3: 文档管理 =====
                with gr.TabItem("📁 文档管理"):
                    self._build_doc_management_tab()

            # 底部配置栏
            with gr.Accordion("⚙️ 参数设置", open=False):
                with gr.Row():
                    self.top_k_slider = gr.Slider(
                        minimum=1, maximum=20, value=5, step=1,
                        label="检索返回数量 (Top-K)"
                    )
                    self.temp_slider = gr.Slider(
                        minimum=0, maximum=2, value=0.7, step=0.1,
                        label="LLM 温度 (Temperature)"
                    )
                with gr.Row():
                    self.sem_weight = gr.Slider(
                        minimum=0, maximum=1, value=0.7, step=0.1,
                        label="语义检索权重"
                    )
                    self.exact_weight = gr.Slider(
                        minimum=0, maximum=1, value=0.3, step=0.1,
                        label="精准匹配权重"
                    )

        return app

    def _build_chat_tab(self):
        """纯对话界面"""
        chatbot = gr.Chatbot(label="对话", height=450, type="messages")
        msg = gr.Textbox(label="输入消息", placeholder="输入你的问题...", lines=2)
        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")

        def respond(message, history):
            history = history or []
            # 将 gradio 历史（messages 格式）转为 ChatAgent 格式
            msg_history = []
            for i in range(0, len(history), 2):
                if i + 1 < len(history):
                    msg_history.append({"role": "user", "content": history[i]["content"]})
                    msg_history.append({"role": "assistant", "content": history[i + 1]["content"]})
                else:
                    msg_history.append({"role": "user", "content": history[i]["content"]})

            self.chat_agent.set_history(msg_history)

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": ""})
            full_response = ""
            for chunk in self.chat_agent.chat_stream(message):
                full_response += chunk
                history[-1]["content"] = full_response
                yield history, ""

        send_btn.click(respond, [msg, chatbot], [chatbot, msg])
        msg.submit(respond, [msg, chatbot], [chatbot, msg])
        clear_btn.click(lambda: ([], None), None, [chatbot, msg]).then(
            self.chat_agent.clear_history
        )

    def _build_rag_tab(self):
        """RAG 问答界面"""
        chatbot = gr.Chatbot(label="RAG 问答", height=400, type="messages")
        msg = gr.Textbox(label="输入问题", placeholder="基于已上传文档提问...", lines=2)

        with gr.Row():
            send_btn = gr.Button("发送", variant="primary")
            clear_btn = gr.Button("清空对话")

        # 文档上传区
        with gr.Row():
            file_upload = gr.File(
                label="上传文档（支持 txt / pdf / docx）",
                file_types=[".txt", ".pdf", ".docx"],
                file_count="multiple",
            )
        upload_status = gr.Textbox(label="上传状态", interactive=False)

        # 检索来源展示区
        with gr.Accordion("📎 检索来源", open=False):
            sources_display = gr.HTML(label="来源详情")

        def upload_files(files):
            if not files:
                return "请选择文件。"
            total_chunks = 0
            doc_names = []
            for f in files:
                fpath = f.name if hasattr(f, "name") else f
                try:
                    # 1. 加载文档
                    pages = DocumentLoader.load(fpath)
                    # 2. NLP 语义分割
                    all_chunks = []
                    for page in pages:
                        chunks = self.splitter.split(page["text"])
                        for chunk in chunks:
                            meta = page["metadata"].copy()
                            all_chunks.append({"text": chunk, "metadata": meta})
                    # 3. 入库（有内容才入库）
                    if all_chunks:
                        self.vector_store.add_documents(all_chunks)
                        total_chunks += len(all_chunks)
                    doc_names.append(Path(fpath).name)
                except Exception as e:
                    return f"上传失败 [{Path(fpath).name}]: {str(e)}"

            # 4. 重建 TF-IDF 索引（获取全部 chunk）
            all_data = self._get_all_chunks()
            self.tfidf.build_index(all_data)

            return f"✅ 成功上传 {len(doc_names)} 个文档，共 {total_chunks} 个分块。\n文档：{', '.join(doc_names)}"

        file_upload.upload(upload_files, file_upload, upload_status)

        def rag_respond(message, history):
            history = history or []
            top_k = int(self.top_k_slider.value)
            result = self.chat_agent.rag_query(message, top_k=top_k)

            answer = result["answer"]
            sources = result.get("sources", [])
            raw_chunks = result.get("raw_chunks", [])

            # 高亮关键词
            query = message
            highlighted_sources = []
            for src in sources:
                highlighted_sources.append(highlight_keywords(src, query))

            # 组装来源 HTML
            sources_html = "<div style='margin: 8px 0;'>"
            for s in highlighted_sources:
                sources_html += f"<div class='source-box'>{s}</div>"
            sources_html += "</div>"

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            return history, "", sources_html

        send_btn.click(rag_respond, [msg, chatbot], [chatbot, msg, sources_display])
        msg.submit(rag_respond, [msg, chatbot], [chatbot, msg, sources_display])
        clear_btn.click(lambda: ([], "", ""), None, [chatbot, msg, sources_display])

    def _build_doc_management_tab(self):
        """文档管理界面"""
        refresh_btn = gr.Button("🔄 刷新文档列表", variant="secondary")
        doc_list = gr.Dataframe(
            headers=["文档名称", "格式", "分块数"],
            label="已上传文档",
            interactive=False,
        )
        with gr.Row():
            delete_name = gr.Textbox(
                label="输入要删除的文档名称",
                placeholder="例如：report.pdf",
                lines=1,
            )
            delete_btn = gr.Button("🗑️ 删除文档", variant="stop")
        delete_status = gr.Textbox(label="操作结果", interactive=False)
        clear_all_btn = gr.Button("⚠️ 清空所有文档", variant="stop")
        clear_status = gr.Textbox(label="清空结果", interactive=False)

        def refresh_list():
            sources = self.vector_store.get_document_ids()
            if not sources:
                return [["（暂无文档）", "-", "0"]]
            # 统计每个文档的分块数
            rows = []
            for src in sources:
                fmt = Path(src).suffix if Path(src).suffix else "未知"
                rows.append([src, fmt, "—"])
            return rows

        refresh_btn.click(refresh_list, None, doc_list)

        def delete_doc(name):
            if not name:
                return "请输入要删除的文档名称。"
            deleted = self.vector_store.delete_by_source(name)
            if deleted > 0:
                # 重建 TF-IDF
                all_chunks = self._get_all_chunks()
                self.tfidf.build_index(all_chunks)
                return f"✅ 已删除文档「{name}」（{deleted} 个分块）。"
            return f"❌ 未找到文档「{name}」。"

        delete_btn.click(delete_doc, delete_name, delete_status)

        def clear_all():
            count = self.vector_store.delete_all()
            self.tfidf = TfidfRetriever()
            self.retriever.tfidf = self.tfidf
            return f"✅ 已清空所有文档（{count} 个分块）。"

        clear_all_btn.click(clear_all, None, clear_status)

    def _get_all_chunks(self) -> List[dict]:
        """从向量库获取所有 chunk（用于重建 TF-IDF 索引）"""
        try:
            collection = self.vector_store._collection
            count = collection.count()
            if count == 0:
                return []
            data = collection.get(include=["documents", "metadatas"])
            chunks = []
            for i in range(len(data["ids"])):
                chunks.append({
                    "text": data["documents"][i],
                    "metadata": data["metadatas"][i],
                })
            return chunks
        except Exception:
            return []

    def launch(self, **kwargs):
        """启动 Gradio 应用"""
        # 避免与 **kwargs 中的参数冲突
        kwargs.setdefault("server_name", "127.0.0.1")
        kwargs.setdefault("server_port", 7860)
        kwargs.setdefault("share", False)
        self.app.queue().launch(**kwargs)


if __name__ == "__main__":
    ui = AgentUI()
    ui.launch()
