# AI 智能助手

基于大模型 API + RAG 技术的智能问答助手，支持纯对话模式和文档增强问答。

## 功能特性

- **💬 纯对话模式** — 直接调用大模型 API 进行对话
- **📚 RAG 智能问答** — 上传文档后基于文档内容进行问答
  - NLP 语义分割：使用 Sentence-Transformers 语义分析自动检测主题边界
  - 混合检索：语义检索（向量相似度）+ 精准检索（TF-IDF 关键词匹配）加权融合
  - 原文定位：检索结果标注来源文档及位置
  - 关键词高亮：结果中高亮匹配的关键词
- **📁 文档管理** — 支持上传/查看/删除文档（txt/pdf/docx）
- **⚙️ 参数可调** — 检索数量、权重、LLM 温度等均可实时调整
- **🔌 多模型支持** — 兼容所有 OpenAI 标准接口（OpenAI / DeepSeek / 通义千问 / 硅基流动等）

## 项目结构

```
agent_project/
├── config/
│   ├── __init__.py
│   ├── api.ini              # LLM API 密钥/地址/模型（用户需编辑此文件）
│   ├── api_config.py        # LLM API 配置读取模块
│   └── settings.py          # 全局配置管理
├── agent/
│   ├── __init__.py
│   ├── llm_client.py        # LLM API 调用封装
│   └── chat_agent.py        # 对话 Agent
├── rag/
│   ├── __init__.py
│   ├── document_loader.py   # 文档加载器（txt/pdf/docx）
│   ├── text_splitter.py     # NLP 语义分割器
│   ├── vector_store.py      # ChromaDB 向量数据库管理
│   ├── retriever.py         # 混合检索器（语义+TF-IDF）
│   └── rag_chain.py         # RAG 问答链路
├── app/
│   ├── __init__.py
│   └── gradio_ui.py         # Gradio 界面
├── data/                    # 文档及数据库存储目录
├── main.py                  # 启动入口
├── .env                     # 配置文件（需手动填写 API Key）
├── .env.example             # 配置模板
├── requirements.txt         # 依赖清单
└── README.md                # 本文件
```

## 快速开始

### 1. 进入环境

```bash
# 使用已有的 conda 环境
D:\Miniconda3\envs\agent\python.exe main.py
```

### 2. 配置 API Key

编辑 `config/api.ini` 文件：

```ini
[llm]
; OpenAI 直接使用
api_key = sk-your-key-here
base_url = https://api.openai.com/v1
model = gpt-4o-mini

; 使用 DeepSeek（取消注释下方配置）
; api_key = sk-deepseek-key
; base_url = https://api.deepseek.com/v1
; model = deepseek-chat

; 使用 通义千问（取消注释下方配置）
; api_key = sk-qwen-key
; base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
; model = qwen-plus
```

> ⚠️ **注意**：LLM API 的密钥/地址/模型已独立到 `config/api.ini` 文件中管理，与 `.env` 中的其他配置分离。请勿将 `api.ini` 提交到版本控制系统。

### 3. 启动

```bash
python main.py
# 或指定端口
python main.py --port 7861
# 或生成公网分享链接（注意数据安全）
python main.py --share
```

浏览器访问 `http://127.0.0.1:7860` 即可使用。

## 技术方案

### 架构：模块化单体

各部分独立模块，通过统一接口协作，兼顾开发效率与可维护性。

### 文档分割：NLP 语义分割

1. 将文本切分为句子
2. 使用 Sentence-Transformers（BAAI/bge-small-zh-v1.5）计算句子嵌入
3. 检测相邻句子间的余弦相似度，低于阈值处识别为主题边界
4. 在边界处分块，并约束块大小在合理范围内

### 向量化：本地 Sentence-Transformers

使用本地模型 `BAAI/bge-small-zh-v1.5` 将文本转为向量，无需联网，无需额外费用。

### 向量数据库：ChromaDB

轻量级本地向量数据库，支持持久化存储和高效相似度检索。

### 精准匹配：TF-IDF

基于 scikit-learn 的 TfidfVectorizer（字符级别 n-gram），实现关键词级别的精确匹配。

### 混合检索

语义检索得分 × 语义权重 + TF-IDF 得分 × 精确权重，按综合得分排序。

### UI：Gradio

Python 原生 Web UI 框架，快速构建交互界面。

## 依赖环境

本项目在 Python 3.10 下开发。核心依赖均已安装在 `agent` 环境中：

| 包 | 用途 |
|------|------|
| openai | LLM API 调用 |
| chromadb | 向量数据库 |
| sentence-transformers | 文本向量化 / 语义分割 |
| transformers | 模型支持 |
| torch | 深度学习框架 |
| scikit-learn | TF-IDF 检索 |
| gradio | Web UI |
| python-docx | Word 文档解析 |
| pypdf2 | PDF 文档解析 |
| python-dotenv | 环境配置管理 |
| tiktoken | Token 计数（可选） |

如有需要安装的包：

```bash
D:\Miniconda3\envs\agent\python.exe -m pip install tiktoken
```

## 使用说明

### 纯对话

1. 切换到「纯对话」标签页
2. 在输入框输入问题，点击发送
3. 支持多轮对话上下文

### RAG 智能问答

1. 切换到「RAG 智能问答」标签页
2. 点击「上传文档」选择文件（支持 txt / pdf / docx）
3. 等待文档处理完成（会显示分块数量）
4. 在输入框提问，系统会基于文档内容回答
5. 展开「检索来源」可查看引用的原文位置

### 文档管理

1. 切换到「文档管理」标签页
2. 可查看已上传的文档列表
3. 支持按名称删除单个文档或清空所有文档
