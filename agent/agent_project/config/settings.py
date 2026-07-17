"""全局配置管理"""
import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


class EmbeddingSettings:
    """Embedding 模型配置 — 本地 Sentence-Transformers"""
    # 如果网络能访问 HuggingFace，可以改回 remote 模型名如 "BAAI/bge-small-zh-v1.5"
    # 本地路径优先
    _local_path = ROOT_DIR / "models" / "bge-small-zh-v1.5"
    if _local_path.exists():
        MODEL_NAME: str = str(_local_path)
    else:
        MODEL_NAME: str = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")


class ChromaSettings:
    """ChromaDB 向量数据库配置"""
    PERSIST_PATH: str = os.getenv("CHROMA_DB_PATH", str(ROOT_DIR / "data" / "chromadb"))


class SplitterSettings:
    """NLP语义分割参数"""
    MIN_SIZE: int = int(os.getenv("CHUNK_MIN_SIZE", "50"))
    MAX_SIZE: int = int(os.getenv("CHUNK_MAX_SIZE", "500"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.5"))


class RetrievalSettings:
    """检索参数"""
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    SEMANTIC_WEIGHT: float = float(os.getenv("SEMANTIC_WEIGHT", "0.7"))
    EXACT_WEIGHT: float = float(os.getenv("EXACT_WEIGHT", "0.3"))
