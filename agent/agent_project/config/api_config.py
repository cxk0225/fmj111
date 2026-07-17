"""LLM API 配置管理

从 config/api.ini 读取大模型的 API 密钥、地址和模型名称。
"""
import configparser
from pathlib import Path


# api.ini 文件路径
_API_INI_PATH = Path(__file__).resolve().parent / "api.ini"


def _load_ini() -> configparser.SectionProxy:
    """读取 api.ini 文件，返回 [llm] 节"""
    if not _API_INI_PATH.exists():
        raise FileNotFoundError(
            f"API 配置文件不存在: {_API_INI_PATH}\n"
            f"请复制 config/api.ini.example 为 config/api.ini 并填写你的 API Key。"
        )
    parser = configparser.ConfigParser()
    parser.read(_API_INI_PATH, encoding="utf-8")
    if "llm" not in parser:
        raise KeyError(f"api.ini 中缺少 [llm] 配置节，请检查文件格式。")
    return parser["llm"]


class LLMConfig:
    """大模型 API 配置"""

    def __init__(self):
        section = _load_ini()
        self.API_KEY: str = section.get("api_key", "")
        self.BASE_URL: str = section.get("base_url", "https://api.openai.com/v1")
        self.MODEL: str = section.get("model", "gpt-4o-mini")


# 全局单例
llm_config = LLMConfig()
