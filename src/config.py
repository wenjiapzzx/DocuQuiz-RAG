"""
环境配置模块
负责初始化环境变量、模型加载和基础设置
"""
import os
import torch
from typing import Optional
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from huggingface_hub import configure_http_backend
from requests.adapters import HTTPAdapter
from requests import Session


def create_http_session() -> Session:
    """创建HTTP会话，用于连接HuggingFace"""
    session = Session()
    adapter = HTTPAdapter(max_retries=3, pool_connections=20, pool_maxsize=100)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def initialize_environment(
    hf_home: Optional[str] = None,
    hf_endpoint: Optional[str] = None,
    cache_folder: Optional[str] = None
) -> None:
    """
    初始化环境配置
    
    Args:
        hf_home: HuggingFace缓存目录
        hf_endpoint: HuggingFace镜像地址
        cache_folder: 模型缓存目录
    """
    # 设置关键环境变量
    os.environ["TORCH_FORCE_WEIGHTS_ONLY"] = "1"
    
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint
    elif "HF_ENDPOINT" not in os.environ:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    if hf_home:
        os.environ["HF_HOME"] = hf_home
    
    # 配置HTTP后端
    configure_http_backend(create_http_session)
    
    print("环境初始化完成！")


def load_embedding_model(
    model_name: str = "BAAI/bge-m3",
    cache_folder: Optional[str] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> HuggingFaceEmbedding:
    """
    加载嵌入模型
    
    Args:
        model_name: 模型名称
        cache_folder: 缓存目录
        device: 设备类型
    
    Returns:
        配置好的嵌入模型
    """
    if cache_folder is None:
        cache_folder = os.environ.get("HF_HOME", "./cache")
    
    embed_model = HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=cache_folder,
        model_kwargs={
            "torch_dtype": torch.float16,
            "device": device
        }
    )
    
    # 设置到LlamaIndex全局配置
    Settings.embed_model = embed_model
    
    print(f"模型加载成功！设备: {device}")
    return embed_model


def get_default_config() -> dict:
    """获取默认配置"""
    return {
        "hf_home": "/root/autodl-tmp/huggingface_cache",
        "hf_endpoint": "https://hf-mirror.com",
        "model_name": "BAAI/bge-m3",
        "vector_store_dir": "/root/autodl-tmp/vector_store",
        "document_cache_dir": "/root/autodl-tmp/vector_store1"
    }