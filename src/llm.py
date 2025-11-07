"""
LLM集成模块
提供DeepSeek API集成和自定义LLM类
"""
import os
import requests
from typing import Dict, Any
from llama_index.core.llms import CustomLLM, CompletionResponse, LLMMetadata
from pydantic import Field, SecretStr
from llama_index.core import Settings


class DeepSeekLLM(CustomLLM):
    """
    DeepSeek API集成的自定义LLM类
    """
    api_key: SecretStr = Field(description="DeepSeek API Key")
    base_url: str = Field(default="https://api.deepseek.com", description="API 基础地址")
    model: str = Field(default="deepseek-chat", description="模型名称")
    context_window: int = Field(default=8192, description="上下文窗口大小")
    
    def __init__(self, **data):
        super().__init__(**data)
    
    @property
    def metadata(self) -> LLMMetadata:
        """返回LLM元数据"""
        return LLMMetadata(
            context_window=self.context_window,
            model_name=self.model,
            is_chat_model=True
        )
    
    def complete(self, prompt: str, **kwargs) -> CompletionResponse:
        """
        完成文本生成请求
        
        Args:
            prompt: 输入提示
            **kwargs: 其他参数
            
        Returns:
            完成响应对象
        """
        headers = {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 1024)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()["choices"][0]["message"]["content"]
            return CompletionResponse(text=result)
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"API请求失败: {e}")
        except KeyError as e:
            raise Exception(f"API响应格式错误: {e}")
    
    def stream_complete(self, prompt: str, **kwargs):
        """流式完成（暂不支持）"""
        raise NotImplementedError("流式完成暂不支持")


def setup_deepseek_llm(
    api_key: str = None,
    base_url: str = "https://api.deepseek.com",
    model: str = "deepseek-chat",
    context_window: int = 8192
) -> DeepSeekLLM:
    """
    设置DeepSeek LLM
    
    Args:
        api_key: API密钥
        base_url: API基础地址
        model: 模型名称
        context_window: 上下文窗口大小
        
    Returns:
        配置好的DeepSeek LLM实例
    """
    if api_key is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError("请设置DEEPSEEK_API_KEY环境变量或传入api_key参数")
    
    llm = DeepSeekLLM(
        api_key=api_key,
        base_url=base_url,
        model=model,
        context_window=context_window
    )
    
    # 设置到LlamaIndex全局配置
    Settings.llm = llm
    
    print(f"DeepSeek LLM设置完成！模型: {model}")
    return llm


def get_llm_config() -> dict:
    """获取LLM配置"""
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-chat",
        "context_window": 8192
    }