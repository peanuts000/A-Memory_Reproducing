"""
LLMController: LLM 后端抽象层。

支持多种后端：
  - OpenAI (GPT-4o-mini, GPT-4o 等)
  - 豆包 Doubao (通过 OpenAI 兼容 API)
  - Ollama (通过 LiteLLM 的本地模型)
  - LiteLLM (通用适配器)

所有后端提供统一的 get_completion() 接口，支持结构化 JSON 输出。
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Any, Literal

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


class BaseLLMController(ABC):
    """LLM 控制器的抽象基类。"""

    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """从 LLM 获取补全结果。

        参数:
            prompt: 用户提示词。
            response_format: 结构化输出的 JSON schema。
            temperature: 采样温度。

        返回:
            LLM 响应字符串。
        """
        pass


class OpenAIController(BaseLLMController):
    """OpenAI 兼容 API 控制器，支持结构化输出。

    可与任何 OpenAI 兼容 API 配合使用（OpenAI、豆包、DeepSeek 等），
    通过指定自定义 base_url 实现。
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "未找到 OpenAI 包。请使用以下命令安装: pip install openai"
            )

        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "未找到 API Key。请设置 OPENAI_API_KEY 环境变量或传入 api_key 参数。"
            )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)
        self.base_url = base_url

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你必须以 JSON 对象格式回复。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2000,
        }

        # 优先尝试结构化输出；如果不支持则回退到普通提示
        use_structured = response_format is not None
        if use_structured:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            # 如果不支持结构化输出，则不使用它重试
            if use_structured and ("response_format" in error_msg or "json_schema" in error_msg or "unsupported" in error_msg):
                print(f"[警告] 不支持结构化输出，回退到普通提示。错误: {e}")
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise


class OllamaController(BaseLLMController):
    """通过 LiteLLM 的 Ollama 控制器，用于本地模型推理。"""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        if not model.startswith("ollama/"):
            self.model = f"ollama/{model}"

    def _generate_empty_response(self, response_format: Optional[dict]) -> dict:
        """生成匹配 schema 结构的空响应。"""
        if not response_format or "json_schema" not in response_format:
            return {}

        schema = response_format["json_schema"].get("schema", {})
        result = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                ptype = prop_schema.get("type", "string")
                if ptype == "array":
                    result[prop_name] = []
                elif ptype == "string":
                    result[prop_name] = ""
                elif ptype == "boolean":
                    result[prop_name] = False
                elif ptype in ("number", "integer"):
                    result[prop_name] = 0
                elif ptype == "object":
                    result[prop_name] = {}
        return result

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        try:
            from litellm import completion

            kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你必须以 JSON 对象格式回复。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if response_format:
                kwargs["response_format"] = response_format

            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"Ollama 补全错误: {e}")
            return json.dumps(self._generate_empty_response(response_format))


class LiteLLMController(BaseLLMController):
    """通用 LiteLLM 控制器，支持任意后端。"""

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

    def _generate_empty_response(self, response_format: Optional[dict]) -> dict:
        if not response_format or "json_schema" not in response_format:
            return {}
        schema = response_format["json_schema"].get("schema", {})
        result = {}
        if "properties" in schema:
            for prop_name, prop_schema in schema["properties"].items():
                ptype = prop_schema.get("type", "string")
                if ptype == "array":
                    result[prop_name] = []
                elif ptype == "string":
                    result[prop_name] = ""
                elif ptype == "boolean":
                    result[prop_name] = False
                elif ptype in ("number", "integer"):
                    result[prop_name] = 0
        return result

    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        try:
            from litellm import completion

            kwargs = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "你必须以 JSON 对象格式回复。",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": temperature,
            }
            if self.api_base:
                kwargs["api_base"] = self.api_base
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if response_format:
                kwargs["response_format"] = response_format

            response = completion(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            print(f"LiteLLM 补全错误: {e}")
            return json.dumps(self._generate_empty_response(response_format))


class LLMController:
    """统一的 LLM 控制器，分发到相应的后端。

    参数:
        backend: 'openai', 'ollama', 'litellm', 'doubao' 之一。
                 如果环境变量中设置了 DOUBAO_API_KEY，则默认使用 'doubao'。
        model: 模型标识符（如 'gpt-4o-mini', 'llama3.2', 'doubao-seed-2-0-lite-260215'）。
               对于 'doubao' 后端，默认使用 DOUBAO_MODEL 环境变量。
        api_key: 可选的后端 API Key。
        api_base: 可选的 API Base URL（'doubao' 必需，其他可选）。
    """

    def __init__(
        self,
        backend: Literal["openai", "ollama", "litellm", "doubao"] = None,
        model: str = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        # 自动检测后端：如果设置了 DOUBAO_API_KEY 且未指定后端，则使用 doubao
        if backend is None:
            if os.getenv("DOUBAO_API_KEY"):
                backend = "doubao"
            elif os.getenv("OPENAI_API_KEY"):
                backend = "openai"
            else:
                backend = "openai"

        if backend == "openai":
            if model is None:
                model = "gpt-4o-mini"
            self.llm = OpenAIController(model, api_key, base_url=api_base)
        elif backend == "doubao":
            # 豆包使用 OpenAI 兼容 API
            if not api_key:
                api_key = os.getenv("DOUBAO_API_KEY")
            if not api_base:
                api_base = os.getenv("DOUBAO_BASE_URL")
            if not api_base:
                raise ValueError(
                    "豆包需要 base_url。请传入 api_base 或设置 DOUBAO_BASE_URL 环境变量。\n"
                    "示例: https://ark.cn-beijing.volces.com/api/v3"
                )
            if model is None:
                model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215")
            self.llm = OpenAIController(model, api_key, base_url=api_base)
        elif backend == "ollama":
            if model is None:
                model = "llama3.2"
            self.llm = OllamaController(model)
        elif backend == "litellm":
            if model is None:
                model = "gpt-4o-mini"
            self.llm = LiteLLMController(model, api_base, api_key)
        else:
            raise ValueError(
                f"未知后端: {backend}。请使用 'openai', 'doubao', 'ollama' 或 'litellm'。"
            )
