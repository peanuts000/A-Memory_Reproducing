"""
LLMController: LLM 后端抽象层。

支持多种后端：
  - OpenAI (GPT-4o-mini, GPT-4o 等)
  - 豆包 Doubao (通过 OpenAI 兼容 API)
  - Ollama (通过 LiteLLM 的本地模型)
  - LiteLLM (通用适配器)
  - SGLang (本地高性能推理服务器)
  - vLLM (本地高性能推理服务器)

所有后端提供统一的 get_completion() 接口，支持结构化 JSON 输出。
"""

import os
import json
import re
import time
import functools
import logging
from abc import ABC, abstractmethod
from typing import Optional, Any, Literal

logger = logging.getLogger("amem")

# 自动加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """LLM 调用重试装饰器，支持指数退避。

    参数:
        max_retries: 最大重试次数（默认 2，即共 3 次尝试）。
        base_delay: 基础延迟秒数（每次重试翻倍）。
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM 调用 %s 失败 (第 %d/%d 次): %s — %.1f 秒后重试",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM 调用 %s 在 %d 次尝试后仍然失败: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


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

    def check_connectivity(self) -> bool:
        """检查 LLM 后端是否可达。

        发送一个简单的测试请求来验证连接。

        返回:
            True 如果连接成功。

        异常:
            ConnectionError: 如果后端不可达。
        """
        try:
            response = self.get_completion(
                "Reply with exactly one word: READY", temperature=0.0
            )
            if not response or not response.strip():
                raise ConnectionError("LLM 后端返回空响应")
            logger.info("LLM 连通性检查通过 (响应: %s)", response.strip()[:50])
            return True
        except Exception as e:
            raise ConnectionError(
                f"无法连接到 LLM 后端: {e}。请检查服务器是否正在运行。"
            ) from e


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

    @retry_llm_call(max_retries=2, base_delay=1.0)
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
                logger.warning("不支持结构化输出，回退到普通提示。错误: %s", e)
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise


class DeepSeekController(BaseLLMController):
    """DeepSeek API 控制器，支持 JSON Mode。

    DeepSeek 使用 response_format={"type": "json_object"} 来启用 JSON Mode，
    并在系统提示中描述期望的 JSON schema。

    参考文档: https://api-docs.deepseek.com/zh-cn/guides/json_mode
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
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
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "未找到 DeepSeek API Key。请设置 DEEPSEEK_API_KEY 环境变量或传入 api_key 参数。\n"
                "获取地址: https://platform.deepseek.com/api_keys"
            )

        if base_url is None:
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url

    def _build_system_prompt(self, response_format: Optional[dict] = None) -> str:
        """构建系统提示，包含 JSON schema 描述（如果有）。"""
        base_prompt = "你必须以 JSON 对象格式回复。"

        if response_format and "json_schema" in response_format:
            schema = response_format["json_schema"].get("schema", {})
            if schema:
                # 将 schema 描述添加到系统提示中
                schema_desc = json.dumps(schema, ensure_ascii=False, indent=2)
                base_prompt += f"\n\n请严格按照以下 JSON schema 格式回复:\n{schema_desc}"
                # 如果有 description，也添加进去
                if "description" in schema:
                    base_prompt += f"\n\n描述: {schema['description']}"

        return base_prompt

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        # 构建包含 schema 的系统提示
        system_prompt = self._build_system_prompt(response_format)

        kwargs = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 4096,
        }

        # DeepSeek 使用 {"type": "json_object"} 启用 JSON Mode
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            # 如果 JSON Mode 失败，回退到普通模式
            if "response_format" in error_msg or "json" in error_msg:
                logger.warning("DeepSeek JSON Mode 失败，回退到普通模式。错误: %s", e)
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

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
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

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
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


class SGLangController(BaseLLMController):
    """SGLang 推理服务器控制器。

    SGLang 是一个高性能的本地推理引擎，支持结构化 JSON 输出。
    适用于本地部署的大模型推理。

    参数:
        model: 模型名称。
        host: SGLang 服务器地址（默认 http://localhost）。
        port: SGLang 服务器端口（默认 30000）。
    """

    def __init__(
        self,
        model: str = "llama2",
        host: str = "http://localhost",
        port: int = 30000,
    ):
        self.model = model
        self.base_url = f"{host}:{port}"

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        import requests

        # 提取 JSON schema（SGLang 接受字符串格式的 schema）
        json_schema_str = None
        if response_format and "json_schema" in response_format:
            json_schema = response_format["json_schema"].get("schema", {})
            json_schema_str = json.dumps(json_schema)

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 2000,
            },
        }
        if json_schema_str:
            payload["sampling_params"]["json_schema"] = json_schema_str

        response = requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(
            f"SGLang 服务器返回状态码 {response.status_code}: {response.text}"
        )


class VLLMController(BaseLLMController):
    """vLLM 推理服务器控制器。

    vLLM 是一个高性能的本地推理引擎，提供 OpenAI 兼容的 API。
    适用于本地部署的大模型推理。

    参数:
        model: 模型名称。
        host: vLLM 服务器地址（默认 http://localhost）。
        port: vLLM 服务器端口（默认 8000）。
    """

    def __init__(
        self,
        model: str = "llama2",
        host: str = "http://localhost",
        port: int = 8000,
    ):
        self.model = model
        self.base_url = f"{host}:{port}"

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        import requests

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "你必须以 JSON 对象格式回复。"},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2000,
        }

        # vLLM 支持 guided_json 进行结构化输出
        if response_format and "json_schema" in response_format:
            json_schema = response_format["json_schema"].get("schema", {})
            payload["guided_json"] = json_schema

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(
            f"vLLM 服务器返回状态码 {response.status_code}: {response.text}"
        )


class LLMController:
    """统一的 LLM 控制器，分发到相应的后端。

    参数:
        backend: 'openai', 'ollama', 'litellm', 'doubao', 'deepseek', 'sglang', 'vllm' 之一。
                 如果环境变量中设置了 DOUBAO_API_KEY，则默认使用 'doubao'。
        model: 模型标识符（如 'gpt-4o-mini', 'llama3.2', 'doubao-seed-2-0-lite-260215'）。
               对于 'doubao' 后端，默认使用 DOUBAO_MODEL 环境变量。
        api_key: 可选的后端 API Key。
        api_base: 可选的 API Base URL（'doubao' 必需，其他可选）。
        sglang_host: SGLang 服务器地址（默认 http://localhost）。
        sglang_port: SGLang 服务器端口（默认 30000）。
        vllm_host: vLLM 服务器地址（默认 http://localhost）。
        vllm_port: vLLM 服务器端口（默认 8000）。
    """

    def __init__(
        self,
        backend: Literal["openai", "ollama", "litellm", "doubao", "deepseek", "sglang", "vllm"] = None,
        model: str = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        vllm_host: str = "http://localhost",
        vllm_port: int = 8000,
        check_connection: bool = False,
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
        elif backend == "deepseek":
            # DeepSeek 使用专用控制器，支持 JSON Mode
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek 需要 API Key。请传入 api_key 或设置 DEEPSEEK_API_KEY 环境变量。\n"
                    "获取地址: https://platform.deepseek.com/api_keys"
                )
            if not api_base:
                api_base = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
            if model is None:
                model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
            self.llm = DeepSeekController(model, api_key, base_url=api_base)
        elif backend == "ollama":
            if model is None:
                model = "llama3.2"
            self.llm = OllamaController(model)
        elif backend == "litellm":
            if model is None:
                model = "gpt-4o-mini"
            self.llm = LiteLLMController(model, api_base, api_key)
        elif backend == "sglang":
            if model is None:
                model = "llama2"
            self.llm = SGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            if model is None:
                model = "llama2"
            self.llm = VLLMController(model, vllm_host, vllm_port)
        else:
            raise ValueError(
                f"未知后端: {backend}。请使用 'openai', 'doubao', 'deepseek', 'ollama', 'litellm', 'sglang' 或 'vllm'。"
            )

        # 可选的连通性检查
        if check_connection:
            self.llm.check_connectivity()
