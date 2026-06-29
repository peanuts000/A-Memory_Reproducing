"""
LLMController: LLM backend abstraction layer.

Supports multiple backends:
  - OpenAI (GPT-4o-mini, GPT-4o, etc.)
  - Doubao (via OpenAI-compatible API)
  - Ollama (local models via direct ollama library)
  - LiteLLM (universal adapter)
  - SGLang (local high-performance inference server)
  - vLLM (local high-performance inference server)

All backends provide a unified get_completion() interface.
Structured JSON output is attempted when supported; plain-text fallback is used otherwise.
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

# Auto-load .env file
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
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
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Base controller
# ---------------------------------------------------------------------------

class BaseLLMController(ABC):
    """Abstract base class for LLM controllers."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """Get a completion from the LLM.

        Args:
            prompt: User prompt.
            response_format: Optional JSON schema for structured output.
            temperature: Sampling temperature.

        Returns:
            LLM response string.
        """
        pass

    def check_connectivity(self) -> bool:
        """Send a test call to verify the backend is reachable.

        Returns:
            True if connection succeeds.

        Raises:
            ConnectionError: If the backend is unreachable.
        """
        try:
            response = self.get_completion(
                "Reply with exactly one word: READY", temperature=0.0
            )
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
            return True
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


# ---------------------------------------------------------------------------
# OpenAI-compatible controller
# ---------------------------------------------------------------------------

class OpenAIController(BaseLLMController):
    """OpenAI-compatible API controller with structured output support.

    Works with any OpenAI-compatible API (OpenAI, Doubao, DeepSeek, etc.)
    by specifying a custom base_url.
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
                "OpenAI package not found. Install it with: pip install openai"
            )

        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY env var or pass api_key parameter."
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
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2000,
        }

        # Try structured output first; fall back to plain prompt if unsupported
        use_structured = response_format is not None
        if use_structured:
            kwargs["response_format"] = response_format

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            # If structured output is not supported, retry without it
            if use_structured and ("response_format" in error_msg or "json_schema" in error_msg or "unsupported" in error_msg):
                logger.warning("Structured output not supported, falling back to plain prompt. Error: %s", e)
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise


# ---------------------------------------------------------------------------
# DeepSeek controller
# ---------------------------------------------------------------------------

class DeepSeekController(BaseLLMController):
    """DeepSeek API controller with JSON Mode.

    DeepSeek uses response_format={"type": "json_object"} to enable JSON Mode,
    with the expected JSON schema described in the system prompt.

    Reference: https://api-docs.deepseek.com/zh-cn/guides/json_mode
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
                "OpenAI package not found. Install it with: pip install openai"
            )

        self.model = model
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if api_key is None:
            raise ValueError(
                "DeepSeek API key not found. Set DEEPSEEK_API_KEY env var or pass api_key parameter.\n"
                "Get one at: https://platform.deepseek.com/api_keys"
            )

        if base_url is None:
            base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url

    def _build_system_prompt(self, response_format: Optional[dict] = None) -> str:
        """Build system prompt with JSON schema description (if any)."""
        base_prompt = self.SYSTEM_MESSAGE

        if response_format and "json_schema" in response_format:
            schema = response_format["json_schema"].get("schema", {})
            if schema:
                schema_desc = json.dumps(schema, ensure_ascii=False, indent=2)
                base_prompt += f"\n\nRespond strictly in the following JSON schema format:\n{schema_desc}"
                if "description" in schema:
                    base_prompt += f"\n\nDescription: {schema['description']}"

        return base_prompt

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        # Build system prompt with schema description
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

        # DeepSeek uses {"type": "json_object"} to enable JSON Mode
        if response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            error_msg = str(e).lower()
            # If JSON Mode fails, fall back to plain mode
            if "response_format" in error_msg or "json" in error_msg:
                logger.warning("DeepSeek JSON Mode failed, falling back to plain mode. Error: %s", e)
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise


# ---------------------------------------------------------------------------
# Ollama controller
# ---------------------------------------------------------------------------

class OllamaController(BaseLLMController):
    """Ollama controller using direct ollama library (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model

    @retry_llm_call(max_retries=2, base_delay=1.0)
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")

        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


# ---------------------------------------------------------------------------
# LiteLLM controller
# ---------------------------------------------------------------------------

class LiteLLMController(BaseLLMController):
    """LiteLLM controller for universal LLM access."""

    def __init__(
        self,
        model: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_base = api_base
        self.api_key = api_key or "EMPTY"

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
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key

        response = completion(**kwargs)
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# SGLang controller
# ---------------------------------------------------------------------------

class SGLangController(BaseLLMController):
    """SGLang inference server controller.

    SGLang is a high-performance local inference engine.
    No JSON schema in payload — uses plain-text prompts only.

    Args:
        model: Model name.
        host: SGLang server address (default http://localhost).
        port: SGLang server port (default 30000).
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

        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 2000,
            },
        }

        response = requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(
            f"SGLang server returned status {response.status_code}: {response.text}"
        )


# ---------------------------------------------------------------------------
# vLLM controller
# ---------------------------------------------------------------------------

class VLLMController(BaseLLMController):
    """vLLM inference server controller.

    vLLM provides an OpenAI-compatible API at /v1/chat/completions.

    Args:
        model: Model name.
        host: vLLM server address (default http://localhost).
        port: vLLM server port (default 8000).
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
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 2000,
        }

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(
            f"vLLM server returned status {response.status_code}: {response.text}"
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class LLMController:
    """Unified LLM controller factory.

    Selects the appropriate backend based on the backend parameter.
    Auto-detects from environment variables if not specified.

    Args:
        backend: 'openai', 'ollama', 'litellm', 'doubao', 'deepseek', 'sglang', 'vllm'.
                 Auto-detects from env vars if None.
        model: Model identifier (e.g. 'gpt-4o-mini', 'llama3.2').
        api_key: Optional backend API key.
        api_base: Optional API base URL.
        sglang_host: SGLang server address (default http://localhost).
        sglang_port: SGLang server port (default 30000).
        vllm_host: vLLM server address (default http://localhost).
        vllm_port: vLLM server port (default 8000).
        check_connection: If True, verify connectivity on init.
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
        # Auto-detect backend from env vars
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
            # Doubao uses OpenAI-compatible API
            if not api_key:
                api_key = os.getenv("DOUBAO_API_KEY")
            if not api_base:
                api_base = os.getenv("DOUBAO_BASE_URL")
            if not api_base:
                raise ValueError(
                    "Doubao requires base_url. Pass api_base or set DOUBAO_BASE_URL env var.\n"
                    "Example: https://ark.cn-beijing.volces.com/api/v3"
                )
            if model is None:
                model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215")
            self.llm = OpenAIController(model, api_key, base_url=api_base)
        elif backend == "deepseek":
            if not api_key:
                api_key = os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                raise ValueError(
                    "DeepSeek requires API key. Pass api_key or set DEEPSEEK_API_KEY env var.\n"
                    "Get one at: https://platform.deepseek.com/api_keys"
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
                f"Unknown backend: {backend}. Use 'openai', 'doubao', 'deepseek', "
                f"'ollama', 'litellm', 'sglang', or 'vllm'."
            )

        # Optional connectivity check
        if check_connection:
            self.llm.check_connectivity()
