"""
LLMController: Abstraction layer for LLM backends.

Supports multiple backends:
  - OpenAI (GPT-4o-mini, GPT-4o, etc.)
  - Doubao (豆包, via OpenAI-compatible API)
  - Ollama (local models via LiteLLM)
  - SGLang (local inference server)

All backends provide a unified get_completion() interface with
structured JSON output support.
"""

import os
import json
import re
from abc import ABC, abstractmethod
from typing import Optional, Any, Literal

# Auto-load .env
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass


class BaseLLMController(ABC):
    """Abstract base class for LLM controllers."""

    @abstractmethod
    def get_completion(
        self,
        prompt: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.7,
    ) -> str:
        """Get a completion from the LLM.

        Args:
            prompt: The user prompt.
            response_format: JSON schema for structured output.
            temperature: Sampling temperature.

        Returns:
            The LLM response as a string.
        """
        pass


class OpenAIController(BaseLLMController):
    """OpenAI-compatible API controller with structured output support.

    Works with any OpenAI-compatible API (OpenAI, Doubao/豆包, DeepSeek, etc.)
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
                "OpenAI package not found. Install with: pip install openai"
            )

        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not found. Set OPENAI_API_KEY environment variable or pass api_key."
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
                {"role": "system", "content": "You must respond with a JSON object."},
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
                print(f"[Warning] Structured output not supported, falling back to plain prompt. Error: {e}")
                kwargs.pop("response_format", None)
                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            raise


class OllamaController(BaseLLMController):
    """Ollama controller via LiteLLM for local model inference."""

    def __init__(self, model: str = "llama3.2"):
        self.model = model
        if not model.startswith("ollama/"):
            self.model = f"ollama/{model}"

    def _generate_empty_response(self, response_format: Optional[dict]) -> dict:
        """Generate an empty response matching the schema structure."""
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
                        "content": "You must respond with a JSON object.",
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
            print(f"Ollama completion error: {e}")
            return json.dumps(self._generate_empty_response(response_format))


class LiteLLMController(BaseLLMController):
    """Universal LiteLLM controller supporting any backend."""

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
                        "content": "You must respond with a JSON object.",
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
            print(f"LiteLLM completion error: {e}")
            return json.dumps(self._generate_empty_response(response_format))


class LLMController:
    """Unified LLM controller that dispatches to the appropriate backend.

    Args:
        backend: One of 'openai', 'ollama', 'litellm', 'doubao'.
                 Defaults to 'doubao' if DOUBAO_API_KEY is set in env.
        model: Model identifier (e.g., 'gpt-4o-mini', 'llama3.2', 'doubao-seed-2-0-lite-260215').
               For 'doubao' backend, defaults to DOUBAO_MODEL env var.
        api_key: Optional API key for the backend.
        api_base: Optional API base URL (required for 'doubao', optional for others).
    """

    def __init__(
        self,
        backend: Literal["openai", "ollama", "litellm", "doubao"] = None,
        model: str = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        # Auto-detect backend: if DOUBAO_API_KEY is set and no backend specified, use doubao
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
                    "Doubao requires a base_url. Pass api_base or set DOUBAO_BASE_URL env var.\n"
                    "Example: https://ark.cn-beijing.volces.com/api/v3"
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
                f"Unknown backend: {backend}. Use 'openai', 'doubao', 'ollama', or 'litellm'."
            )
