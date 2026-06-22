"""
LLMController: Abstraction layer for LLM backends.

Supports multiple backends:
  - OpenAI (GPT-4o-mini, GPT-4o, etc.)
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
    """OpenAI API controller with structured output support."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
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
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable."
            )
        self.client = OpenAI(api_key=api_key)

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
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content


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
        backend: One of 'openai', 'ollama', 'litellm'.
        model: Model identifier (e.g., 'gpt-4o-mini', 'llama3.2').
        api_key: Optional API key for the backend.
        api_base: Optional API base URL for LiteLLM.
    """

    def __init__(
        self,
        backend: Literal["openai", "ollama", "litellm"] = "openai",
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ):
        if backend == "openai":
            self.llm = OpenAIController(model, api_key)
        elif backend == "ollama":
            self.llm = OllamaController(model)
        elif backend == "litellm":
            self.llm = LiteLLMController(model, api_base, api_key)
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'openai', 'ollama', or 'litellm'.")
