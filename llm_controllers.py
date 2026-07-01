"""
LLM Controllers for A-MEM reproduction.
Supports DeepSeek and OpenAI-compatible APIs.
"""

import os
import json
import time
import logging
import functools
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

logger = logging.getLogger("amem")


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 3, base_delay: float = 1.0):
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
# Base Controller
# ---------------------------------------------------------------------------

class BaseLLMController(ABC):
    """Base class for LLM controllers."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


# ---------------------------------------------------------------------------
# DeepSeek Controller (OpenAI-compatible)
# ---------------------------------------------------------------------------

class DeepSeekController(BaseLLMController):
    """Controller for DeepSeek API (OpenAI-compatible)."""

    def __init__(self, model: str = "deepseek-v4-flash",
                 api_key: Optional[str] = None,
                 base_url: str = "https://api.deepseek.com"):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

        self.model = model

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        logger.info("Initialized DeepSeek controller with model: %s", model)

    @retry_llm_call(max_retries=3)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# OpenAI Controller
# ---------------------------------------------------------------------------

class OpenAIController(BaseLLMController):
    """Controller for OpenAI API."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")

        self.model = model
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)

    @retry_llm_call(max_retries=3)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=1000,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Controller Factory
# ---------------------------------------------------------------------------

class LLMControllerFactory:
    """Factory for creating LLM controllers."""

    @staticmethod
    def create(backend: str, model: str, **kwargs) -> BaseLLMController:
        """
        Create an LLM controller.

        Args:
            backend: One of "deepseek", "openai"
            model: Model name
            **kwargs: Additional arguments for the controller

        Returns:
            BaseLLMController instance
        """
        if backend == "deepseek":
            return DeepSeekController(
                model=model,
                api_key=kwargs.get("api_key"),
                base_url=kwargs.get("base_url", "https://api.deepseek.com"),
            )
        elif backend == "openai":
            return OpenAIController(
                model=model,
                api_key=kwargs.get("api_key"),
            )
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'deepseek' or 'openai'.")
