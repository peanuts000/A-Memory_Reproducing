"""
Configuration for A-MEM reproduction with DeepSeek backend.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DeepSeekConfig:
    """DeepSeek API configuration."""
    api_key: str = ""
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    temperature: float = 0.7
    max_tokens: int = 1000

    def __post_init__(self):
        # Try to get API key from environment if not provided
        if not self.api_key:
            self.api_key = os.getenv("DEEPSEEK_API_KEY", "")


@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    dataset_path: str = "data/locomo10.json"
    output_dir: str = "results"
    memory_cache_dir: str = "cached_memories"

    # Retrieval settings
    retrieve_k: int = 10
    embedding_model: str = "all-MiniLM-L6-v2"

    # Evaluation settings
    ratio: float = 1.0  # Fraction of dataset to evaluate
    temperature_c5: float = 0.5  # Temperature for category 5 questions

    # DeepSeek settings
    deepseek: DeepSeekConfig = None

    def __post_init__(self):
        if self.deepseek is None:
            self.deepseek = DeepSeekConfig()

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.memory_cache_dir, exist_ok=True)
