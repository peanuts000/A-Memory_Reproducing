"""
A-Mem: Agentic Memory for LLM Agents
A faithful reproduction of the paper: "A-Mem: Agentic Memory for LLM Agents"
(arXiv:2502.12110)

Based on the Zettelkasten method, this system enables dynamic memory structuring
for LLM agents through three core mechanisms:
  1. Note Construction - Generate structured memory notes with LLM
  2. Link Generation - Establish connections between related memories
  3. Memory Evolution - Dynamically update existing memories with new context
"""

from .memory_note import MemoryNote
from .retriever import SimpleEmbeddingRetriever
from .llm_controller import LLMController, BaseLLMController
from .agentic_memory import AgenticMemorySystem

__version__ = "1.0.0"
__all__ = [
    "MemoryNote",
    "SimpleEmbeddingRetriever",
    "LLMController",
    "BaseLLMController",
    "AgenticMemorySystem",
]
