"""
A-Mem: 面向 LLM Agent 的智能记忆系统

本项目是对论文 "A-Mem: Agentic Memory for LLM Agents" (arXiv:2502.12110) 的忠实复现。

基于 Zettelkasten 方法，本系统通过以下三个核心机制实现 LLM Agent 的动态记忆管理：
  1. 笔记构建 (Note Construction) - 使用 LLM 生成结构化记忆笔记
  2. 链接生成 (Link Generation) - 在相关记忆之间建立连接
  3. 记忆演化 (Memory Evolution) - 根据新信息动态更新已有记忆
"""

# 导入时自动加载 .env 文件
try:
    from dotenv import load_dotenv
    import os
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))
except ImportError:
    pass

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
