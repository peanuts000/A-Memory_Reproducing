"""
基础演示：A-Mem 智能记忆系统

本脚本演示 A-Mem 系统的核心功能：
  1. 添加记忆笔记（触发笔记构建 + 链接生成 + 记忆演化）
  2. 检索相关记忆
  3. 检查记忆结构

使用方法：
    # 自动从 .env 检测配置（推荐）:
    python examples/basic_demo.py

    # 使用豆包:
    python examples/basic_demo.py --backend doubao

    # 使用 OpenAI:
    python examples/basic_demo.py --backend openai --api-key sk-xxx

    # 使用 Ollama:
    python examples/basic_demo.py --backend ollama --model llama3.2
"""

import argparse
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem


def main():
    # 从 .env 自动检测默认配置
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215") if default_backend == "doubao" else "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="A-Mem 基础演示")
    parser.add_argument(
        "--backend",
        type=str,
        default=default_backend,
        choices=["openai", "ollama", "litellm", "doubao"],
        help="要使用的 LLM 后端（从 .env 自动检测）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=default_model,
        help="LLM 模型标识符（从 .env 自动检测）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API Key（如未提供则从 .env 自动读取）",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="API Base URL（如未提供则从 .env 自动读取）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("A-Mem: 智能记忆系统 - 基础演示")
    print("=" * 60)
    print(f"后端: {args.backend}, 模型: {args.model}")
    print()

    # 初始化记忆系统（api_key/api_base 为 None 时会自动从 .env 读取）
    memory_system = AgenticMemorySystem(
        llm_backend=args.backend,
        llm_model=args.model,
        api_key=args.api_key,
        api_base=args.base_url,
        top_k=5,
    )

    # -----------------------------------------------------------------------
    # 步骤 1：添加记忆笔记
    # -----------------------------------------------------------------------
    print("步骤 1: 添加记忆笔记...")
    print("-" * 40)

    conversations = [
        "Speaker Dave says: Hey Calvin, long time no talk! A lot has happened. I've taken up photography and it's been great - been taking pics of the scenery around here which is really cool.",
        "Speaker Calvin says: Thanks, Dave! It feels great having my own space to work in. I've been experimenting with different genres lately, pushing myself out of my comfort zone. Adding electronic elements to my songs gives them a fresh vibe. It's been an exciting process of self-discovery and growth!",
        "Speaker Dave says: I also started learning about machine learning. The math is challenging but I enjoy the problem-solving aspect. I've been working through some online courses.",
        "Speaker Calvin says: That's awesome! I've been thinking about using AI to help with music composition. Maybe we could collaborate on something that combines both our interests.",
        "Speaker Dave says: Great idea! I could use my photography skills to create visual content while you work on the music. We could create a multimedia project.",
    ]

    memory_ids = []
    for i, conv in enumerate(conversations):
        print(f"\n添加记忆 {i + 1}/{len(conversations)}...")
        note_id = memory_system.add_note(conv)
        memory_ids.append(note_id)
        print(f"  ID: {note_id[:8]}...")

    print(f"\n已存储记忆总数: {memory_system.get_memory_count()}")

    # -----------------------------------------------------------------------
    # 步骤 2：检索记忆
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 2: 检索记忆...")
    print("-" * 40)

    queries = [
        "Dave 有什么爱好?",
        "Calvin 在做什么?",
        "告诉我他们的合作计划",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = memory_system.retrieve(query, k=3)
        for j, mem in enumerate(results):
            print(f"  [{j + 1}] 内容: {mem.content[:80]}...")
            print(f"      上下文: {mem.context}")
            print(f"      关键词: {mem.keywords}")
            print(f"      标签: {mem.tags}")

    # -----------------------------------------------------------------------
    # 步骤 3：检查记忆结构
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("步骤 3: 检查记忆结构...")
    print("-" * 40)

    for mid in memory_ids:
        note = memory_system.memories[mid]
        print(f"\n记忆: {note.content[:60]}...")
        print(f"  关键词: {note.keywords}")
        print(f"  标签: {note.tags}")
        print(f"  上下文: {note.context}")
        print(f"  链接: {note.links}")

    print("\n" + "=" * 60)
    print("演示完成!")


if __name__ == "__main__":
    main()
