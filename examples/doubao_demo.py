"""
豆包 (Doubao) Demo: A-Mem with Doubao-Seed-2.0

使用火山引擎的豆包模型运行 A-Mem 系统。

前置条件:
  1. 拥有火山引擎 Ark API Key
  2. 已在 .env 文件中配置 DOUBAO_API_KEY, DOUBAO_BASE_URL, DOUBAO_MODEL

Usage:
    # 直接运行（从 .env 自动读取配置）:
    python examples/doubao_demo.py

    # 或命令行覆盖:
    python examples/doubao_demo.py --api-key "你的Key" --model "你的模型"
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem


def main():
    parser = argparse.ArgumentParser(description="A-Mem with Doubao")
    parser.add_argument("--api-key", type=str, default=None, help="Doubao API Key")
    parser.add_argument("--base-url", type=str, default=None, help="Doubao API Base URL")
    parser.add_argument("--model", type=str, default=None, help="Doubao model name")
    args = parser.parse_args()

    # 从参数或 .env 环境变量读取配置
    api_key = args.api_key or os.getenv("DOUBAO_API_KEY")
    base_url = args.base_url or os.getenv("DOUBAO_BASE_URL")
    model = args.model or os.getenv("DOUBAO_MODEL")

    if not api_key:
        print("错误: 请提供 Doubao API Key")
        print("  方法 A: python examples/doubao_demo.py --api-key 你的key")
        print("  方法 B: set DOUBAO_API_KEY=你的key")
        return

    print("=" * 60)
    print("A-Mem: Agentic Memory - 豆包 (Doubao) Demo")
    print("=" * 60)
    print(f"Model:    {model}")
    print(f"Base URL: {base_url}")
    print()

    # 初始化记忆系统
    memory_system = AgenticMemorySystem(
        llm_backend="doubao",
        llm_model=model,
        api_key=api_key,
        api_base=base_url,
        top_k=5,
    )

    # 添加记忆
    print("Step 1: 添加记忆笔记...")
    print("-" * 40)

    conversations = [
        "Speaker Dave says: Hey Calvin, long time no talk! A lot has happened. I've taken up photography and it's been great - been taking pics of the scenery around here which is really cool.",
        "Speaker Calvin says: Thanks, Dave! It feels great having my own space to work in. I've been experimenting with different genres lately, pushing myself out of my comfort zone.",
        "Speaker Dave says: I also started learning about machine learning. The math is challenging but I enjoy the problem-solving aspect.",
        "Speaker Calvin says: That's awesome! I've been thinking about using AI to help with music composition. Maybe we could collaborate.",
        "Speaker Dave says: Great idea! I could use my photography skills to create visual content while you work on the music.",
    ]

    for i, conv in enumerate(conversations):
        print(f"\n添加记忆 {i + 1}/{len(conversations)}...")
        note_id = memory_system.add_note(conv)
        note = memory_system.memories[note_id]
        print(f"  ID:       {note_id[:12]}...")
        print(f"  Keywords: {note.keywords}")
        print(f"  Tags:     {note.tags}")
        print(f"  Context:  {note.context[:80]}...")

    print(f"\n总记忆数: {memory_system.get_memory_count()}")

    # 检索测试
    print("\n" + "=" * 60)
    print("Step 2: 检索测试...")
    print("-" * 40)

    queries = [
        "Dave 有什么爱好?",
        "Calvin 在做什么?",
        "他们有什么合作计划?",
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        results = memory_system.retrieve(query, k=3)
        for j, mem in enumerate(results):
            print(f"  [{j + 1}] {mem.content[:70]}...")
            print(f"      Keywords: {mem.keywords}")
            print(f"      Context:  {mem.context[:60]}...")

    # 记忆结构检查
    print("\n" + "=" * 60)
    print("Step 3: 记忆结构...")
    print("-" * 40)

    for mid, note in memory_system.memories.items():
        print(f"\n  [{mid[:8]}] {note.content[:50]}...")
        print(f"    Keywords: {note.keywords}")
        print(f"    Tags:     {note.tags}")
        print(f"    Links:    {len(note.links)} connections")

    print("\n" + "=" * 60)
    print("豆包 Demo 完成!")


if __name__ == "__main__":
    main()
