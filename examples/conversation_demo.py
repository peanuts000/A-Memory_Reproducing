"""
对话演示：交互式 A-Mem 记忆系统。

演示 A-Mem 如何在多轮对话中维护记忆，
构建随时间演化的互连知识网络。

使用方法：
    python examples/conversation_demo.py --backend openai --model gpt-4o-mini
    python examples/conversation_demo.py --backend doubao
    python examples/conversation_demo.py --interactive
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem


def run_conversation_demo(memory_system: AgenticMemorySystem):
    """运行预设的多轮对话演示。"""

    conversations = [
        # 第一轮对话：互相了解
        ("2023-07-21", "Calvin", "嘿 Dave！我刚搬进新的工作室。有自己的地方做音乐真是太棒了。"),
        ("2023-07-21", "Dave", "太好了 Calvin！我最近迷上了徒步。这附近有一些很棒的步道。"),
        ("2023-07-21", "Calvin", "不错！我一直在尝试在歌曲中加入电子元素。合成器让歌曲有了全新的感觉。"),
        ("2023-07-21", "Dave", "你应该去看看湖边的山间步道。那里的风景也非常适合摄影。"),

        # 第二轮对话：叙旧
        ("2023-09-15", "Dave", "我一直在做一个摄影项目。在不同地点拍摄日落。"),
        ("2023-09-15", "Calvin", "听起来很美！我一直在和一个本地 DJ 合作。我们下个月计划一场现场演出。"),
        ("2023-09-15", "Dave", "说到合作，你有没有想过用 AI 来辅助音乐创作？"),
        ("2023-09-15", "Calvin", "确实有！我一直在阅读关于 AI 音乐生成的文章。它可以帮助创作伴奏。"),

        # 第三轮对话：深入交流
        ("2023-10-22", "Calvin", "现场演出非常成功！有超过 200 人参加。电子音乐让观众非常兴奋。"),
        ("2023-10-22", "Dave", "恭喜！我完成了我的摄影系列。下个月将在本地画廊展出。"),
        ("2023-10-22", "Calvin", "我们来办个联合展览吧！我的音乐配上你的照片。会是一种沉浸式体验。"),
        ("2023-10-22", "Dave", "好主意！我们可以叫它'律动中的和谐'——音乐与视觉艺术的结合。"),
    ]

    print("正在向 A-Mem 系统添加对话记忆...\n")

    for timestamp, speaker, content in conversations:
        formatted = f"Speaker {speaker} says: {content}"
        note_id = memory_system.add_note(formatted, time=timestamp)
        print(f"[{timestamp}] {speaker}: {content[:60]}...")
        print(f"  → 记忆 ID: {note_id[:8]}...")

    print(f"\n记忆总数: {memory_system.get_memory_count()}")

    # 使用各种查询测试检索
    print("\n" + "=" * 60)
    print("测试记忆检索")
    print("=" * 60)

    test_queries = [
        "Calvin 开始了什么爱好?",
        "告诉我 Dave 的摄影项目",
        "他们讨论了什么合作计划?",
        "现场音乐演出是什么时候?",
        "他们联合展览的名字是什么?",
    ]

    for query in test_queries:
        print(f"\n查询: '{query}'")
        results = memory_system.retrieve(query, k=3)
        for i, mem in enumerate(results):
            print(f"  [{i + 1}] {mem.content[:80]}...")
            print(f"      上下文: {mem.context}")
            print(f"      关键词: {mem.keywords}")

    # 通过检查笔记展示记忆演化
    print("\n" + "=" * 60)
    print("演化后的记忆结构")
    print("=" * 60)

    for mid, note in memory_system.memories.items():
        if note.links:
            print(f"\n记忆: {note.content[:60]}...")
            print(f"  链接到: {len(note.links)} 条其他记忆")
            print(f"  标签: {note.tags}")


def interactive_mode(memory_system: AgenticMemorySystem):
    """运行交互模式进行手动测试。"""
    print("\n" + "=" * 60)
    print("交互模式 - A-Mem 记忆系统")
    print("=" * 60)
    print("命令:")
    print("  add <内容>   - 添加新记忆")
    print("  ask <问题>   - 查询记忆系统")
    print("  show         - 显示所有记忆")
    print("  quit         - 退出")
    print()

    while True:
        try:
            user_input = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            break

        if user_input.lower() == "show":
            for mid, note in memory_system.memories.items():
                print(f"\n  [{mid[:8]}] {note.content}")
                print(f"    关键词: {note.keywords}")
                print(f"    标签: {note.tags}")
                print(f"    上下文: {note.context}")
                print(f"    链接: {note.links}")
            continue

        if user_input.lower().startswith("add "):
            content = user_input[4:].strip()
            if content:
                note_id = memory_system.add_note(content)
                print(f"已添加记忆: {note_id[:8]}...")
            continue

        if user_input.lower().startswith("ask "):
            query = user_input[4:].strip()
            if query:
                results = memory_system.retrieve(query, k=3)
                if not results:
                    print("未找到相关记忆。")
                else:
                    for i, mem in enumerate(results):
                        print(f"\n  [{i + 1}] {mem.content}")
                        print(f"      上下文: {mem.context}")
                        print(f"      关键词: {mem.keywords}")
            continue

        print("未知命令。请使用 'add', 'ask', 'show' 或 'quit'。")


def main():
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215") if default_backend == "doubao" else "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="A-Mem 对话演示")
    parser.add_argument("--backend", type=str, default=default_backend, choices=["openai", "ollama", "litellm", "doubao"])
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API Base URL（从 .env 自动读取）")
    parser.add_argument("--interactive", action="store_true", help="运行交互模式")
    args = parser.parse_args()

    memory_system = AgenticMemorySystem(
        llm_backend=args.backend,
        llm_model=args.model,
        api_key=args.api_key,
        api_base=args.base_url,
    )

    if args.interactive:
        interactive_mode(memory_system)
    else:
        run_conversation_demo(memory_system)


if __name__ == "__main__":
    main()
