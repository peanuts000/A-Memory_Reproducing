"""
Conversation Demo: Interactive A-Mem memory system.

Demonstrates how A-Mem maintains memory across a multi-turn conversation,
building an interconnected knowledge network that evolves over time.

Usage:
    python examples/conversation_demo.py --backend openai --model gpt-4o-mini
    python examples/conversation_demo.py --backend ollama --model llama3.2
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem


def run_conversation_demo(memory_system: AgenticMemorySystem):
    """Run a scripted multi-turn conversation demo."""

    conversations = [
        # Session 1: Getting to know each other
        ("2023-07-21", "Calvin", "Hey Dave! I just moved into my new studio space. It's amazing having my own place to work on music."),
        ("2023-07-21", "Dave", "That's great Calvin! I've been really into hiking lately. Found some amazing trails around here."),
        ("2023-07-21", "Calvin", "Nice! I've been experimenting with electronic elements in my songs. Adding synths gives them a fresh vibe."),
        ("2023-07-21", "Dave", "You should check out the mountain trails near the lake. The scenery is perfect for photography too."),

        # Session 2: Catching up
        ("2023-09-15", "Dave", "I've been working on a photography project. Capturing sunsets at different locations."),
        ("2023-09-15", "Calvin", "That sounds beautiful! I've been collaborating with a local DJ. We're planning a live show next month."),
        ("2023-09-15", "Dave", "Speaking of collaborations, have you thought about using AI for your music compositions?"),
        ("2023-09-15", "Calvin", "Actually yes! I've been reading about AI music generation. It could help with creating backing tracks."),

        # Session 3: Deep dive
        ("2023-10-22", "Calvin", "The live show went amazing! We had over 200 people. The electronic sets really got the crowd going."),
        ("2023-10-22", "Dave", "Congrats! I finished my photography series. Going to exhibit at the local gallery next month."),
        ("2023-10-22", "Calvin", "Let's do a joint exhibition! My music with your photos. It would be an immersive experience."),
        ("2023-10-22", "Dave", "That's a brilliant idea! We could call it 'Harmony in Motion' - music and visual art combined."),
    ]

    print("Adding conversation memories to A-Mem system...\n")

    for timestamp, speaker, content in conversations:
        formatted = f"Speaker {speaker} says: {content}"
        note_id = memory_system.add_note(formatted, time=timestamp)
        print(f"[{timestamp}] {speaker}: {content[:60]}...")
        print(f"  → Memory ID: {note_id[:8]}...")

    print(f"\nTotal memories: {memory_system.get_memory_count()}")

    # Test retrieval with various queries
    print("\n" + "=" * 60)
    print("Testing Memory Retrieval")
    print("=" * 60)

    test_queries = [
        "What hobby did Calvin start?",
        "Tell me about Dave's photography project",
        "What collaboration ideas did they discuss?",
        "When was the live music show?",
        "What is the name of their joint exhibition?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = memory_system.retrieve(query, k=3)
        for i, mem in enumerate(results):
            print(f"  [{i + 1}] {mem.content[:80]}...")
            print(f"      Context: {mem.context}")
            print(f"      Keywords: {mem.keywords}")

    # Demonstrate memory evolution by inspecting notes
    print("\n" + "=" * 60)
    print("Memory Structure After Evolution")
    print("=" * 60)

    for mid, note in memory_system.memories.items():
        if note.links:
            print(f"\nMemory: {note.content[:60]}...")
            print(f"  Links to: {len(note.links)} other memories")
            print(f"  Tags: {note.tags}")


def interactive_mode(memory_system: AgenticMemorySystem):
    """Run in interactive mode for manual testing."""
    print("\n" + "=" * 60)
    print("Interactive Mode - A-Mem Memory System")
    print("=" * 60)
    print("Commands:")
    print("  add <content>  - Add a new memory")
    print("  ask <question> - Query the memory system")
    print("  show           - Show all memories")
    print("  quit           - Exit")
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
                print(f"    Keywords: {note.keywords}")
                print(f"    Tags: {note.tags}")
                print(f"    Context: {note.context}")
                print(f"    Links: {note.links}")
            continue

        if user_input.lower().startswith("add "):
            content = user_input[4:].strip()
            if content:
                note_id = memory_system.add_note(content)
                print(f"Added memory: {note_id[:8]}...")
            continue

        if user_input.lower().startswith("ask "):
            query = user_input[4:].strip()
            if query:
                results = memory_system.retrieve(query, k=3)
                if not results:
                    print("No relevant memories found.")
                else:
                    for i, mem in enumerate(results):
                        print(f"\n  [{i + 1}] {mem.content}")
                        print(f"      Context: {mem.context}")
                        print(f"      Keywords: {mem.keywords}")
            continue

        print("Unknown command. Use 'add', 'ask', 'show', or 'quit'.")


def main():
    default_backend = "doubao" if os.getenv("DOUBAO_API_KEY") else "openai"
    default_model = os.getenv("DOUBAO_MODEL", "doubao-seed-2-0-lite-260215") if default_backend == "doubao" else "gpt-4o-mini"

    parser = argparse.ArgumentParser(description="A-Mem Conversation Demo")
    parser.add_argument("--backend", type=str, default=default_backend, choices=["openai", "ollama", "litellm", "doubao"])
    parser.add_argument("--model", type=str, default=default_model)
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None, help="API base URL (auto-read from .env)")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
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
