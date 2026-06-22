"""
Basic Demo: A-Mem Agentic Memory System

This script demonstrates the core functionality of the A-Mem system:
  1. Adding memory notes (triggers Note Construction + Link Generation + Evolution)
  2. Retrieving relevant memories
  3. Inspecting memory structure

Usage:
    # With OpenAI (requires OPENAI_API_KEY):
    python examples/basic_demo.py

    # With Ollama (requires ollama running):
    python examples/basic_demo.py --backend ollama --model llama3.2
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amem import AgenticMemorySystem


def main():
    parser = argparse.ArgumentParser(description="A-Mem Basic Demo")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "ollama", "litellm"],
        help="LLM backend to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model identifier",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (or set OPENAI_API_KEY env var)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("A-Mem: Agentic Memory System - Basic Demo")
    print("=" * 60)
    print(f"Backend: {args.backend}, Model: {args.model}")
    print()

    # Initialize the memory system
    memory_system = AgenticMemorySystem(
        llm_backend=args.backend,
        llm_model=args.model,
        api_key=args.api_key,
        top_k=5,
    )

    # -----------------------------------------------------------------------
    # Step 1: Add memory notes
    # -----------------------------------------------------------------------
    print("Step 1: Adding memory notes...")
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
        print(f"\nAdding memory {i + 1}/{len(conversations)}...")
        note_id = memory_system.add_note(conv)
        memory_ids.append(note_id)
        print(f"  ID: {note_id[:8]}...")

    print(f"\nTotal memories stored: {memory_system.get_memory_count()}")

    # -----------------------------------------------------------------------
    # Step 2: Retrieve memories
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 2: Retrieving memories...")
    print("-" * 40)

    queries = [
        "Which hobby did Dave pick up?",
        "What is Calvin working on?",
        "Tell me about their collaboration plans",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")
        results = memory_system.retrieve(query, k=3)
        for j, mem in enumerate(results):
            print(f"  [{j + 1}] Content: {mem.content[:80]}...")
            print(f"      Context: {mem.context}")
            print(f"      Keywords: {mem.keywords}")
            print(f"      Tags: {mem.tags}")

    # -----------------------------------------------------------------------
    # Step 3: Inspect memory structure
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Step 3: Memory structure inspection...")
    print("-" * 40)

    for mid in memory_ids:
        note = memory_system.memories[mid]
        print(f"\nMemory: {note.content[:60]}...")
        print(f"  Keywords: {note.keywords}")
        print(f"  Tags: {note.tags}")
        print(f"  Context: {note.context}")
        print(f"  Links: {note.links}")

    print("\n" + "=" * 60)
    print("Demo complete!")


if __name__ == "__main__":
    main()
