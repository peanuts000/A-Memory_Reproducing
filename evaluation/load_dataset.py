"""
LoCoMo dataset loader for A-Mem evaluation.

LoCoMo (Long Conversation Memory) dataset contains long conversations
with QA pairs across five categories:
  1. Single-hop: Answerable from a single session
  2. Multi-hop: Requires synthesizing information across sessions
  3. Temporal: Tests understanding of time-related information
  4. Open-domain: Requires integration with external knowledge
  5. Adversarial: Evaluates ability to recognize unanswerable queries
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class QAPair:
    """Question-answer pair from the dataset."""
    question: str
    answer: str
    category: int  # 0=single-hop, 1=multi-hop, 2=temporal, 3=open-domain, 4=adversarial
    category_name: str = ""


@dataclass
class Turn:
    """A single turn in a conversation."""
    speaker: str
    content: str
    timestamp: str = ""


@dataclass
class Session:
    """A session containing multiple conversation turns."""
    session_id: int
    turns: List[Turn] = field(default_factory=list)


@dataclass
class Conversation:
    """A complete conversation with multiple sessions and QA pairs."""
    conversation_id: str
    sessions: List[Session] = field(default_factory=list)
    qa_pairs: List[QAPair] = field(default_factory=list)


# Category name mapping
CATEGORY_NAMES = {
    0: "Single Hop",
    1: "Multi Hop",
    2: "Temporal",
    3: "Open Domain",
    4: "Adversarial",
}


def load_locomo_dataset(data_dir: str) -> List[Conversation]:
    """Load LoCoMo dataset from a directory.

    Expected directory structure:
        data_dir/
            conversation_0.json
            conversation_1.json
            ...

    Each JSON file should contain:
        {
            "conversation_id": "...",
            "sessions": [
                {
                    "session_id": 1,
                    "turns": [
                        {"speaker": "A", "content": "...", "timestamp": "..."},
                        ...
                    ]
                },
                ...
            ],
            "qa_pairs": [
                {
                    "question": "...",
                    "answer": "...",
                    "category": 0-4
                },
                ...
            ]
        }

    Args:
        data_dir: Path to directory containing conversation JSON files.

    Returns:
        List of Conversation objects.
    """
    conversations = []

    if not os.path.exists(data_dir):
        print(f"Warning: Data directory not found: {data_dir}")
        return conversations

    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".json"):
            continue

        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        conv = _parse_conversation(data)
        conversations.append(conv)

    return conversations


def load_locomo_from_single_file(filepath: str) -> List[Conversation]:
    """Load LoCoMo from a single JSON file containing all conversations.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of Conversation objects.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [_parse_conversation(conv) for conv in data]
    elif isinstance(data, dict):
        if "conversations" in data:
            return [_parse_conversation(c) for c in data["conversations"]]
        return [_parse_conversation(data)]
    return []


def _parse_conversation(data: dict) -> Conversation:
    """Parse a conversation from a dictionary."""
    conv = Conversation(conversation_id=data.get("conversation_id", "unknown"))

    # Parse sessions
    for session_data in data.get("sessions", []):
        session = Session(session_id=session_data.get("session_id", 0))
        for turn_data in session_data.get("turns", []):
            turn = Turn(
                speaker=turn_data.get("speaker", "Unknown"),
                content=turn_data.get("content", ""),
                timestamp=turn_data.get("timestamp", ""),
            )
            session.turns.append(turn)
        conv.sessions.append(session)

    # Parse QA pairs
    for qa_data in data.get("qa_pairs", data.get("questions", [])):
        qa = QAPair(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", qa_data.get("reference", "")),
            category=qa_data.get("category", 0),
            category_name=CATEGORY_NAMES.get(qa_data.get("category", 0), "Unknown"),
        )
        conv.qa_pairs.append(qa)

    return conv


def conversations_to_memory_texts(conversations: List[Conversation]) -> List[List[Dict]]:
    """Convert conversations to memory-ready text format.

    Each turn becomes a memory note with content and timestamp.

    Args:
        conversations: List of Conversation objects.

    Returns:
        Nested list of dicts, one list per conversation, each dict containing
        'content' and 'timestamp' keys.
    """
    all_texts = []
    for conv in conversations:
        texts = []
        for session in conv.sessions:
            for turn in session.turns:
                content = f"Speaker {turn.speaker} says: {turn.content}"
                texts.append({
                    "content": content,
                    "timestamp": turn.timestamp or f"session_{session.session_id}",
                })
        all_texts.append(texts)
    return all_texts
