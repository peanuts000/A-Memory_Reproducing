"""
LoCoMo 数据集加载器，用于 A-Mem 评估。

LoCoMo（长对话记忆）数据集包含长对话，
配有五个类别的 QA 对：
  1. 单跳 (Single-hop)：可从单个会话回答
  2. 多跳 (Multi-hop)：需要跨会话综合信息
  3. 时间 (Temporal)：测试对时间相关信息的理解
  4. 开放领域 (Open-domain)：需要与外部知识整合
  5. 对抗性 (Adversarial)：评估识别无法回答查询的能力
"""

import json
import os
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class QAPair:
    """数据集中的问答对。"""
    question: str
    answer: str
    category: int  # 1=单跳, 2=多跳, 3=时间, 4=开放领域, 5=对抗性
    category_name: str = ""


@dataclass
class Turn:
    """对话中的一轮。"""
    speaker: str
    content: str
    timestamp: str = ""


@dataclass
class Session:
    """包含多轮对话的会话。"""
    session_id: int
    turns: List[Turn] = field(default_factory=list)


@dataclass
class Conversation:
    """包含多个会话和 QA 对的完整对话。"""
    conversation_id: str
    sessions: List[Session] = field(default_factory=list)
    qa_pairs: List[QAPair] = field(default_factory=list)


# 类别名称映射
CATEGORY_NAMES = {
    0: "单跳",
    1: "多跳",
    2: "时间",
    3: "开放领域",
    4: "对抗性",
}


def load_locomo_dataset(data_dir: str) -> List[Conversation]:
    """从目录加载 LoCoMo 数据集。

    期望的目录结构：
        data_dir/
            conversation_0.json
            conversation_1.json
            ...

    每个 JSON 文件应包含：
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

    参数:
        data_dir: 包含对话 JSON 文件的目录路径。

    返回:
        Conversation 对象列表。
    """
    conversations = []

    if not os.path.exists(data_dir):
        print(f"警告: 未找到数据目录: {data_dir}")
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
    """从单个 JSON 文件加载 LoCoMo，文件包含所有对话。

    参数:
        filepath: JSON 文件路径。

    返回:
        Conversation 对象列表。
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
    """从字典解析对话。"""
    conv = Conversation(conversation_id=data.get("conversation_id", "unknown"))

    # 解析会话
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

    # 解析 QA 对
    for qa_data in data.get("qa_pairs", data.get("questions", [])):
        qa = QAPair(
            question=qa_data.get("question", ""),
            answer=qa_data.get("answer", qa_data.get("reference", "")),
            category=qa_data.get("category", 0),
            category_name=CATEGORY_NAMES.get(qa_data.get("category", 0), "未知"),
        )
        conv.qa_pairs.append(qa)

    return conv


def conversations_to_memory_texts(conversations: List[Conversation]) -> List[List[Dict]]:
    """将对话转换为记忆就绪的文本格式。

    每一轮成为一条带有内容和时间戳的记忆笔记。

    参数:
        conversations: Conversation 对象列表。

    返回:
        嵌套字典列表，每个对话一个列表，每个字典包含
        'content' 和 'timestamp' 键。
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
