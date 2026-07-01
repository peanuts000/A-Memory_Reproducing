"""
Dataset loader for LoCoMo dataset.
"""

import json
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QA:
    question: str
    answer: Optional[str]
    evidence: List[str]
    category: Optional[int] = None
    adversarial_answer: Optional[str] = None

    @property
    def final_answer(self) -> Optional[str]:
        """Get the appropriate answer based on category."""
        if self.category == 5:
            return self.adversarial_answer
        return self.answer


@dataclass
class Turn:
    speaker: str
    dia_id: str
    text: str


@dataclass
class Session:
    session_id: int
    date_time: str
    turns: List[Turn]


@dataclass
class Conversation:
    speaker_a: str
    speaker_b: str
    sessions: Dict[int, Session]


@dataclass
class EventSummary:
    events: Dict[str, Dict[str, List[str]]]


@dataclass
class Observation:
    observations: Dict[str, Dict[str, List[List[str]]]]


@dataclass
class LoCoMoSample:
    """A single sample from the LoCoMo dataset."""
    sample_id: str
    qa: List[QA]
    conversation: Conversation
    event_summary: EventSummary
    observation: Observation
    session_summary: Dict[str, str]


def parse_session(session_data: List[dict], session_id: int, date_time: str) -> Session:
    """Parse a single session's data, including turns with images by using their captions."""
    turns = []
    for turn in session_data:
        text = turn.get("text", "")
        if "img_url" in turn and "blip_caption" in turn:
            caption_text = f"[Image: {turn['blip_caption']}]"
            if text:
                text = f"{caption_text} {text}"
            else:
                text = caption_text

        turns.append(Turn(
            speaker=turn["speaker"],
            dia_id=turn["dia_id"],
            text=text
        ))
    return Session(session_id=session_id, date_time=date_time, turns=turns)


def parse_conversation(conv_data: dict) -> Conversation:
    """Parse conversation data."""
    sessions = {}
    for key, value in conv_data.items():
        if key.startswith("session_") and isinstance(value, list):
            session_id = int(key.split("_")[1])
            date_time = conv_data.get(f"{key}_date_time")
            if date_time:
                session = parse_session(value, session_id, date_time)
                if session.turns:
                    sessions[session_id] = session

    return Conversation(
        speaker_a=conv_data["speaker_a"],
        speaker_b=conv_data["speaker_b"],
        sessions=sessions
    )


def load_locomo_dataset(file_path: Union[str, Path]) -> List[LoCoMoSample]:
    """
    Load the LoCoMo dataset from a JSON file.

    Args:
        file_path: Path to the JSON file containing the dataset

    Returns:
        List of LoCoMoSample objects containing the parsed data
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found at {file_path}")

    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    samples = []
    total_qa = 0
    qa_counts_per_sample = []

    for sample_idx, sample in enumerate(data):
        try:
            qa_list = []
            sample_qa_count = 0

            for qa_idx, qa in enumerate(sample["qa"]):
                try:
                    qa_obj = QA(
                        question=qa["question"],
                        answer=qa.get("answer"),
                        evidence=qa.get("evidence", []),
                        category=qa.get("category"),
                        adversarial_answer=qa.get("adversarial_answer")
                    )
                    qa_list.append(qa_obj)
                    sample_qa_count += 1

                except KeyError as e:
                    print(f"Error in sample {sample_idx}, QA pair {qa_idx}:")
                    print(f"QA data: {qa}")
                    raise e

            conversation = parse_conversation(sample["conversation"])
            event_summary = EventSummary(events=sample["event_summary"])
            observation = Observation(observations=sample["observation"])
            session_summary = sample.get("session_summary", {})

            sample_obj = LoCoMoSample(
                sample_id=str(sample_idx),
                qa=qa_list,
                conversation=conversation,
                event_summary=event_summary,
                observation=observation,
                session_summary=session_summary
            )
            samples.append(sample_obj)

            total_qa += sample_qa_count
            qa_counts_per_sample.append(sample_qa_count)

            print(f"Sample {sample_idx}: {sample_qa_count} QAs")

        except Exception as e:
            print(f"Error processing sample {sample_idx}: {str(e)}")
            raise e

    print(f"\nTotal QAs: {total_qa}")
    print(f"Average QAs per sample: {total_qa / len(samples):.2f}")

    return samples
