"""
Response parsers for A-Mem LLM outputs.

Handles both JSON structured output and plain-text fallback parsing
with section-marker extraction.
"""

import json
import re
from typing import Dict, List, Any


def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_json_response(response: str) -> Dict[str, Any]:
    """Parse JSON from LLM response, handling markdown fences.

    Args:
        response: Raw LLM response string.

    Returns:
        Parsed dictionary, or empty dict on failure.
    """
    cleaned = strip_markdown_fences(response)
    # Try to find JSON object
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def parse_analysis_response(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the note construction analysis response.

    Expected JSON format:
        {"keywords": [...], "context": "...", "tags": [...]}

    Args:
        response: Raw LLM response.
        content: Original content (used for fallback).

    Returns:
        Dictionary with keywords, context, and tags.
    """
    result = parse_json_response(response)

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # Validate and repair
    if not isinstance(keywords, list):
        keywords = []
    if not isinstance(context, str):
        context = str(context) if context else ""
    if not isinstance(tags, list):
        tags = []

    # Fallback: extract from content if empty
    if not keywords and content:
        keywords = _heuristic_keywords(content)
    if not context and content:
        context = _heuristic_context(content)
    if not tags and keywords:
        tags = keywords[:3]

    return {"keywords": keywords, "context": context, "tags": tags}


def parse_link_response(response: str) -> Dict[str, Any]:
    """Parse the link generation response.

    Expected JSON format:
        {"should_link": bool, "suggested_connections": [...], "reason": "..."}

    Args:
        response: Raw LLM response.

    Returns:
        Dictionary with should_link, suggested_connections, and reason.
    """
    result = parse_json_response(response)

    should_link = result.get("should_link", False)
    suggested_connections = result.get("suggested_connections", [])
    reason = result.get("reason", "")

    # Ensure connections are integers
    connections = []
    for c in suggested_connections:
        try:
            connections.append(int(c))
        except (ValueError, TypeError):
            pass

    return {
        "should_link": bool(should_link),
        "suggested_connections": connections,
        "reason": str(reason),
    }


def parse_evolution_response(response: str, num_neighbors: int) -> Dict[str, Any]:
    """Parse the memory evolution response.

    Expected JSON format:
        {
            "should_evolve": bool,
            "actions": [...],
            "suggested_connections": [...],
            "tags_to_update": [...],
            "new_context_neighborhood": [...],
            "new_tags_neighborhood": [[...], ...]
        }

    Args:
        response: Raw LLM response.
        num_neighbors: Expected number of neighbors.

    Returns:
        Dictionary with evolution decisions and updated values.
    """
    result = parse_json_response(response)

    should_evolve = result.get("should_evolve", False)
    actions = result.get("actions", [])
    suggested_connections = result.get("suggested_connections", [])
    tags_to_update = result.get("tags_to_update", [])
    new_context = result.get("new_context_neighborhood", [])
    new_tags = result.get("new_tags_neighborhood", [])

    # Ensure lengths match num_neighbors
    if len(new_context) < num_neighbors:
        new_context.extend([""] * (num_neighbors - len(new_context)))
    if len(new_tags) < num_neighbors:
        new_tags.extend([[] for _ in range(num_neighbors - len(new_tags))])

    # Ensure connections are integers
    connections = []
    for c in suggested_connections:
        try:
            connections.append(int(c))
        except (ValueError, TypeError):
            pass

    return {
        "should_evolve": bool(should_evolve),
        "actions": actions if isinstance(actions, list) else [],
        "suggested_connections": connections,
        "tags_to_update": tags_to_update if isinstance(tags_to_update, list) else [],
        "new_context_neighborhood": new_context[:num_neighbors],
        "new_tags_neighborhood": new_tags[:num_neighbors],
    }


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "to", "of", "in",
        "for", "on", "with", "at", "by", "from", "as", "into", "through",
        "during", "before", "after", "above", "below", "between", "out",
        "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "all", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not",
        "only", "own", "same", "so", "than", "too", "very", "just",
        "because", "but", "and", "or", "if", "while", "about", "up",
        "it", "its", "i", "me", "my", "you", "your", "he", "she", "they",
        "we", "this", "that", "these", "those", "what", "which", "who",
        "says", "said", "speaker",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", content)
    scored = []
    seen = set()
    for w in words:
        w_lower = w.lower()
        if w_lower in stop_words or w_lower in seen:
            continue
        seen.add(w_lower)
        score = 2 if w[0].isupper() else 1
        scored.append((w_lower, score))
    scored.sort(key=lambda x: -x[1])
    return [w for w, _ in scored[:max_keywords]]


def _heuristic_context(content: str) -> str:
    """Extract a heuristic context sentence from content."""
    match = re.match(r"(.+?[.!?])\s", content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()
