"""
A-Mem LLM 输出的响应解析器。

处理 JSON 结构化输出和纯文本回退解析，
支持分段标记提取。
"""

import json
import re
from typing import Dict, List, Any


def strip_markdown_fences(text: str) -> str:
    """移除 LLM 输出中的 ```json ... ``` 或 ``` ... ``` 围栏。"""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


def parse_json_response(response: str) -> Dict[str, Any]:
    """从 LLM 响应中解析 JSON，处理 markdown 围栏。

    参数:
        response: 原始 LLM 响应字符串。

    返回:
        解析后的字典，失败时返回空字典。
    """
    cleaned = strip_markdown_fences(response)
    # 尝试查找 JSON 对象
    start_idx = cleaned.find("{")
    end_idx = cleaned.rfind("}")
    if start_idx != -1 and end_idx != -1:
        cleaned = cleaned[start_idx : end_idx + 1]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def parse_analysis_response(response: str, content: str = "") -> Dict[str, Any]:
    """解析笔记构建的分析响应。

    期望的 JSON 格式：
        {"keywords": [...], "context": "...", "tags": [...]}

    参数:
        response: 原始 LLM 响应。
        content: 原始内容（用于回退）。

    返回:
        包含 keywords、context 和 tags 的字典。
    """
    result = parse_json_response(response)

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    # 验证和修复
    if not isinstance(keywords, list):
        keywords = []
    if not isinstance(context, str):
        context = str(context) if context else ""
    if not isinstance(tags, list):
        tags = []

    # 回退：如果为空则从内容中提取
    if not keywords and content:
        keywords = _heuristic_keywords(content)
    if not context and content:
        context = _heuristic_context(content)
    if not tags and keywords:
        tags = keywords[:3]

    return {"keywords": keywords, "context": context, "tags": tags}


def parse_link_response(response: str) -> Dict[str, Any]:
    """解析链接生成响应。

    期望的 JSON 格式：
        {"should_link": bool, "suggested_connections": [...], "reason": "..."}

    参数:
        response: 原始 LLM 响应。

    返回:
        包含 should_link、suggested_connections 和 reason 的字典。
    """
    result = parse_json_response(response)

    should_link = result.get("should_link", False)
    suggested_connections = result.get("suggested_connections", [])
    reason = result.get("reason", "")

    # 确保连接为整数
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
    """解析记忆演化响应。

    期望的 JSON 格式：
        {
            "should_evolve": bool,
            "actions": [...],
            "suggested_connections": [...],
            "tags_to_update": [...],
            "new_context_neighborhood": [...],
            "new_tags_neighborhood": [[...], ...]
        }

    参数:
        response: 原始 LLM 响应。
        num_neighbors: 期望的邻居数量。

    返回:
        包含演化决策和更新值的字典。
    """
    result = parse_json_response(response)

    should_evolve = result.get("should_evolve", False)
    actions = result.get("actions", [])
    suggested_connections = result.get("suggested_connections", [])
    tags_to_update = result.get("tags_to_update", [])
    new_context = result.get("new_context_neighborhood", [])
    new_tags = result.get("new_tags_neighborhood", [])

    # 确保长度匹配 num_neighbors
    if len(new_context) < num_neighbors:
        new_context.extend([""] * (num_neighbors - len(new_context)))
    if len(new_tags) < num_neighbors:
        new_tags.extend([[] for _ in range(num_neighbors - len(new_tags))])

    # 确保连接为整数
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
    """从内容文本中启发式提取关键词。"""
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
    """从内容中启发式提取上下文句子。"""
    match = re.match(r"(.+?[.!?])\s", content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()
