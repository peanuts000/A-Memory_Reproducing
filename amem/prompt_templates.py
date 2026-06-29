"""
Plain-text prompt templates, section-marker parsers, and validation logic
for the robust A-MEM system. Replaces JSON-schema LLM calls with plain-text
prompts that work with any LLM backend (Ollama, SGLang, OpenAI, etc.).

Prompt templates are copied directly from the reference implementation.
"""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following content and provide:
1. KEYWORDS: The most important keywords (nouns, verbs, key concepts). Order from most to least important. At least three keywords. Do not include speaker names or time references.
2. CONTEXT: One sentence summarizing the main topic, key points, and purpose.
3. TAGS: Broad categories/themes for classification (domain, format, type). At least three tags.

Respond using EXACTLY this format (one section per header):

KEYWORDS: keyword1, keyword2, keyword3, ...
CONTEXT: A single sentence summarizing the content.
TAGS: tag1, tag2, tag3, ...

Content for analysis:
{content}"""


EVOLUTION_DECISION_PROMPT = """You are an AI memory evolution agent. Analyze the new memory note and its nearest neighbors to decide if evolution is needed.

New memory:
- Context: {context}
- Content: {content}
- Keywords: {keywords}

Nearest neighbor memories:
{nearest_neighbors_memories}

Based on the relationships between the new memory and its neighbors, decide:
- NO_EVOLUTION: The memory stands alone, no changes needed.
- STRENGTHEN: The new memory should be linked to some neighbors and its tags updated.
- UPDATE_NEIGHBOR: The neighbors' context/tags should be updated based on new understanding.
- STRENGTHEN_AND_UPDATE: Both strengthen and update neighbors.

Respond using EXACTLY this format:
DECISION: <one of NO_EVOLUTION, STRENGTHEN, UPDATE_NEIGHBOR, STRENGTHEN_AND_UPDATE>
REASON: <brief explanation>"""


STRENGTHEN_DETAILS_PROMPT = """Given the new memory and its neighbors, provide updated connections and tags.

New memory:
- Content: {content}
- Keywords: {keywords}

Neighbor memories:
{nearest_neighbors_memories}

Which neighbor indices should the new memory connect to? What tags best describe this memory?

Respond using EXACTLY this format:
CONNECTIONS: 0, 2, 3
TAGS: tag1, tag2, tag3, ..."""


UPDATE_NEIGHBORS_PROMPT = """Given the new memory and its neighbor memories, update each neighbor's context and tags based on a holistic understanding of all these memories together.

New memory:
- Content: {content}
- Context: {context}

Neighbor memories:
{nearest_neighbors_memories}

For each neighbor (indexed 0 to {max_neighbor_idx}), provide updated context and tags. If no change is needed, repeat the original values.

Respond using EXACTLY this format (one block per neighbor):

NEIGHBOR 0:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

NEIGHBOR 1:
CONTEXT: updated context sentence
TAGS: tag1, tag2, tag3

(continue for all {neighbor_count} neighbors)"""


FOCUSED_KEYWORDS_PROMPT = """List exactly 5 keywords that capture the main concepts of the following text. Output only the keywords, comma-separated, nothing else.

Text: {content}"""


# ---------------------------------------------------------------------------
# JSON schemas for structured output (used by backends that support it)
# ---------------------------------------------------------------------------

ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "analysis_response",
        "schema": {
            "type": "object",
            "properties": {
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "context": {
                    "type": "string",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["keywords", "context", "tags"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVOLUTION_DECISION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evolution_decision",
        "schema": {
            "type": "object",
            "properties": {
                "decision": {"type": "string"},
                "reason": {"type": "string"},
            },
            "required": ["decision", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

STRENGTHEN_DETAILS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "strengthen_details",
        "schema": {
            "type": "object",
            "properties": {
                "connections": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["connections", "tags"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

UPDATE_NEIGHBORS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "update_neighbors",
        "schema": {
            "type": "object",
            "properties": {
                "new_context_neighborhood": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "new_tags_neighborhood": {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
            },
            "required": ["new_context_neighborhood", "new_tags_neighborhood"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
