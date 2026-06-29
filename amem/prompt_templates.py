"""
A-Mem 记忆操作的提示词模板。

这些模板对应论文中描述的提示词：
  - Ps1: 笔记构建（第 3.1 节，附录 B.1）
  - Ps2: 链接生成（第 3.2 节，附录 B.2）
  - Ps3: 记忆演化（第 3.3 节，附录 B.3）
"""

# ---------------------------------------------------------------------------
# Ps1: 笔记构建（附录 B.1）
# K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ Ps1)
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Analyze the following conversation turn and extract structured metadata.

IMPORTANT RULES:
1. Keywords MUST include: named entities (people, places, organizations, dates), key actions, and important concepts
2. Context MUST be a complete sentence describing "Who did/said what, and when" using the timestamp
3. Tags MUST include: topic domain, action type, and temporal category if applicable

Timestamp: {timestamp}

Conversation content:
{content}

Respond with a JSON object:
{{
    "keywords": [
        // Named entities first (people, places, organizations)
        // Then key actions/verbs
        // Then important concepts/objects
        // At least 3 keywords, ordered by importance
        // Include specific names, dates, locations mentioned
    ],
    "context": "One sentence summarizing WHO did/said WHAT and WHEN (use the timestamp to make it specific)",
    "tags": [
        // Topic domain tag (e.g., "health", "family", "career", "hobbies")
        // Action type tag (e.g., "planning", "sharing", "asking", "celebrating")
        // Temporal tag if applicable (e.g., "summer_2023", "july", "weekend")
        // At least 3 tags
    ]
}}"""


# ---------------------------------------------------------------------------
# Ps2: 链接生成（附录 B.2）
# L_i ← LLM(m_n ∥ M_n^near ∥ Ps2)
# ---------------------------------------------------------------------------

LINK_GENERATION_PROMPT = """你是一个负责管理和演化知识库的 AI 记忆演化代理。

请根据关键词和上下文分析新的记忆笔记，以及其若干最近邻记忆。

新记忆上下文: {context}
内容: {content}
关键词: {keywords}

最近邻记忆:
{nearest_neighbors_memories}

基于以上信息，请判断：
该记忆是否应该与任何最近邻记忆建立链接？
请考虑它们之间的语义关系和共享属性。

请以 JSON 格式响应：
{{
    "should_link": true/false,
    "suggested_connections": [要链接的邻居记忆索引列表],
    "reason": "简要说明建立这些连接的原因"
}}"""


# ---------------------------------------------------------------------------
# Ps3: 记忆演化（附录 B.3）
# m*_j ← LLM(m_n ∥ M_n^near \ m_j ∥ m_j ∥ Ps3)
# ---------------------------------------------------------------------------

EVOLUTION_PROMPT = """你是一个负责管理和演化知识库的 AI 记忆演化代理。

请根据关键词和上下文分析新的记忆笔记，以及其若干最近邻记忆。
对其演化做出决策。

新记忆上下文: {context}
内容: {content}
关键词: {keywords}

最近邻记忆:
{nearest_neighbors_memories}

基于以上信息，请判断：
1. 应该采取哪些具体操作（强化连接 strengthen、更新邻居 update_neighbor）？
   1.1 如果选择强化连接，应该与哪个记忆连接？能否给出该记忆更新后的标签？
   1.2 如果选择更新邻居，你可以基于对这些记忆的理解更新它们的上下文和标签。
       如果上下文和标签不需要更新，则新上下文和标签应与原始相同。
       请按输入邻居的顺序生成新的上下文和标签。
标签应根据这些记忆的特征确定，以便后续检索和分类使用。
注意：new_tags_neighborhood 的长度必须等于输入邻居数量，new_context_neighborhood 的长度也必须等于输入邻居数量。
邻居数量为 {neighbor_number}。

所有上述信息应按顺序以列表格式返回：
[[new_memory],[neighbor_memory_1],...[neighbor_memory_n]]

这些操作可以组合使用。

请以 JSON 格式返回决策，结构如下：
{{
    "should_evolve": true/false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": [邻居记忆ID列表],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["新上下文",...,"新上下文"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]]
}}"""


# ---------------------------------------------------------------------------
# 分步演化提示词（用于优雅降级）
# ---------------------------------------------------------------------------

# 步骤 1：演化决策（仅判断是否需要演化及操作类型）
EVOLUTION_DECISION_PROMPT = """你是一个 AI 记忆演化代理。分析新的记忆笔记及其最近邻记忆，决定是否需要演化。

新记忆:
- 上下文: {context}
- 内容: {content}
- 关键词: {keywords}

最近邻记忆:
{nearest_neighbors_memories}

基于以上信息，请决定:
- NO_EVOLUTION: 记忆独立存在，无需变化。
- STRENGTHEN: 新记忆应与某些邻居建立链接，并更新其标签。
- UPDATE_NEIGHBOR: 邻居的上下文/标签应基于新理解进行更新。
- STRENGTHEN_AND_UPDATE: 同时进行强化和更新。

请以 JSON 格式响应:
{{
    "decision": "NO_EVOLUTION 或 STRENGTHEN 或 UPDATE_NEIGHBOR 或 STRENGTHEN_AND_UPDATE",
    "reason": "简要说明原因"
}}"""

# 步骤 2：强化详情（仅在需要强化时调用）
STRENGTHEN_DETAILS_PROMPT = """给定新记忆及其邻居，提供更新的连接和标签。

新记忆:
- 内容: {content}
- 关键词: {keywords}

邻居记忆:
{nearest_neighbors_memories}

新记忆应该与哪些邻居索引连接？什么标签最能描述这个记忆？

请以 JSON 格式响应:
{{
    "suggested_connections": [0, 2, 3],
    "tags_to_update": ["tag1", "tag2", "tag3"]
}}"""

# 步骤 3：更新邻居（仅在需要更新时调用）
UPDATE_NEIGHBORS_PROMPT = """给定新记忆及其邻居记忆，基于对所有记忆的整体理解更新每个邻居的上下文和标签。

新记忆:
- 内容: {content}
- 上下文: {context}

邻居记忆:
{nearest_neighbors_memories}

对每个邻居（索引 0 到 {max_neighbor_idx}），提供更新的上下文和标签。如果不需要更改，请重复原始值。
邻居数量为 {neighbor_count}。

请以 JSON 格式响应:
{{
    "new_context_neighborhood": ["更新的上下文", ...],
    "new_tags_neighborhood": [["tag1", "tag2"], ...]
}}"""

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
                "suggested_connections": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "tags_to_update": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["suggested_connections", "tags_to_update"],
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


# ---------------------------------------------------------------------------
# 问答提示词：基于检索到的记忆回答问题
# ---------------------------------------------------------------------------

QA_PROMPT = """You are a precise assistant. Answer the following question based ONLY on the provided conversation memories.

Question: {question}

Relevant memories:
{memories}

RULES:
1. Answer with a SHORT PHRASE only (not a full sentence)
2. Use EXACT words from the memories when possible
3. If the information is NOT in the memories, answer "Not mentioned"
4. Do NOT output JSON format
5. Do NOT add explanations or reasoning

Answer:"""


# ---------------------------------------------------------------------------
# 结构化输出的 JSON schema 定义
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

LINK_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "link_response",
        "schema": {
            "type": "object",
            "properties": {
                "should_link": {"type": "boolean"},
                "suggested_connections": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "reason": {"type": "string"},
            },
            "required": ["should_link", "suggested_connections", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

EVOLUTION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evolution_response",
        "schema": {
            "type": "object",
            "properties": {
                "should_evolve": {"type": "boolean"},
                "actions": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "suggested_connections": {
                    "type": "array",
                    "items": {"type": "integer"},
                },
                "tags_to_update": {
                    "type": "array",
                    "items": {"type": "string"},
                },
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
            "required": [
                "should_evolve",
                "actions",
                "suggested_connections",
                "tags_to_update",
                "new_context_neighborhood",
                "new_tags_neighborhood",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
}
