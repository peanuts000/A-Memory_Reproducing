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

ANALYZE_CONTENT_PROMPT = """通过对以下内容进行结构化分析：
1. 识别最显著的关键词（聚焦名词、动词和关键概念）
2. 提取核心主题和上下文元素
3. 创建相关的分类标签

请以 JSON 对象格式返回响应：
{{
    "keywords": [
        // 若干个具体、独特的关键词，捕获关键概念和术语
        // 从最重要到最不重要排序
        // 不要包含说话者名称或时间相关的关键词
        // 至少三个关键词，但不要过于冗余
    ],
    "context":
        // 一句话总结：
        // - 主要主题/领域
        // - 关键论点/要点
        // - 目标受众/目的
    ,
    "tags": [
        // 若干个用于分类的广泛类别/主题
        // 包含领域、格式和类型标签
        // 至少三个标签，但不要过于冗余
    ]
}}

待分析内容：
{content}"""


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
# 问答提示词：基于检索到的记忆回答问题
# ---------------------------------------------------------------------------

QA_PROMPT = """你是一个有用的助手。请根据提供的记忆上下文回答以下问题。

问题: {question}

相关记忆:
{memories}

请基于上述记忆中的信息提供简洁准确的回答。如果记忆中没有足够的信息来回答问题，请说"我没有足够的信息来回答这个问题"。

回答:"""


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
