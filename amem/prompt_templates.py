"""
Prompt Templates for A-Mem memory operations.

These templates correspond to the prompts described in the paper:
  - Ps1: Note Construction (Section 3.1, Appendix B.1)
  - Ps2: Link Generation (Section 3.2, Appendix B.2)
  - Ps3: Memory Evolution (Section 3.3, Appendix B.3)
"""

# ---------------------------------------------------------------------------
# Ps1: Note Construction (Appendix B.1)
# K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ Ps1)
# ---------------------------------------------------------------------------

ANALYZE_CONTENT_PROMPT = """Generate a structured analysis of the following content by:
1. Identifying the most salient keywords (focus on nouns, verbs, and key concepts)
2. Extracting core themes and contextual elements
3. Creating relevant categorical tags

Format the response as a JSON object:
{{
    "keywords": [
        // several specific, distinct keywords that capture key concepts and terminology
        // Order from most to least important
        // Don't include keywords that are the name of the speaker or time
        // At least three keywords, but don't be too redundant.
    ],
    "context":
        // one sentence summarizing:
        // - Main topic/domain
        // - Key arguments/points
        // - Intended audience/purpose
    ,
    "tags": [
        // several broad categories/themes for classification
        // Include domain, format, and type tags
        // At least three tags, but don't be too redundant.
    ]
}}

Content for analysis:
{content}"""


# ---------------------------------------------------------------------------
# Ps2: Link Generation (Appendix B.2)
# L_i ← LLM(m_n ∥ M_n^near ∥ Ps2)
# ---------------------------------------------------------------------------

LINK_GENERATION_PROMPT = """You are an AI memory evolution agent responsible for managing and evolving a knowledge base.

Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.

The new memory context: {context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
Should this memory be linked to any of the nearest neighbor memories?
Consider the semantic relationships and shared attributes between them.

Respond in JSON format:
{{
    "should_link": true/false,
    "suggested_connections": [list of neighbor memory indices to link to],
    "reason": "brief explanation of why these connections were made"
}}"""


# ---------------------------------------------------------------------------
# Ps3: Memory Evolution (Appendix B.3)
# m*_j ← LLM(m_n ∥ M_n^near \ m_j ∥ m_j ∥ Ps3)
# ---------------------------------------------------------------------------

EVOLUTION_PROMPT = """You are an AI memory evolution agent responsible for managing and evolving a knowledge base.

Analyze the new memory note according to keywords and context, also with their several nearest neighbors memory.
Make decisions about its evolution.

The new memory context: {context}
content: {content}
keywords: {keywords}

The nearest neighbors memories:
{nearest_neighbors_memories}

Based on this information, determine:
1. What specific actions should be taken (strengthen, update_neighbor)?
   1.1 If choose to strengthen the connection, which memory should it be connected to? Can you give the updated tags of this memory?
   1.2 If choose to update neighbor, you can update the context and tags of these memories based on the understanding of these memories. If the context and the tags are not updated, the new context and tags should be the same as the original ones. Generate the new context and tags in the sequential order of the input neighbors.
Tags should be determined by the content of these characteristic of these memories, which can be used to retrieve them later and categorize them.
Note that the length of new_tags_neighborhood must equal the number of input neighbors, and the length of new_context_neighborhood must equal the number of input neighbors.
The number of neighbors is {neighbor_number}.

All the above information should be returned in a list format according to the sequence:
[[new_memory],[neighbor_memory_1],...[neighbor_memory_n]]

These actions can be combined.

Return your decision in JSON format with the following structure:
{{
    "should_evolve": true/false,
    "actions": ["strengthen", "update_neighbor"],
    "suggested_connections": [neighbor_memory_ids],
    "tags_to_update": ["tag_1",..."tag_n"],
    "new_context_neighborhood": ["new context",...,"new context"],
    "new_tags_neighborhood": [["tag_1",...,"tag_n"],...["tag_1",...,"tag_n"]]
}}"""


# ---------------------------------------------------------------------------
# QA Prompt for answering questions with retrieved memories
# ---------------------------------------------------------------------------

QA_PROMPT = """You are a helpful assistant. Answer the following question based on the provided memory context.

Question: {question}

Relevant memories:
{memories}

Provide a concise and accurate answer based on the information in the memories above. If the memories do not contain enough information to answer the question, say "I don't have enough information to answer this question."

Answer:"""


# ---------------------------------------------------------------------------
# JSON schema definitions for structured output
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
