"""
A-MEM Memory Layer implementation.

This module implements the agentic memory system with:
- MemoryNote: Basic memory unit with metadata
- SimpleEmbeddingRetriever: Retrieval using embeddings
- AgenticMemorySystem: Main memory management system
"""

from typing import List, Dict, Optional, Any
import json
import re
import uuid
import logging
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize

from llm_controllers import BaseLLMController

logger = logging.getLogger("amem")


def simple_tokenize(text):
    """Simple tokenization using nltk."""
    return word_tokenize(text)


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
# Parsing helpers
# ---------------------------------------------------------------------------

def strip_markdown_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences from LLM output."""
    text = text.strip()
    text = re.sub(r'^```(?:json)?\s*\n?', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n?\s*```$', '', text, flags=re.MULTILINE)
    return text.strip()


def _parse_list_items(text: str) -> List[str]:
    """Parse a section of text into a list of items."""
    if not text or not text.strip():
        return []

    lines = text.strip().splitlines()
    items: List[str] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Strip bullet markers
        line = re.sub(r'^[\-\*•]\s*', '', line)
        line = re.sub(r'^\d+[\.\)]\s*', '', line)
        # Strip surrounding quotes
        line = line.strip().strip('"').strip("'").strip()
        if not line:
            continue
        # If the line contains commas, split on them
        if ',' in line:
            for part in line.split(','):
                part = part.strip().strip('"').strip("'").strip()
                if part:
                    items.append(part)
        else:
            items.append(line)

    return items


def _extract_section(text: str, marker: str, next_markers: Optional[List[str]] = None) -> str:
    """Extract the text between *marker*: and the next known marker (or end)."""
    pattern = re.compile(
        rf'^\s*{re.escape(marker)}\s*:\s*(.*)$',
        re.IGNORECASE | re.MULTILINE,
    )
    match = pattern.search(text)
    if not match:
        return ""

    start = match.end()
    first_line = match.group(1).strip()

    end = len(text)
    if next_markers:
        for nm in next_markers:
            nm_pattern = re.compile(
                rf'^\s*{re.escape(nm)}\s*:', re.IGNORECASE | re.MULTILINE
            )
            nm_match = nm_pattern.search(text, start)
            if nm_match and nm_match.start() < end:
                end = nm_match.start()

    rest = text[start:end].strip()
    if first_line and rest:
        return first_line + "\n" + rest
    return first_line or rest


def parse_with_json_fallback(response: str, plain_text_parser, *parser_args) -> Any:
    """Try JSON parsing first; fall back to section-marker parsing."""
    try:
        cleaned = strip_markdown_fences(response)
        result = json.loads(cleaned)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass
    return plain_text_parser(response, *parser_args)


def _heuristic_keywords(content: str, max_keywords: int = 5) -> List[str]:
    """Extract heuristic keywords from content text."""
    stop_words = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
        'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
        'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'out', 'off', 'over', 'under', 'again',
        'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
        'how', 'all', 'both', 'each', 'few', 'more', 'most', 'other',
        'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
        'than', 'too', 'very', 'just', 'because', 'but', 'and', 'or',
        'if', 'while', 'about', 'up', 'it', 'its', 'i', 'me', 'my',
        'you', 'your', 'he', 'she', 'they', 'we', 'this', 'that', 'these',
        'those', 'what', 'which', 'who', 'whom', 'says', 'said', 'speaker',
    }
    words = re.findall(r'\b[a-zA-Z]{3,}\b', content)
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
    match = re.match(r'(.+?[.!?])\s', content)
    if match:
        return match.group(1).strip()
    return content[:200].strip()


def validate_analysis_result(result: Dict[str, Any], content: str = "") -> Dict[str, Any]:
    """Validate and repair the analysis result."""
    if not isinstance(result, dict):
        result = {"keywords": [], "context": "", "tags": []}

    keywords = result.get("keywords", [])
    context = result.get("context", "")
    tags = result.get("tags", [])

    if isinstance(keywords, str):
        keywords = _parse_list_items(keywords)
    if isinstance(tags, str):
        tags = _parse_list_items(tags)
    if isinstance(context, list):
        context = " ".join(context)

    if not keywords and content:
        keywords = _heuristic_keywords(content)

    if not context and content:
        context = _heuristic_context(content)

    if not tags and keywords:
        tags = keywords[:3]

    result["keywords"] = keywords
    result["context"] = context
    result["tags"] = tags
    return result


# ---------------------------------------------------------------------------
# Parsers for LLM responses
# ---------------------------------------------------------------------------

def parse_analyze_content(response: str, content: str = "") -> Dict[str, Any]:
    """Parse the analyze_content LLM response."""
    def _section_parse(resp: str, content_text: str = "") -> Dict[str, Any]:
        keywords_text = _extract_section(resp, "KEYWORDS", ["CONTEXT", "TAGS"])
        context_text = _extract_section(resp, "CONTEXT", ["TAGS", "KEYWORDS"])
        tags_text = _extract_section(resp, "TAGS", ["KEYWORDS", "CONTEXT"])

        keywords = _parse_list_items(keywords_text)
        context = context_text.strip() if context_text.strip() else ""
        tags = _parse_list_items(tags_text)

        return {"keywords": keywords, "context": context, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse, content)
    result = validate_analysis_result(result, content)
    return result


def parse_evolution_decision(response: str) -> Dict[str, str]:
    """Parse the evolution decision response."""
    def _section_parse(resp: str) -> Dict[str, str]:
        decision_text = _extract_section(resp, "DECISION", ["REASON"])
        reason_text = _extract_section(resp, "REASON", ["DECISION"])

        decision = decision_text.strip().upper().replace(" ", "_")
        valid_decisions = {
            "NO_EVOLUTION", "STRENGTHEN", "UPDATE_NEIGHBOR",
            "STRENGTHEN_AND_UPDATE"
        }
        if decision not in valid_decisions:
            resp_upper = resp.upper()
            if "STRENGTHEN" in resp_upper and "UPDATE" in resp_upper:
                decision = "STRENGTHEN_AND_UPDATE"
            elif "STRENGTHEN" in resp_upper:
                decision = "STRENGTHEN"
            elif "UPDATE" in resp_upper:
                decision = "UPDATE_NEIGHBOR"
            else:
                decision = "NO_EVOLUTION"

        return {"decision": decision, "reason": reason_text.strip()}

    result = parse_with_json_fallback(response, _section_parse)

    if "should_evolve" in result:
        should_evolve = result.get("should_evolve", False)
        actions = result.get("actions", [])
        if not should_evolve:
            decision = "NO_EVOLUTION"
        elif "strengthen" in actions and "update_neighbor" in actions:
            decision = "STRENGTHEN_AND_UPDATE"
        elif "strengthen" in actions:
            decision = "STRENGTHEN"
        elif "update_neighbor" in actions:
            decision = "UPDATE_NEIGHBOR"
        else:
            decision = "NO_EVOLUTION"
        result = {"decision": decision, "reason": ""}

    if "decision" not in result:
        result = {"decision": "NO_EVOLUTION", "reason": ""}

    return result


def parse_strengthen_details(response: str) -> Dict[str, Any]:
    """Parse the strengthen details response."""
    def _section_parse(resp: str) -> Dict[str, Any]:
        conn_text = _extract_section(resp, "CONNECTIONS", ["TAGS"])
        tags_text = _extract_section(resp, "TAGS", ["CONNECTIONS"])

        connections = []
        for item in _parse_list_items(conn_text):
            try:
                connections.append(int(item.strip()))
            except (ValueError, TypeError):
                pass

        tags = _parse_list_items(tags_text)
        return {"connections": connections, "tags": tags}

    result = parse_with_json_fallback(response, _section_parse)

    if "suggested_connections" in result and "connections" not in result:
        result["connections"] = [int(x) for x in result.get("suggested_connections", []) if isinstance(x, (int, float))]
    if "tags_to_update" in result and "tags" not in result:
        result["tags"] = result.get("tags_to_update", [])

    result.setdefault("connections", [])
    result.setdefault("tags", [])
    return result


def parse_update_neighbors(response: str, num_neighbors: int) -> List[Dict[str, Any]]:
    """Parse the update neighbors response."""
    def _section_parse(resp: str, n_neighbors: int) -> List[Dict[str, Any]]:
        neighbors = []
        for i in range(n_neighbors):
            pattern = re.compile(
                rf'NEIGHBOR\s+{i}\s*:', re.IGNORECASE
            )
            match = pattern.search(resp)
            if not match:
                neighbors.append({"context": "", "tags": []})
                continue

            next_pattern = re.compile(
                rf'NEIGHBOR\s+{i + 1}\s*:', re.IGNORECASE
            )
            next_match = next_pattern.search(resp, match.end())
            block_end = next_match.start() if next_match else len(resp)
            block = resp[match.end():block_end]

            ctx = _extract_section(block, "CONTEXT", ["TAGS"])
            tags_text = _extract_section(block, "TAGS", ["CONTEXT"])
            tags = _parse_list_items(tags_text)

            neighbors.append({"context": ctx.strip(), "tags": tags})

        return neighbors

    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            contexts = data.get("new_context_neighborhood", [])
            tags_list = data.get("new_tags_neighborhood", [])
            neighbors = []
            for i in range(num_neighbors):
                ctx = contexts[i] if i < len(contexts) else ""
                tags = tags_list[i] if i < len(tags_list) else []
                neighbors.append({"context": ctx, "tags": tags})
            return neighbors
    except (json.JSONDecodeError, ValueError):
        pass

    return _section_parse(response, num_neighbors)


def parse_plain_text_answer(response: str) -> str:
    """Parse a plain-text answer response (for QA evaluation)."""
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


def parse_keywords_response(response: str) -> str:
    """Parse generate_query_llm response."""
    try:
        cleaned = strip_markdown_fences(response)
        data = json.loads(cleaned)
        if isinstance(data, dict) and "keywords" in data:
            return str(data["keywords"])
    except (json.JSONDecodeError, ValueError):
        pass
    return response.strip()


# ---------------------------------------------------------------------------
# Memory Note
# ---------------------------------------------------------------------------

class MemoryNote:
    """Basic memory unit with metadata."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[BaseLLMController] = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

    @staticmethod
    def analyze_content(content: str, llm_controller: BaseLLMController) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = ANALYZE_CONTENT_PROMPT.format(content=content)
        try:
            response = llm_controller.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.get_completion(retry_prompt, temperature=0.3)
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            logger.error("Error analyzing content: %s", e)
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# Simple Embedding Retriever
# ---------------------------------------------------------------------------

class SimpleEmbeddingRetriever:
    """Simple retrieval system using only text embeddings."""

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.corpus = []
        self.embeddings = None
        self.document_ids = {}

    def add_documents(self, documents: List[str]):
        """Add documents to the retriever."""
        if not self.corpus:
            self.corpus = documents
            self.embeddings = self.model.encode(documents)
            self.document_ids = {doc: idx for idx, doc in enumerate(documents)}
        else:
            start_idx = len(self.corpus)
            self.corpus.extend(documents)
            new_embeddings = self.model.encode(documents)
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            for idx, doc in enumerate(documents):
                self.document_ids[doc] = start_idx + idx

    def search(self, query: str, k: int = 5) -> List[int]:
        """Search for similar documents using cosine similarity."""
        if not self.corpus:
            return []

        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return top_k_indices.tolist()

    def save(self, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Save retriever state to disk."""
        import pickle
        if self.embeddings is not None:
            np.save(retriever_cache_embeddings_file, self.embeddings)

        state = {
            'corpus': self.corpus,
            'document_ids': self.document_ids
        }
        with open(retriever_cache_file, 'wb') as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, retriever_cache_file: str, retriever_cache_embeddings_file: str):
        """Load retriever state from disk."""
        import pickle
        import os

        with open(retriever_cache_file, 'rb') as f:
            state = pickle.load(f)

        retriever = cls(model_name='all-MiniLM-L6-v2')
        retriever.corpus = state['corpus']
        retriever.document_ids = state['document_ids']

        if os.path.exists(retriever_cache_embeddings_file):
            retriever.embeddings = np.load(retriever_cache_embeddings_file)

        return retriever

    @classmethod
    def load_from_local_memory(cls, memories: Dict, model_name: str) -> 'SimpleEmbeddingRetriever':
        """Load retriever state from memory."""
        all_docs = []
        for m in memories.values():
            metadata_text = f"{m.context} {' '.join(m.keywords)} {' '.join(m.tags)}"
            doc = f"{m.content} , {metadata_text}"
            all_docs.append(doc)

        retriever = cls(model_name)
        retriever.add_documents(all_docs)
        return retriever


# ---------------------------------------------------------------------------
# Agentic Memory System
# ---------------------------------------------------------------------------

class AgenticMemorySystem:
    """Memory management system with embedding-based retrieval."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_controller: Optional[BaseLLMController] = None,
                 evo_threshold: int = 100):

        self.memories: Dict[str, MemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = llm_controller
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        note = MemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            **kwargs,
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags)
        ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    def process_memory(self, note: MemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls."""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        try:
            # Call 1: Evolution decision
            decision_prompt = EVOLUTION_DECISION_PROMPT.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            decision_response = self.llm_controller.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # Call 2: Strengthen details (conditional)
            if should_strengthen:
                strengthen_prompt = STRENGTHEN_DETAILS_PROMPT.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                strengthen_response = self.llm_controller.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # Call 3: Update neighbors (conditional)
            if should_update:
                update_prompt = UPDATE_NEIGHBORS_PROMPT.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                update_response = self.llm_controller.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note
