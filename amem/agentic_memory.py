"""
AgenticMemorySystem: A-Mem 的核心记忆管理系统。

实现三阶段记忆存储流水线：
  1. 笔记构建 (Note Construction)（第 3.1 节）
  2. 链接生成 (Link Generation)（第 3.2 节）
  3. 记忆演化 (Memory Evolution)（第 3.3 节）

以及记忆检索机制（第 3.4 节）。

使用方法：
    from amem import AgenticMemorySystem

    memory = AgenticMemorySystem(
        llm_backend="doubao",
        llm_model="doubao-seed-2-0-lite-260215"
    )
    memory.add_note("缓存系统在生产环境中运行良好。")
    results = memory.retrieve("告诉我关于缓存系统的信息")
"""

import json
from typing import List, Dict, Optional, Tuple

from .memory_note import MemoryNote
from .retriever import SimpleEmbeddingRetriever
from .llm_controller import LLMController
from .prompt_templates import (
    ANALYZE_CONTENT_PROMPT,
    LINK_GENERATION_PROMPT,
    EVOLUTION_PROMPT,
    ANALYSIS_SCHEMA,
    LINK_SCHEMA,
    EVOLUTION_SCHEMA,
)
from .parsers import (
    parse_analysis_response,
    parse_link_response,
    parse_evolution_response,
)


class AgenticMemorySystem:
    """面向 LLM Agent 的智能记忆系统。

    实现完整的 A-Mem 流水线：
      - 笔记构建：使用 LLM 生成结构化笔记（关键词、上下文、标签）
      - 链接生成：找到语义相似的记忆并建立连接
      - 记忆演化：根据新信息更新已有记忆
      - 记忆检索：使用余弦相似度检索相关记忆

    参数:
        model_name: 用于嵌入的 SentenceTransformer 模型。
        llm_backend: LLM 后端（'openai', 'ollama', 'litellm', 'doubao'）。
        llm_model: LLM 模型标识符。
        evo_threshold: 每 N 次演化触发一次整合。
        api_key: 可选的 LLM 后端 API Key。
        api_base: 可选的 API Base URL。
        top_k: 操作的默认近邻数量。
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = None,
        llm_model: str = None,
        evo_threshold: int = 100,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        top_k: int = 5,
    ):
        self.memories: Dict[str, MemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = LLMController(llm_backend, llm_model, api_key, api_base)
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold
        self.top_k = top_k

    # -----------------------------------------------------------------------
    # 公共 API
    # -----------------------------------------------------------------------

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """向系统添加新的记忆笔记。

        触发完整流水线：
          1. 笔记构建（通过 LLM 生成关键词、上下文、标签）
          2. 链接生成（找到并建立连接）
          3. 记忆演化（更新相关记忆）

        参数:
            content: 原始交互内容。
            time: 可选的时间戳字符串。
            **kwargs: MemoryNote 的其他关键字参数。

        返回:
            新创建的记忆笔记 ID。
        """
        # 步骤 1：笔记构建
        note = MemoryNote(content=content, timestamp=time, **kwargs)
        note = self._construct_note(note)

        # 步骤 2 & 3：链接生成 + 记忆演化
        evolved = self._process_memory(note)

        # 存储笔记
        self.memories[note.id] = note

        # 添加到检索器索引
        doc = self._note_to_document(note)
        self.retriever.add_documents([doc])

        # 跟踪演化计数以进行整合
        if evolved:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()

        return note.id

    def retrieve(self, query: str, k: int = None) -> List[MemoryNote]:
        """检索 top-k 最相关记忆。

        实现第 3.4 节的记忆检索：
          e_q = f_enc(q)
          s_{q,i} = (e_q · e_i) / (|e_q| |e_i|)
          M_retrieved = {m_i | rank(s_{q,i}) <= k}

        参数:
            query: 查询文本。
            k: 返回结果数量。默认使用 self.top_k。

        返回:
            最相关的 MemoryNote 对象列表。
        """
        if not self.memories:
            return []

        k = k or self.top_k
        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        results = []
        for idx in indices:
            if idx < len(all_memories):
                results.append(all_memories[idx])
        return results

    def retrieve_with_context(self, query: str, k: int = None) -> str:
        """检索记忆并格式化为上下文字符串。

        参数:
            query: 查询文本。
            k: 结果数量。

        返回:
            用于提示词注入的格式化记忆上下文字符串。
        """
        memories = self.retrieve(query, k)
        if not memories:
            return ""

        context_parts = []
        for m in memories:
            part = (
                f"对话开始时间: {m.timestamp}\n"
                f"记忆内容: {m.content}\n"
                f"记忆上下文: {m.context}\n"
                f"记忆关键词: {m.keywords}\n"
                f"记忆标签: {m.tags}\n"
            )
            context_parts.append(part)
        return "\n---\n".join(context_parts)

    def get_memory_count(self) -> int:
        """返回已存储的记忆数量。"""
        return len(self.memories)

    def consolidate_memories(self) -> None:
        """从所有当前记忆重建检索器索引。

        每隔 evo_threshold 次演化调用一次，
        以确保检索器的嵌入在记忆演化改变上下文和标签后保持最新。
        """
        try:
            model_name = self.retriever.model.get_config_dict()["model_name"]
        except (AttributeError, KeyError):
            model_name = "all-MiniLM-L6-v2"

        self.retriever = SimpleEmbeddingRetriever(model_name)
        docs = [self._note_to_document(m) for m in self.memories.values()]
        if docs:
            self.retriever.add_documents(docs)

    # -----------------------------------------------------------------------
    # 内部流水线
    # -----------------------------------------------------------------------

    def _construct_note(self, note: MemoryNote) -> MemoryNote:
        """笔记构建（第 3.1 节）。

        使用 LLM 生成关键词 (K_i)、上下文 (X_i) 和标签 (G_i)。
        然后计算嵌入向量 (e_i)。

        K_i, G_i, X_i ← LLM(c_i ∥ t_i ∥ Ps1)
        e_i = f_enc[ concat(c_i, K_i, G_i, X_i) ]
        """
        prompt = ANALYZE_CONTENT_PROMPT.format(content=note.content)

        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=ANALYSIS_SCHEMA, temperature=0.3
            )
            analysis = parse_analysis_response(response, note.content)
        except Exception as e:
            print(f"笔记构建错误: {e}")
            analysis = {"keywords": [], "context": "", "tags": []}

        note.keywords = analysis["keywords"]
        note.context = analysis["context"]
        note.tags = analysis["tags"]

        return note

    def _process_memory(self, note: MemoryNote) -> bool:
        """处理记忆笔记，执行链接生成和记忆演化。

        参数:
            note: 新构建的记忆笔记。

        返回:
            如果发生了演化则返回 True，否则返回 False。
        """
        # 找到带相似度分数的最近邻
        neighbor_str, indices, scores = self._find_related_memories_with_scores(
            note.content, self.top_k
        )

        if not indices:
            return False

        # 过滤：只考虑相似度 > 阈值的邻居
        SIMILARITY_THRESHOLD = 0.3
        filtered = [(idx, score) for idx, score in zip(indices, scores) if score > SIMILARITY_THRESHOLD]
        if not filtered:
            return False

        filtered_indices = [idx for idx, _ in filtered]
        filtered_scores = [score for _, score in filtered]

        # 仅使用过滤后的邻居重建邻居字符串
        all_memories = list(self.memories.values())
        filtered_neighbor_str = ""
        for i, (idx, score) in enumerate(zip(filtered_indices, filtered_scores)):
            if idx >= len(all_memories):
                continue
            m = all_memories[idx]
            filtered_neighbor_str += (
                f"邻居索引: {i}\t"
                f"相似度: {score:.3f}\t"
                f"对话开始时间: {m.timestamp}\t"
                f"记忆内容: {m.content}\t"
                f"记忆上下文: {m.context}\t"
                f"记忆关键词: {m.keywords}\t"
                f"记忆标签: {m.tags}\n"
            )

        # 组合链接生成 + 演化提示词
        prompt = EVOLUTION_PROMPT.format(
            context=note.context,
            content=note.content,
            keywords=note.keywords,
            nearest_neighbors_memories=filtered_neighbor_str,
            neighbor_number=len(filtered_indices),
        )

        try:
            response = self.llm_controller.llm.get_completion(
                prompt, response_format=EVOLUTION_SCHEMA, temperature=0.3
            )
            evolution_result = parse_evolution_response(response, len(filtered_indices))
        except Exception as e:
            print(f"记忆演化错误: {e}")
            return False

        should_evolve = evolution_result["should_evolve"]
        if not should_evolve:
            return False

        actions = evolution_result["actions"]

        # 强化：为新记忆添加链接和更新标签
        if "strengthen" in actions:
            connections = evolution_result["suggested_connections"]
            new_tags = evolution_result["tags_to_update"]
            # 将连接索引映射回 filtered_indices（真实记忆索引）
            notes_list = list(self.memories.values())
            notes_ids = list(self.memories.keys())
            new_note_idx = len(notes_list)  # 新笔记将被插入的索引位置
            for conn in connections:
                if 0 <= conn < len(filtered_indices):
                    real_idx = filtered_indices[conn]
                    if real_idx not in note.links:
                        note.links.append(real_idx)
                        # 添加反向链接：更新被连接记忆的 links
                        if real_idx < len(notes_list):
                            target_note = notes_list[real_idx]
                            if new_note_idx not in target_note.links:
                                target_note.links.append(new_note_idx)
                                self.memories[notes_ids[real_idx]] = target_note
            if new_tags:
                note.tags = new_tags

        # 更新邻居：更新现有记忆的上下文和标签
        if "update_neighbor" in actions:
            new_contexts = evolution_result["new_context_neighborhood"]
            new_tags_list = evolution_result["new_tags_neighborhood"]
            notes_list = list(self.memories.values())
            notes_ids = list(self.memories.keys())

            for i in range(min(len(filtered_indices), len(new_tags_list))):
                memory_idx = filtered_indices[i]
                if memory_idx >= len(notes_list):
                    continue

                tag = new_tags_list[i]
                context = new_contexts[i] if i < len(new_contexts) else ""

                notetmp = notes_list[memory_idx]
                if tag:
                    notetmp.tags = tag
                if context:
                    notetmp.context = context
                self.memories[notes_ids[memory_idx]] = notetmp

        return True

    def _find_related_memories(
        self, query: str, k: int = 5
    ) -> Tuple[str, List[int]]:
        """使用检索器找到相关记忆。

        参数:
            query: 查询文本。
            k: 邻居数量。

        返回:
            (格式化的记忆字符串, 索引列表) 元组。
        """
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                f"记忆索引: {i}\t"
                f"对话开始时间: {m.timestamp}\t"
                f"记忆内容: {m.content}\t"
                f"记忆上下文: {m.context}\t"
                f"记忆关键词: {m.keywords}\t"
                f"记忆标签: {m.tags}\n"
            )
        return memory_str, indices

    def _find_related_memories_with_scores(
        self, query: str, k: int = 5
    ) -> Tuple[str, List[int], List[float]]:
        """带相似度分数的相关记忆查找。

        参数:
            query: 查询文本。
            k: 邻居数量。

        返回:
            (格式化的记忆字符串, 索引列表, 分数列表) 元组。
        """
        if not self.memories:
            return "", [], []

        results = self.retriever.search_with_scores(query, k)
        all_memories = list(self.memories.values())

        memory_str = ""
        indices = []
        scores = []
        for idx, score in results:
            if idx >= len(all_memories):
                continue
            m = all_memories[idx]
            memory_str += (
                f"记忆索引: {idx}\t"
                f"相似度: {score:.3f}\t"
                f"对话开始时间: {m.timestamp}\t"
                f"记忆内容: {m.content}\t"
                f"记忆上下文: {m.context}\t"
                f"记忆关键词: {m.keywords}\t"
                f"记忆标签: {m.tags}\n"
            )
            indices.append(idx)
            scores.append(score)
        return memory_str, indices, scores

    def _find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """查找包含链接邻居的相关记忆。

        当检索到一个记忆时，其链接的邻居也会被访问。

        参数:
            query: 查询文本。
            k: top-k 结果数量。

        返回:
            包含链接的格式化相关记忆字符串。
        """
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""

        for i in indices:
            if i >= len(all_memories):
                continue
            m = all_memories[i]
            memory_str += (
                f"对话开始时间: {m.timestamp}\t"
                f"记忆内容: {m.content}\t"
                f"记忆上下文: {m.context}\t"
                f"记忆关键词: {m.keywords}\t"
                f"记忆标签: {m.tags}\n"
            )
            # 包含链接的邻居
            for neighbor_id in m.links:
                if neighbor_id in self.memories:
                    nm = self.memories[neighbor_id]
                    memory_str += (
                        f"  [链接] 对话开始时间: {nm.timestamp}\t"
                        f"记忆内容: {nm.content}\t"
                        f"记忆上下文: {nm.context}\t"
                        f"记忆关键词: {nm.keywords}\t"
                        f"记忆标签: {nm.tags}\n"
                    )
        return memory_str

    @staticmethod
    def _note_to_document(note: MemoryNote) -> str:
        """将记忆笔记转换为检索器的文档字符串。

        按论文要求组合内容、上下文、关键词和标签。
        """
        return (
            f"content: {note.content} "
            f"context: {note.context} "
            f"keywords: {', '.join(note.keywords)} "
            f"tags: {', '.join(note.tags)}"
        )

    # -----------------------------------------------------------------------
    # 序列化
    # -----------------------------------------------------------------------

    def save(self, path: str) -> None:
        """将记忆系统状态保存到磁盘。

        参数:
            path: 保存状态的目录路径。
        """
        import os

        os.makedirs(path, exist_ok=True)

        # 保存记忆
        memories_data = {
            mid: note.to_dict() for mid, note in self.memories.items()
        }
        with open(os.path.join(path, "memories.json"), "w", encoding="utf-8") as f:
            json.dump(memories_data, f, ensure_ascii=False, indent=2)

        # 保存检索器
        self.retriever.save(
            os.path.join(path, "retriever_cache.pkl"),
            os.path.join(path, "retriever_embeddings.npy"),
        )

        # 保存元数据
        meta = {
            "evo_cnt": self.evo_cnt,
            "evo_threshold": self.evo_threshold,
            "top_k": self.top_k,
            "memory_count": len(self.memories),
        }
        with open(os.path.join(path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str,
        llm_backend: str = "openai",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
    ) -> "AgenticMemorySystem":
        """从磁盘加载保存的记忆系统。

        参数:
            path: 保存状态的目录路径。
            llm_backend: 恢复系统的 LLM 后端。
            llm_model: 恢复系统的 LLM 模型。
            api_key: 可选的 API Key。

        返回:
            恢复的 AgenticMemorySystem 实例。
        """
        import os

        # 加载元数据
        with open(os.path.join(path, "meta.json"), "r") as f:
            meta = json.load(f)

        system = cls(
            llm_backend=llm_backend,
            llm_model=llm_model,
            api_key=api_key,
            evo_threshold=meta.get("evo_threshold", 100),
            top_k=meta.get("top_k", 5),
        )
        system.evo_cnt = meta.get("evo_cnt", 0)

        # 加载记忆
        with open(os.path.join(path, "memories.json"), "r", encoding="utf-8") as f:
            memories_data = json.load(f)
        system.memories = {
            mid: MemoryNote.from_dict(data) for mid, data in memories_data.items()
        }

        # 加载检索器
        system.retriever = SimpleEmbeddingRetriever.load(
            os.path.join(path, "retriever_cache.pkl"),
            os.path.join(path, "retriever_embeddings.npy"),
        )

        return system
