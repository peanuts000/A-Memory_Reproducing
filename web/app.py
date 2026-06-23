"""
A-Mem Web 可视化服务器

提供 Web 界面来可视化 A-Mem 智能记忆系统：
  - 添加记忆，实时查看笔记构建过程
  - 可视化记忆网络图（节点 + 连接）
  - 查询记忆并查看检索结果
  - 观察记忆演化过程中标签和上下文的更新

使用方法：
    python web/app.py
    # 然后在浏览器中打开 http://localhost:5000
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify
from amem import AgenticMemorySystem

app = Flask(__name__)

# 持久化存储目录
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 全局记忆系统实例
memory_system = None
# 所有操作的活动日志
activity_log = []


def get_memory_system():
    global memory_system
    if memory_system is None:
        memory_system = AgenticMemorySystem(
            llm_backend="doubao",
            top_k=5,
        )
        # 尝试加载已保存的状态
        if os.path.exists(os.path.join(SAVE_DIR, "memories.json")):
            try:
                memory_system = AgenticMemorySystem.load(
                    SAVE_DIR,
                    llm_backend="doubao",
                )
                print(f"[已加载] 从磁盘恢复了 {memory_system.get_memory_count()} 条记忆")
            except Exception as e:
                print(f"[警告] 加载保存状态失败: {e}")
    return memory_system


def auto_save():
    """自动保存记忆系统状态到磁盘。"""
    try:
        ms = get_memory_system()
        ms.save(SAVE_DIR)
    except Exception as e:
        print(f"[警告] 自动保存失败: {e}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/add_note", methods=["POST"])
def add_note():
    """添加新记忆笔记并返回完整流水线结果。"""
    data = request.json
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "内容为空"}), 400

    ms = get_memory_system()

    # 获取添加前的状态
    before_count = ms.get_memory_count()
    before_tags = {}
    for mid, note in ms.memories.items():
        before_tags[mid] = list(note.tags)

    # 添加笔记
    note_id = ms.add_note(content)
    note = ms.memories[note_id]

    # 添加后自动保存
    auto_save()

    # 检测哪些记忆被演化了（标签发生变化）
    evolved_memories = []
    for mid, note_obj in ms.memories.items():
        if mid != note_id and mid in before_tags:
            old_tags = set(before_tags[mid])
            new_tags = set(note_obj.tags)
            if old_tags != new_tags:
                evolved_memories.append({
                    "id": mid[:8],
                    "content": note_obj.content[:60],
                    "added_tags": list(new_tags - old_tags),
                    "removed_tags": list(old_tags - new_tags),
                })

    # 构建活动日志条目
    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": "add",
        "content": content[:80],
        "note_id": note_id[:8],
        "keywords": note.keywords,
        "tags": note.tags,
        "context": note.context,
        "links_count": len(note.links),
        "evolved_count": len(evolved_memories),
    }
    activity_log.append(log_entry)

    return jsonify({
        "success": True,
        "note_id": note_id,
        "note": {
            "id": note_id[:8],
            "content": note.content,
            "keywords": note.keywords,
            "tags": note.tags,
            "context": note.context,
            "timestamp": note.timestamp,
            "links": [l[:8] if isinstance(l, str) else l for l in note.links],
        },
        "total_memories": ms.get_memory_count(),
        "evolved_memories": evolved_memories,
    })


@app.route("/api/retrieve", methods=["POST"])
def retrieve():
    """检索与查询相关的记忆。"""
    data = request.json
    query = data.get("query", "").strip()
    k = data.get("k", 3)
    if not query:
        return jsonify({"error": "查询为空"}), 400

    ms = get_memory_system()
    results = ms.retrieve(query, k=k)

    log_entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "type": "query",
        "query": query,
        "results_count": len(results),
    }
    activity_log.append(log_entry)

    return jsonify({
        "success": True,
        "query": query,
        "results": [
            {
                "id": r.id[:8],
                "content": r.content,
                "keywords": r.keywords,
                "tags": r.tags,
                "context": r.context,
                "timestamp": r.timestamp,
                "links": [l[:8] if isinstance(l, str) else l for l in r.links],
            }
            for r in results
        ],
    })


@app.route("/api/network")
def get_network():
    """获取完整的记忆网络用于图可视化。"""
    ms = get_memory_system()
    nodes = []
    edges = []

    # 构建索引到 ID 的映射以解析整数链接
    mem_ids = list(ms.memories.keys())
    id_to_short = {mid: mid[:8] for mid in mem_ids}

    for mid, note in ms.memories.items():
        short_id = id_to_short[mid]
        nodes.append({
            "id": short_id,
            "label": note.content[:40] + "..." if len(note.content) > 40 else note.content,
            "title": f"关键词: {', '.join(note.keywords)}\n标签: {', '.join(note.tags)}\n上下文: {note.context[:100]}",
            "keywords": note.keywords,
            "tags": note.tags,
            "context": note.context[:120],
            "content_short": note.content[:60],
            "links_count": len(note.links),
            "group": _assign_group(note.keywords),
        })

        # 解析链接：链接可以是整数索引或字符串 ID
        for link in note.links:
            target_short = None
            if isinstance(link, int):
                # 整数索引指向 memories 列表
                if 0 <= link < len(mem_ids):
                    target_short = id_to_short[mem_ids[link]]
            elif isinstance(link, str):
                # 字符串 ID 或前缀
                for m_id in mem_ids:
                    if m_id == link or m_id[:8] == link[:8] or m_id.startswith(link):
                        target_short = id_to_short[m_id]
                        break

            if target_short and target_short != short_id:
                # 避免重复边
                edge = {"from": short_id, "to": target_short}
                reverse = {"from": target_short, "to": short_id}
                if edge not in edges and reverse not in edges:
                    edges.append(edge)

    return jsonify({"nodes": nodes, "edges": edges})


@app.route("/api/stats")
def get_stats():
    """获取系统统计信息。"""
    ms = get_memory_system()

    # 计算唯一边数（每条边只计算一次，而不是两次）
    edges = set()
    for mid, note in ms.memories.items():
        short_id = mid[:8]
        for link in note.links:
            # 将链接解析为目标 ID
            target_short = None
            if isinstance(link, int):
                mem_ids = list(ms.memories.keys())
                if 0 <= link < len(mem_ids):
                    target_short = mem_ids[link][:8]
            elif isinstance(link, str):
                target_short = link[:8]
            if target_short:
                edge = tuple(sorted([short_id, target_short]))
                edges.add(edge)

    all_tags = set()
    all_keywords = set()
    for n in ms.memories.values():
        all_tags.update(n.tags)
        all_keywords.update(n.keywords)

    return jsonify({
        "total_memories": ms.get_memory_count(),
        "total_links": len(edges),
        "unique_tags": len(all_tags),
        "unique_keywords": len(all_keywords),
        "all_tags": sorted(all_tags),
        "all_keywords": sorted(all_keywords),
    })


@app.route("/api/activity")
def get_activity():
    """获取最近的活动日志。"""
    return jsonify({"activities": activity_log[-50:]})


@app.route("/api/clear", methods=["POST"])
def clear_memory():
    """清空所有记忆并重置系统。"""
    global memory_system, activity_log
    memory_system = None
    activity_log = []
    # 删除保存的文件
    for fname in ["memories.json", "meta.json", "retriever_cache.pkl", "retriever_embeddings.npy"]:
        fpath = os.path.join(SAVE_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    return jsonify({"success": True})


def _assign_group(keywords):
    """根据关键词内容分配颜色组。"""
    kw_set = set(k.lower() for k in keywords)
    if kw_set & {"photography", "photo", "camera", "sunset", "gallery", "exhibition"}:
        return "photography"
    elif kw_set & {"music", "song", "genre", "electronic", "studio", "dj", "performance"}:
        return "music"
    elif kw_set & {"ai", "machine learning", "collaboration", "project", "composition"}:
        return "tech"
    elif kw_set & {"hiking", "trails", "outdoor", "scenery"}:
        return "outdoor"
    else:
        return "general"


if __name__ == "__main__":
    print("=" * 60)
    print("A-Mem Web 可视化")
    print("请在浏览器中打开 http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False)
