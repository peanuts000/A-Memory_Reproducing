"""将 snap-research/LoCoMo 数据集转换为评估脚本所需的格式。"""

import json
import os


def convert_locomo_dataset(input_file: str, output_dir: str):
    """转换 LoCoMo 数据集。

    参数:
        input_file: 输入的 locomo10.json 文件路径
        output_dir: 输出目录
    """
    print(f"加载数据集: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"共 {len(data)} 个对话")

    os.makedirs(output_dir, exist_ok=True)

    # 类别映射 (1-5 -> 0-4)
    category_map = {
        1: 0,  # Single-hop
        2: 1,  # Multi-hop (论文中是 2，但我们的评估脚本用 1)
        3: 2,  # Temporal
        4: 3,  # Open-domain
        5: 4,  # Adversarial
    }

    for idx, conv_data in enumerate(data):
        print(f"\n处理对话 {idx + 1}/{len(data)}: {conv_data.get('sample_id', f'conv-{idx}')}")

        conversation = {
            "conversation_id": f"conversation_{idx}",
            "sessions": [],
            "qa_pairs": []
        }

        # 提取说话者名称
        speaker_a = conv_data.get("conversation", {}).get("speaker_a", "Speaker A")
        speaker_b = conv_data.get("conversation", {}).get("speaker_b", "Speaker B")

        # 提取会话
        conversation_dict = conv_data.get("conversation", {})
        session_idx = 1
        while f"session_{session_idx}" in conversation_dict:
            session_key = f"session_{session_idx}"
            date_key = f"session_{session_idx}_date_time"

            session_data = conversation_dict.get(session_key, [])
            timestamp = conversation_dict.get(date_key, "")

            # 解析会话内容
            session = {
                "session_id": session_idx,
                "turns": []
            }

            # session_data 是一个列表，每个元素是 dict
            if isinstance(session_data, list):
                for turn_data in session_data:
                    if isinstance(turn_data, dict):
                        session["turns"].append({
                            "speaker": turn_data.get("speaker", "Unknown"),
                            "content": turn_data.get("text", ""),
                            "timestamp": timestamp,
                            "dia_id": turn_data.get("dia_id", "")
                        })

            conversation["sessions"].append(session)
            session_idx += 1

        # 提取 QA 对
        for qa_data in conv_data.get("qa", []):
            category = qa_data.get("category", 1)
            # 映射类别到 0-4
            mapped_category = category_map.get(category, 0)

            conversation["qa_pairs"].append({
                "question": qa_data.get("question", ""),
                "answer": qa_data.get("answer", ""),
                "category": mapped_category,
                "evidence": qa_data.get("evidence", [])
            })

        # 保存对话
        output_file = os.path.join(output_dir, f"conversation_{idx}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(conversation, f, ensure_ascii=False, indent=2)

        total_turns = sum(len(s["turns"]) for s in conversation["sessions"])
        print(f"  会话数: {len(conversation['sessions'])}")
        print(f"  总轮次: {total_turns}")
        print(f"  QA 对数: {len(conversation['qa_pairs'])}")

    print(f"\n转换完成！数据已保存到 {output_dir}")

    # 打印统计信息
    total_qa = 0
    category_counts = {}
    for idx in range(len(data)):
        with open(os.path.join(output_dir, f"conversation_{idx}.json"), "r", encoding="utf-8") as f:
            conv = json.load(f)
            total_qa += len(conv["qa_pairs"])
            for qa in conv["qa_pairs"]:
                cat = qa.get("category", 0)
                category_counts[cat] = category_counts.get(cat, 0) + 1

    print(f"\n统计:")
    print(f"  总 QA 对数: {total_qa}")
    print(f"  各类别分布:")
    category_names = {0: "单跳", 1: "多跳", 2: "时间", 3: "开放领域", 4: "对抗性"}
    for cat in sorted(category_counts.keys()):
        print(f"    {category_names.get(cat, f'类别{cat}')}: {category_counts[cat]}")


if __name__ == "__main__":
    input_file = "./locomo_repo/data/locomo10.json"
    output_dir = "./data/locomo_v2"

    if not os.path.exists(input_file):
        print(f"错误: 未找到输入文件 {input_file}")
        print("请先运行: git clone https://github.com/snap-research/LoCoMo.git locomo_repo")
    else:
        convert_locomo_dataset(input_file, output_dir)
