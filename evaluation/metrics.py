"""
A-Mem 系统的评估指标。

实现论文中使用的指标（第 4.1 节，附录 A.2）：
  - F1 分数（词元级别）
  - BLEU-1
  - ROUGE-L / ROUGE-2
  - METEOR
  - SBERT 相似度
"""

import re
import numpy as np
from typing import Dict, List
from collections import defaultdict
import statistics


def simple_tokenize(text: str) -> List[str]:
    """简单分词：小写化，按空格分割。"""
    text = str(text).lower()
    for ch in [".", ",", "!", "?", ";", ":", "(", ")"]:
        text = text.replace(ch, " ")
    return text.split()


def calculate_f1(prediction: str, reference: str) -> float:
    """计算词元级别的 F1 分数（公式 11-13）。

    F1 = 2 * precision * recall / (precision + recall)
    """
    if not prediction or not reference:
        return 0.0

    pred_tokens = set(simple_tokenize(prediction))
    ref_tokens = set(simple_tokenize(reference))

    if not pred_tokens or not ref_tokens:
        return 0.0

    common = pred_tokens & ref_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def calculate_bleu1(prediction: str, reference: str) -> float:
    """计算 BLEU-1 分数（公式 14-16）。

    BLEU-1 = BP * exp(w_1 * log(p_1))
    """
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

        pred_tokens = simple_tokenize(prediction)
        ref_tokens = [simple_tokenize(reference)]
        smooth = SmoothingFunction().method1
        return sentence_bleu(
            ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth
        )
    except ImportError:
        # 回退：一元组精度
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = simple_tokenize(reference)
        if not pred_tokens:
            return 0.0
        ref_counts = {}
        for t in ref_tokens:
            ref_counts[t] = ref_counts.get(t, 0) + 1
        match_count = 0
        for t in pred_tokens:
            if t in ref_counts and ref_counts[t] > 0:
                match_count += 1
                ref_counts[t] -= 1
        bp = 1.0 if len(pred_tokens) > len(ref_tokens) else np.exp(1 - len(ref_tokens) / max(len(pred_tokens), 1))
        return bp * (match_count / len(pred_tokens))


def calculate_rouge_l(prediction: str, reference: str) -> float:
    """计算 ROUGE-L 分数（公式 17-19）。

    ROUGE-L = (1 + beta^2) * R_l * P_l / (R_l + beta^2 * P_l)
    """
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rougeL"].fmeasure
    except ImportError:
        # 回退：基于 LCS 的计算
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = simple_tokenize(reference)
        lcs_len = _lcs_length(pred_tokens, ref_tokens)
        if lcs_len == 0:
            return 0.0
        r_l = lcs_len / len(ref_tokens) if ref_tokens else 0
        p_l = lcs_len / len(pred_tokens) if pred_tokens else 0
        beta = 1.2
        if r_l + beta * beta * p_l == 0:
            return 0.0
        return (1 + beta * beta) * r_l * p_l / (r_l + beta * beta * p_l)


def calculate_rouge2(prediction: str, reference: str) -> float:
    """计算 ROUGE-2 分数（公式 20）。"""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge2"], use_stemmer=True)
        scores = scorer.score(reference, prediction)
        return scores["rouge2"].fmeasure
    except ImportError:
        # 回退：二元组重叠
        pred_tokens = simple_tokenize(prediction)
        ref_tokens = simple_tokenize(reference)
        pred_bigrams = _get_bigrams(pred_tokens)
        ref_bigrams = _get_bigrams(ref_tokens)
        if not ref_bigrams:
            return 0.0
        ref_counts = {}
        for bg in ref_bigrams:
            ref_counts[bg] = ref_counts.get(bg, 0) + 1
        overlap = 0
        for bg in pred_bigrams:
            if bg in ref_counts and ref_counts[bg] > 0:
                overlap += 1
                ref_counts[bg] -= 1
        return overlap / len(ref_bigrams)


def calculate_meteor(prediction: str, reference: str) -> float:
    """计算 METEOR 分数（公式 21-23）。"""
    try:
        from nltk.translate.meteor_score import meteor_score

        return meteor_score([reference.split()], prediction.split())
    except ImportError:
        # 回退：简单一元组重叠
        return calculate_f1(prediction, reference)


def calculate_sbert_similarity(prediction: str, reference: str) -> float:
    """计算 SBERT 余弦相似度（公式 24-25）。

    SBERT_Similarity = cos(SBERT(x), SBERT(y))
    """
    try:
        from sentence_transformers import SentenceTransformer
        from sentence_transformers.util import pytorch_cos_sim

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb1 = model.encode([prediction], convert_to_tensor=True)
        emb2 = model.encode([reference], convert_to_tensor=True)
        return float(pytorch_cos_sim(emb1, emb2).item())
    except ImportError:
        return 0.0


def calculate_all_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """计算所有评估指标。

    参数:
        prediction: 模型预测的答案。
        reference: 真实参考答案。

    返回:
        指标名称到分数的字典。
    """
    if not prediction or not reference:
        return {
            "f1": 0.0,
            "bleu1": 0.0,
            "rougeL": 0.0,
            "rouge2": 0.0,
            "meteor": 0.0,
            "sbert_similarity": 0.0,
        }

    prediction = str(prediction).strip()
    reference = str(reference).strip()

    return {
        "f1": calculate_f1(prediction, reference),
        "bleu1": calculate_bleu1(prediction, reference),
        "rougeL": calculate_rouge_l(prediction, reference),
        "rouge2": calculate_rouge2(prediction, reference),
        "meteor": calculate_meteor(prediction, reference),
        "sbert_similarity": calculate_sbert_similarity(prediction, reference),
    }


def aggregate_metrics(
    all_metrics: List[Dict[str, float]],
    all_categories: List[int],
) -> Dict[str, Dict[str, float]]:
    """按类别聚合指标。

    参数:
        all_metrics: 指标字典列表。
        all_categories: 每个指标对应的类别 ID 列表。

    返回:
        包含 'overall' 和各类别统计信息的字典。
    """
    if not all_metrics:
        return {}

    aggregates = defaultdict(list)
    category_aggregates = defaultdict(lambda: defaultdict(list))

    for metrics, category in zip(all_metrics, all_categories):
        for metric_name, value in metrics.items():
            aggregates[metric_name].append(value)
            category_aggregates[category][metric_name].append(value)

    results = {"overall": {}}
    for metric_name, values in aggregates.items():
        results["overall"][metric_name] = {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "count": len(values),
        }

    for category in sorted(category_aggregates.keys()):
        cat_key = f"category_{category}"
        results[cat_key] = {}
        for metric_name, values in category_aggregates[category].items():
            if values:
                results[cat_key][metric_name] = {
                    "mean": statistics.mean(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "count": len(values),
                }

    return results


def _lcs_length(a: List[str], b: List[str]) -> int:
    """计算最长公共子序列的长度。"""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def _get_bigrams(tokens: List[str]) -> List[tuple]:
    """从词元列表中获取二元组。"""
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
