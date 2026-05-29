#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_novelty_type_stratified_analysis.py

用途：
对某个实验输出目录，按 novelty_type 做分层切分，生成后续评测需要的子集文件。

注意：
- 本脚本只保留 paper_id 为纯数字的样本。
- 例如:
  "123"    -> 保留，并转换为 int: 123
  "00123"  -> 保留，并转换为 int: 123
  "paper_123" -> 跳过
  "ACL2024_123" -> 跳过

输入:
1. target_papers.json
2. experiment_output_dir
   例如：
   - multi_agent_outputs
   - ablation_outputs/full
   - background_evidence_outputs/selected_background

输出:
output_dir/
  - summary.json
  - summary.csv
  - per_type/
      - method/
          - paper_ids.json
          - target_papers_subset.json
          - predictions_subset.json
          - evaluation_input.json
      - dataset/
      - task/
      - application/
      - theory/
      - result/

evaluation_input.json 中会同时保留：
- target paper metadata
- novelty_type
- gold-related fields（如果 target_papers.json 里有）
- prediction-related fields（来自 final_output.json）

你后续可以直接在每个 type 子目录上跑你已有的四项指标代码。
"""

import csv
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List

ALLOWED_TYPES = ["method", "dataset", "task", "application", "theory", "result"]


# =========================================================
# 基础工具
# =========================================================

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def is_pure_numeric_id(x: Any) -> bool:
    """
    只接受纯数字形式的 paper_id。
    例如:
    - "123" -> True
    - "00123" -> True
    - "paper_123" -> False
    - "ACL2024_123" -> False
    """
    s = safe_str(x)
    return bool(s) and s.isdigit()


def to_numeric_paper_id(x: Any) -> int:
    """
    将纯数字 paper_id 转换为 int。
    非纯数字 ID 会报错。
    """
    s = safe_str(x)
    if not is_pure_numeric_id(s):
        raise ValueError(f"paper_id is not pure numeric: {s}")
    return int(s)


def normalize_string_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out = []
        seen = set()
        for item in x:
            s = safe_str(item)
            if s:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    out.append(s)
        return out
    return []


def normalize_type_list(x: Any) -> List[str]:
    vals = normalize_string_list(x)
    out = []
    for v in vals:
        v = v.lower()
        if v in ALLOWED_TYPES and v not in out:
            out.append(v)
    return out


# =========================================================
# 主逻辑
# =========================================================

def get_bucket_types(
    paper: Dict[str, Any],
    mode: str = "any",
) -> List[str]:
    """
    mode = "any": 一篇 paper 可进入多个 novelty_type bucket
    mode = "primary": 只取第一个 novelty_type
    """
    types = normalize_type_list(paper.get("novelty_type", []))
    if not types:
        types = normalize_type_list(paper.get("novelty_type_preliminary", []))

    if not types:
        return []

    if mode == "primary":
        return [types[0]]

    return types


def build_prediction_record(
    paper: Dict[str, Any],
    final_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    构造一个统一的 prediction record，便于后续跑四项指标。

    注意：
    - paper_id 输出为 int
    - paper_id_original 保留原始字符串，方便回查文件路径
    """
    paper_id_original = safe_str(paper.get("paper_id", ""))
    paper_id_numeric = to_numeric_paper_id(paper_id_original)

    return {
        "paper_id": paper_id_numeric,
        "paper_id_original": paper_id_original,
        "title": paper.get("title", ""),
        "venue": paper.get("venue", ""),
        "year": paper.get("year", None),

        # 原始输入 / gold 相关
        "paper_novelty": paper.get("paper_novelty", ""),
        "review_novelty": paper.get("review_novelty", ""),
        "review": paper.get("review", ""),
        "novelty_type": paper.get("novelty_type", []),
        "novelty_type_preliminary": paper.get("novelty_type_preliminary", []),

        # 预测结果
        "positive_novelty_evaluations": final_output.get("positive_novelty_evaluations", []),
        "neutral_novelty_evaluations": final_output.get("neutral_novelty_evaluations", []),
        "negative_novelty_evaluations": final_output.get("negative_novelty_evaluations", []),
        "final_review_text": final_output.get("final_review_text", ""),

        # 保留原完整结果，便于后续分析
        "final_output_path": final_output.get("_source_path", ""),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_papers_json", type=str, required=True)
    parser.add_argument("--experiment_output_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        default="any",
        choices=["any", "primary"],
        help="Grouping mode for multi-label novelty_type."
    )
    parser.add_argument(
        "--require_prediction",
        action="store_true",
        help="If set, only keep papers whose final_output.json exists."
    )

    args = parser.parse_args()

    target_papers_path = Path(args.target_papers_json)
    experiment_output_dir = Path(args.experiment_output_dir)
    output_dir = Path(args.output_dir)

    if not target_papers_path.exists():
        raise FileNotFoundError(f"target_papers_json 不存在: {target_papers_path}")
    if not experiment_output_dir.exists():
        raise FileNotFoundError(f"experiment_output_dir 不存在: {experiment_output_dir}")

    target_papers = read_json(target_papers_path)
    if not isinstance(target_papers, list):
        raise ValueError("target_papers_json 必须是 list")

    # 每个 type 对应一个 bucket
    bucket_papers: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ALLOWED_TYPES}
    bucket_predictions: Dict[str, List[Dict[str, Any]]] = {t: [] for t in ALLOWED_TYPES}

    missing_prediction_ids: List[int] = []
    no_type_ids: List[int] = []
    non_numeric_ids: List[str] = []

    total_seen = 0
    total_numeric_kept = 0

    for paper in target_papers:
        total_seen += 1

        paper_id_str = safe_str(paper.get("paper_id", ""))
        if not paper_id_str:
            non_numeric_ids.append("")
            continue

        # 只保留纯数字 paper_id
        if not is_pure_numeric_id(paper_id_str):
            non_numeric_ids.append(paper_id_str)
            continue

        paper_id_num = to_numeric_paper_id(paper_id_str)
        total_numeric_kept += 1

        bucket_types = get_bucket_types(paper, mode=args.mode)
        if not bucket_types:
            no_type_ids.append(paper_id_num)
            continue

        # 注意：路径仍然使用原始字符串 ID
        # 因为你的输出目录名可能仍然是 "00123" 而不是 123
        final_output_path = experiment_output_dir / paper_id_str / "final_output.json"
        has_prediction = final_output_path.exists()

        if args.require_prediction and not has_prediction:
            missing_prediction_ids.append(paper_id_num)
            continue

        final_output: Dict[str, Any] = {}
        if has_prediction:
            final_output = read_json(final_output_path)
            if isinstance(final_output, dict):
                final_output["_source_path"] = str(final_output_path)
            else:
                final_output = {}

        for t in bucket_types:
            bucket_papers[t].append(paper)

            if has_prediction:
                pred_record = build_prediction_record(paper, final_output)
                bucket_predictions[t].append(pred_record)

    # 写 summary
    summary_rows = []
    summary_json = {
        "mode": args.mode,
        "target_papers_json": str(target_papers_path),
        "experiment_output_dir": str(experiment_output_dir),

        "total_seen": total_seen,
        "total_numeric_kept": total_numeric_kept,
        "num_non_numeric_skipped": len(non_numeric_ids),
        "non_numeric_ids": non_numeric_ids,

        "num_missing_predictions": len(missing_prediction_ids),
        "missing_prediction_ids": missing_prediction_ids,

        "num_no_type": len(no_type_ids),
        "no_type_ids": no_type_ids,

        "types": {},
    }

    for t in ALLOWED_TYPES:
        num_papers = len(bucket_papers[t])
        num_predictions = len(bucket_predictions[t])

        summary_json["types"][t] = {
            "num_papers": num_papers,
            "num_predictions": num_predictions,
        }

        summary_rows.append({
            "type": t,
            "num_papers": num_papers,
            "num_predictions": num_predictions,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "summary.json", summary_json)

    with open(output_dir / "summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["type", "num_papers", "num_predictions"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    # 写每个 type 的子集
    per_type_dir = output_dir / "per_type"
    for t in ALLOWED_TYPES:
        t_dir = per_type_dir / t
        t_dir.mkdir(parents=True, exist_ok=True)

        paper_ids = [
            to_numeric_paper_id(p.get("paper_id", ""))
            for p in bucket_papers[t]
            if is_pure_numeric_id(p.get("paper_id", ""))
        ]

        write_json(t_dir / "paper_ids.json", paper_ids)
        write_json(t_dir / "target_papers_subset.json", bucket_papers[t])
        write_json(t_dir / "predictions_subset.json", bucket_predictions[t])

        # evaluation_input: 后续你算四项指标时最方便直接读这个
        evaluation_input = {
            "type": t,
            "mode": args.mode,
            "num_papers": len(bucket_papers[t]),
            "num_predictions": len(bucket_predictions[t]),
            "records": bucket_predictions[t],
        }
        write_json(t_dir / "evaluation_input.json", evaluation_input)

    print("[完成] novelty type 分层统计文件已生成")
    print(f"输出目录: {output_dir}")
    print(f"总样本数: {total_seen}")
    print(f"保留纯数字 ID 样本数: {total_numeric_kept}")
    print(f"跳过非纯数字 ID 样本数: {len(non_numeric_ids)}")
    print(f"缺失预测样本数: {len(missing_prediction_ids)}")
    print(f"无 novelty_type 样本数: {len(no_type_ids)}")


if __name__ == "__main__":
    main()