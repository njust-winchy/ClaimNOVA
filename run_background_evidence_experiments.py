#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_background_evidence_experiments.py

背景证据有效性实验：
1. claim_only
2. retrieved_background
3. selected_background
4. random_background

依赖：
- run_multi_agent_novelty_pipeline.py
"""

import json
import random
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from run_multi_agent_novelty_pipeline import (
    MultiAgentNoveltyPipeline,
    VLLMMultiAgentEngine,
    read_json,
    write_json,
    safe_str,
    normalize_string_list,
    SentenceTransformer,
)


# =========================================================
# 实验配置
# =========================================================

@dataclass
class BackgroundEvidenceConfig:
    name: str
    mode: str  # claim_only / retrieved_background / selected_background / random_background


EXPERIMENTS = [
    BackgroundEvidenceConfig(name="claim_only", mode="claim_only"),
    BackgroundEvidenceConfig(name="retrieved_background", mode="retrieved_background"),
    BackgroundEvidenceConfig(name="selected_background", mode="selected_background"),
    BackgroundEvidenceConfig(name="random_background", mode="random_background"),
]


# =========================================================
# 工具函数
# =========================================================

def stable_seed_from_text(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_random_cards(
    cards: List[Dict[str, Any]],
    k: int,
    seed_text: str,
) -> List[Dict[str, Any]]:
    if not cards:
        return []
    rnd = random.Random(stable_seed_from_text(seed_text))
    if len(cards) <= k:
        out = cards[:]
        rnd.shuffle(out)
        return out
    idxs = list(range(len(cards)))
    rnd.shuffle(idxs)
    idxs = idxs[:k]
    return [cards[i] for i in idxs]


# =========================================================
# Pipeline
# =========================================================

class BackgroundEvidencePipeline(MultiAgentNoveltyPipeline):
    def __init__(
        self,
        llm_engine: VLLMMultiAgentEngine,
        experiment: BackgroundEvidenceConfig,
        embed_model=None,
        selector_prefilter_k: int = 12,
        selector_final_k: int = 5,
    ):
        super().__init__(
            llm_engine=llm_engine,
            embed_model=embed_model,
            selector_prefilter_k=selector_prefilter_k,
            selector_final_k=selector_final_k,
        )
        self.experiment = experiment
        self._current_background_cards: List[Dict[str, Any]] = []
        self._current_paper_id: str = ""

    def run_for_paper(
        self,
        target_paper: Dict[str, Any],
        background_cards: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._current_background_cards = background_cards
        self._current_paper_id = safe_str(target_paper.get("paper_id", ""))
        return super().run_for_paper(target_paper, background_cards)

    def run_selector_batch(
        self,
        target_paper: Dict[str, Any],
        points: List[Dict[str, Any]],
        prefiltered_cards_per_point: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        mode = self.experiment.mode

        # 1) claim_only: 完全不给背景
        if mode == "claim_only":
            outputs = []
            for point in points:
                outputs.append({
                    "point_id": point["point_id"],
                    "claim": point["claim"],
                    "novelty_type": point.get("novelty_type", []),
                    "selected_background_cards": [],
                })
            return outputs

        # 2) retrieved_background: 直接用粗筛 top-k，不做 selector
        if mode == "retrieved_background":
            outputs = []
            for point, candidate_cards in zip(points, prefiltered_cards_per_point):
                outputs.append({
                    "point_id": point["point_id"],
                    "claim": point["claim"],
                    "novelty_type": point.get("novelty_type", []),
                    "selected_background_cards": candidate_cards[:self.selector_final_k],
                })
            return outputs

        # 3) random_background: 从整篇 paper 的全部背景卡片里随机取 k 张
        if mode == "random_background":
            outputs = []
            for point in points:
                seed_text = f"{self._current_paper_id}::{point['point_id']}::{point['claim']}"
                random_cards = sample_random_cards(
                    self._current_background_cards,
                    self.selector_final_k,
                    seed_text=seed_text,
                )
                outputs.append({
                    "point_id": point["point_id"],
                    "claim": point["claim"],
                    "novelty_type": point.get("novelty_type", []),
                    "selected_background_cards": random_cards,
                })
            return outputs

        # 4) selected_background: 完整方法
        if mode == "selected_background":
            return super().run_selector_batch(
                target_paper=target_paper,
                points=points,
                prefiltered_cards_per_point=prefiltered_cards_per_point,
            )

        raise ValueError(f"Unknown experiment mode: {mode}")


# =========================================================
# 评测 Hook（占位）
# =========================================================

def evaluate_variant_outputs(
    variant_output_dir: Path,
    target_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    占位函数。
    你后面自己用你那篇录用文章的四项指标去算即可。

    当前只统计跑完了多少篇。
    """
    completed = 0
    for paper in target_papers:
        pid = safe_str(paper.get("paper_id", ""))
        if not pid:
            continue
        final_path = variant_output_dir / pid / "final_output.json"
        if final_path.exists():
            completed += 1

    return {
        "num_papers": completed,
        "Rel.": None,
        "Cov.": None,
        "Clarity": None,
        "DistAcc": None,
    }


# =========================================================
# 单篇处理
# =========================================================

def process_one_paper(
    target_paper: Dict[str, Any],
    background_card_root: Path,
    output_root: Path,
    pipeline: BackgroundEvidencePipeline,
    overwrite: bool = False,
) -> None:
    paper_id = safe_str(target_paper.get("paper_id", ""))
    if not paper_id:
        print("[跳过] 缺少 paper_id")
        return

    bg_path = background_card_root / paper_id / "background_cards.json"
    out_dir = output_root / paper_id
    complete_flag = out_dir / "_complete.flag"
    final_path = out_dir / "final_output.json"

    if complete_flag.exists() and final_path.exists() and not overwrite:
        print(f"[跳过] {pipeline.experiment.name}: 已完成 {paper_id}")
        return

    background_cards = []
    if pipeline.experiment.mode != "claim_only":
        if not bg_path.exists():
            print(f"[跳过] {pipeline.experiment.name}: {paper_id} 缺少 background_cards.json")
            return
        try:
            background_cards = read_json(bg_path)
        except Exception as e:
            print(f"[跳过] {pipeline.experiment.name}: {paper_id} 读取 background_cards.json 失败: {e}")
            return

        if not isinstance(background_cards, list):
            print(f"[跳过] {pipeline.experiment.name}: {paper_id} background_cards 不是 list")
            return

    result = pipeline.run_for_paper(target_paper, background_cards)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target_paper.json", target_paper)
    write_json(out_dir / "planner.json", result["planner"])
    write_json(out_dir / "selector.json", result["selector"])
    write_json(out_dir / "judge.json", result["judge"])
    write_json(out_dir / "writer.json", result["writer"])
    write_json(out_dir / "critic.json", result["critic"])

    final_positive = normalize_string_list(result["writer"].get("positive_novelty_evaluations", []))
    final_neutral = normalize_string_list(result["writer"].get("neutral_novelty_evaluations", []))
    final_negative = normalize_string_list(result["writer"].get("negative_novelty_evaluations", []))

    write_json(out_dir / "final_output.json", {
        "paper_id": paper_id,
        "title": target_paper.get("title", ""),
        "experiment": asdict(pipeline.experiment),
        "positive_novelty_evaluations": final_positive,
        "neutral_novelty_evaluations": final_neutral,
        "negative_novelty_evaluations": final_negative,
        "final_review_text": result["writer"].get("final_review_text", ""),
    })

    with open(complete_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[完成] {pipeline.experiment.name}: {paper_id}")


# =========================================================
# 单个实验运行
# =========================================================

def run_one_experiment(
    experiment: BackgroundEvidenceConfig,
    target_papers: List[Dict[str, Any]],
    background_card_root: Path,
    output_root: Path,
    model_name: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_new_tokens: int,
    trust_remote_code: bool,
    dtype: str,
    embed_model_name: Optional[str],
    selector_prefilter_k: int,
    selector_final_k: int,
    overwrite: bool,
) -> Dict[str, Any]:
    print(f"\n========== Running experiment: {experiment.name} ==========")

    embed_model = None
    if embed_model_name and SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(embed_model_name)
            print(f"[信息] {experiment.name}: 已加载 embedding 模型 {embed_model_name}")
        except Exception as e:
            print(f"[警告] {experiment.name}: embedding 模型加载失败，将仅使用规则粗排: {e}")

    llm_engine = VLLMMultiAgentEngine(
        model_name=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        trust_remote_code=trust_remote_code,
        dtype=dtype,
        max_tokens=max_new_tokens,
        enable_prefix_caching=True,
    )

    pipeline = BackgroundEvidencePipeline(
        llm_engine=llm_engine,
        experiment=experiment,
        embed_model=embed_model,
        selector_prefilter_k=selector_prefilter_k,
        selector_final_k=selector_final_k,
    )

    variant_output_dir = output_root / experiment.name
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    for paper in target_papers:
        try:
            process_one_paper(
                target_paper=paper,
                background_card_root=background_card_root,
                output_root=variant_output_dir,
                pipeline=pipeline,
                overwrite=overwrite,
            )
        except Exception as e:
            pid = safe_str(paper.get("paper_id", "UNKNOWN"))
            print(f"[错误] {experiment.name}: {pid} 处理失败: {e}")

    metrics = evaluate_variant_outputs(variant_output_dir, target_papers)
    summary = {
        "experiment": asdict(experiment),
        "metrics": metrics,
    }
    write_json(variant_output_dir / "summary.json", summary)
    return summary


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_papers_json", type=str, required=True)
    parser.add_argument("--background_card_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--selector_prefilter_k", type=int, default=12)
    parser.add_argument("--selector_final_k", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument(
        "--variants",
        type=str,
        default="claim_only,retrieved_background,selected_background,random_background",
        help="Comma-separated experiment variants to run."
    )

    args = parser.parse_args()

    target_papers_path = Path(args.target_papers_json)
    background_card_root = Path(args.background_card_dir)
    output_root = Path(args.output_dir)

    if not target_papers_path.exists():
        raise FileNotFoundError(f"target_papers_json 不存在: {target_papers_path}")
    if not background_card_root.exists():
        raise FileNotFoundError(f"background_card_dir 不存在: {background_card_root}")

    target_papers = read_json(target_papers_path)
    if not isinstance(target_papers, list):
        raise ValueError("target_papers_json 必须是 list")

    requested = [x.strip() for x in args.variants.split(",") if x.strip()]
    selected_experiments = [e for e in EXPERIMENTS if e.name in requested]

    if not selected_experiments:
        raise ValueError(f"未匹配到任何 experiment variants: {requested}")

    all_summaries = []
    for experiment in selected_experiments:
        summary = run_one_experiment(
            experiment=experiment,
            target_papers=target_papers,
            background_card_root=background_card_root,
            output_root=output_root,
            model_name=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
            trust_remote_code=args.trust_remote_code,
            dtype=args.dtype,
            embed_model_name=args.embed_model,
            selector_prefilter_k=args.selector_prefilter_k,
            selector_final_k=args.selector_final_k,
            overwrite=args.overwrite,
        )
        all_summaries.append(summary)

    write_json(output_root / "background_evidence_summary_all.json", all_summaries)
    print("\n[背景证据有效性实验全部完成]")


if __name__ == "__main__":
    main()