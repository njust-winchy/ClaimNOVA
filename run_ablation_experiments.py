#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_ablation_experiments.py

用途：
对当前 claim-level novelty evaluation pipeline 跑核心消融实验。

依赖：
- 你当前的主脚本：run_multi_agent_novelty_pipeline.py
- 其中应包含：
    - MultiAgentNoveltyPipeline
    - VLLMMultiAgentEngine
    - read_json
    - write_json
    - safe_str
    - normalize_string_list
    - SentenceTransformer（若已安装）
"""

import json
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
# 消融配置
# =========================================================

@dataclass
class AblationConfig:
    name: str
    use_background: bool = True
    use_selector: bool = True
    use_novelty_type: bool = True
    use_eval_sentence: bool = True


ABLATIONS = [
    AblationConfig(name="full", use_background=True, use_selector=True, use_novelty_type=True, use_eval_sentence=True),
    AblationConfig(name="wo_background", use_background=False, use_selector=False, use_novelty_type=True, use_eval_sentence=True),
    AblationConfig(name="wo_selector", use_background=True, use_selector=False, use_novelty_type=True, use_eval_sentence=True),
    AblationConfig(name="wo_novelty_type", use_background=True, use_selector=True, use_novelty_type=False, use_eval_sentence=True),
    AblationConfig(name="wo_eval_sentence", use_background=True, use_selector=True, use_novelty_type=True, use_eval_sentence=False),
]


# =========================================================
# 消融版 Pipeline
# =========================================================

class AblationPipeline(MultiAgentNoveltyPipeline):
    def __init__(
        self,
        llm_engine: VLLMMultiAgentEngine,
        ablation: AblationConfig,
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
        self.ablation = ablation

    # -------------------------
    # Planner: 可选禁用 novelty_type
    # -------------------------
    def run_planner(self, target_paper: Dict[str, Any]) -> Dict[str, Any]:
        planner_output = super().run_planner(target_paper)

        if not self.ablation.use_novelty_type:
            for p in planner_output.get("novelty_points", []):
                p["novelty_type"] = []
        return planner_output

    # -------------------------
    # Selector batch:
    # - wo_background: 不用背景
    # - wo_selector: 跳过 LLM selector，直接用 prefilter top-k
    # -------------------------
    def run_selector_batch(
        self,
        target_paper: Dict[str, Any],
        points: List[Dict[str, Any]],
        prefiltered_cards_per_point: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        # wo_background: 每个 point 没有背景卡片
        if not self.ablation.use_background:
            outputs = []
            for point in points:
                outputs.append({
                    "point_id": point["point_id"],
                    "claim": point["claim"],
                    "novelty_type": point.get("novelty_type", []),
                    "selected_background_cards": [],
                })
            return outputs

        # wo_selector: 跳过 LLM selector，直接用 prefilter top-k
        if not self.ablation.use_selector:
            outputs = []
            for point, candidate_cards in zip(points, prefiltered_cards_per_point):
                outputs.append({
                    "point_id": point["point_id"],
                    "claim": point["claim"],
                    "novelty_type": point.get("novelty_type", []),
                    "selected_background_cards": candidate_cards[:self.selector_final_k],
                })
            return outputs

        # full / 其他情况：照常调用父类
        return super().run_selector_batch(
            target_paper=target_paper,
            points=points,
            prefiltered_cards_per_point=prefiltered_cards_per_point,
        )

    # -------------------------
    # Judge batch:
    # - wo_eval_sentence: 忽略 Judge 的 evaluation_sentence，
    #   让 Writer 退回原 claim 分组
    # -------------------------
    def run_judge_batch(
        self,
        target_paper: Dict[str, Any],
        selector_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        judge_outputs = super().run_judge_batch(
            target_paper=target_paper,
            selector_outputs=selector_outputs,
        )

        if not self.ablation.use_eval_sentence:
            for item in judge_outputs:
                if "judgment" in item:
                    item["judgment"]["evaluation_sentence"] = ""

        return judge_outputs

    # -------------------------
    # Writer:
    # - full: 用 Judge 的 evaluation_sentence
    # - wo_eval_sentence: 用原 claim 直接分组
    # -------------------------
    def run_writer(
        self,
        target_paper: Dict[str, Any],
        planner_output: Dict[str, Any],
        judgments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pos, neu, neg = [], [], []

        for j in judgments:
            label = safe_str(j["judgment"].get("novelty_label", "neutral")).lower()

            if self.ablation.use_eval_sentence:
                sent = safe_str(j["judgment"].get("evaluation_sentence", ""))
                if not sent:
                    sent = safe_str(j.get("claim", ""))
            else:
                sent = safe_str(j.get("claim", ""))

            if not sent:
                continue

            if label == "positive":
                pos.append(sent)
            elif label == "negative":
                neg.append(sent)
            else:
                neu.append(sent)

        pos = normalize_string_list(pos)
        neu = normalize_string_list(neu)
        neg = normalize_string_list(neg)

        return {
            "overall_summary": f"Ablation variant: {self.ablation.name}",
            "positive_novelty_evaluations": pos,
            "neutral_novelty_evaluations": neu,
            "negative_novelty_evaluations": neg,
            "final_review_text": self._render_grouped(pos, neu, neg),
        }

    @staticmethod
    def _render_grouped(pos: List[str], neu: List[str], neg: List[str]) -> str:
        def render_block(title: str, items: List[str]) -> str:
            lines = [title]
            if items:
                for x in items:
                    lines.append(f"- {x}")
            else:
                lines.append("-")
            return "\n".join(lines)

        return "\n\n".join([
            render_block("Positive Novelty Evaluations", pos),
            render_block("Neutral Novelty Evaluations", neu),
            render_block("Negative Novelty Evaluations", neg),
        ])


# =========================================================
# 评测 Hook
# =========================================================

def evaluate_variant_outputs(
    variant_output_dir: Path,
    target_papers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    这里先给一个占位版本。
    你可以把你那篇已录用文章里的四项指标评测函数挂进来。

    期望输出:
    {
      "num_papers": ...,
      "Rel.": ...,
      "Cov.": ...,
      "Clarity": ...,
      "DistAcc": ...
    }

    当前默认只统计成功跑完的 paper 数量。
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
# 单个变体运行
# =========================================================

def process_one_paper(
    target_paper: Dict[str, Any],
    background_card_root: Path,
    output_root: Path,
    pipeline: AblationPipeline,
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
        print(f"[跳过] {pipeline.ablation.name}: 已完成 {paper_id}")
        return

    background_cards = []
    if pipeline.ablation.use_background:
        if not bg_path.exists():
            print(f"[跳过] {pipeline.ablation.name}: {paper_id} 缺少 background_cards.json")
            return
        try:
            background_cards = read_json(bg_path)
        except Exception as e:
            print(f"[跳过] {pipeline.ablation.name}: {paper_id} 读取 background_cards.json 失败: {e}")
            return

        if not isinstance(background_cards, list):
            print(f"[跳过] {pipeline.ablation.name}: {paper_id} background_cards 不是 list")
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
        "ablation": asdict(pipeline.ablation),
        "positive_novelty_evaluations": final_positive,
        "neutral_novelty_evaluations": final_neutral,
        "negative_novelty_evaluations": final_negative,
        "final_review_text": result["writer"].get("final_review_text", ""),
    })

    with open(complete_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[完成] {pipeline.ablation.name}: {paper_id}")


def run_one_ablation(
    ablation: AblationConfig,
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
    print(f"\n========== Running ablation: {ablation.name} ==========")

    embed_model = None
    if embed_model_name and SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(embed_model_name)
            print(f"[信息] {ablation.name}: 已加载 embedding 模型 {embed_model_name}")
        except Exception as e:
            print(f"[警告] {ablation.name}: embedding 模型加载失败，将仅使用规则粗排: {e}")

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

    pipeline = AblationPipeline(
        llm_engine=llm_engine,
        ablation=ablation,
        embed_model=embed_model,
        selector_prefilter_k=selector_prefilter_k,
        selector_final_k=selector_final_k,
    )

    variant_output_dir = output_root / ablation.name
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
            print(f"[错误] {ablation.name}: {pid} 处理失败: {e}")

    metrics = evaluate_variant_outputs(variant_output_dir, target_papers)
    summary = {
        "ablation": asdict(ablation),
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
        default="full,wo_background,wo_selector,wo_novelty_type,wo_eval_sentence",
        help="Comma-separated ablation variants to run."
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
    selected_ablations = [a for a in ABLATIONS if a.name in requested]

    if not selected_ablations:
        raise ValueError(f"未匹配到任何 ablation variants: {requested}")

    all_summaries = []
    for ablation in selected_ablations:
        summary = run_one_ablation(
            ablation=ablation,
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

    write_json(output_root / "ablation_summary_all.json", all_summaries)
    print("\n[全部消融完成]")


if __name__ == "__main__":
    main()