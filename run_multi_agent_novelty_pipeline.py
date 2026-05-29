#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_multi_agent_novelty_pipeline.py

路线 A：claim-level novelty evaluation

输入:
1. target_papers.json
2. background_card_store/{paper_id}/background_cards.json

输出:
multi_agent_outputs/{paper_id}/
  - target_paper.json
  - planner.json
  - selector.json
  - judge.json
  - writer.json
  - critic.json
  - final_output.json
  - _complete.flag

说明:
- Planner: 保留 introduction 中的原始 novelty sentences
- Selector: 为每条原始 novelty sentence 选择相关背景卡片
- Judge: 结合背景文献，对每条原始 novelty sentence 做 polarity judgment
         并生成一条 claim-grounded evaluation sentence
- Writer: 只负责把 Judge 的 evaluation_sentence 按标签组织成三类
- Critic: 只做分析，不参与最终 metric 输出
"""

import re
import json
import math
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from pydantic import BaseModel, Field
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


# =========================================================
# 基础配置
# =========================================================

ALLOWED_TYPES = ["method", "dataset", "task", "application", "theory", "result"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "at", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "our", "we", "they",
    "he", "she", "his", "her", "them", "such", "than", "also", "can", "could",
    "may", "might", "will", "would", "should", "into", "about", "over", "under",
    "between", "after", "before", "during", "through", "using", "use", "used"
}


# =========================================================
# Pydantic schemas
# =========================================================

class NoveltyPoint(BaseModel):
    point_id: str = Field(description="Point identifier like P1, P2, ...")
    claim: str = Field(description="One originality-related sentence from the introduction.")
    novelty_type: List[Literal["method", "dataset", "task", "application", "theory", "result"]] = Field(
        min_length=1,
        max_length=3,
        description="1-3 novelty types from the allowed set."
    )
    importance: int = Field(description="Importance score from 1 to 5.")


class PlannerOutput(BaseModel):
    novelty_points: List[NoveltyPoint] = Field(
        min_length=1,
        max_length=8,
        description="A list of originality-related sentences preserved from the introduction."
    )


class SelectorDecision(BaseModel):
    source_title: str
    selected: bool
    relevance_reason: str
    relation_type: Literal["directly_related", "partially_related", "weakly_related"]


class SelectorOutput(BaseModel):
    decisions: List[SelectorDecision] = Field(
        min_length=1,
        max_length=20,
        description="Selection decisions for the candidate background cards."
    )


class JudgeOutput(BaseModel):
    novelty_label: Literal["positive", "neutral", "negative"]
    overlap_level: Literal["low", "medium", "high"]
    is_incremental: bool
    overall_novelty_implication: Literal[
        "strengthens_overall_novelty",
        "mixed",
        "limits_overall_novelty"
    ]
    rationale: str
    evidence_from_background: List[str] = Field(min_length=1, max_length=5)
    evaluation_sentence: str


class CriticOutput(BaseModel):
    overall_consistency: Literal["high", "medium", "low"]
    missing_points: List[str] = Field(default_factory=list)
    unsupported_judgments: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    paraphrase_warnings: List[str] = Field(default_factory=list)
    revision_suggestions: List[str] = Field(default_factory=list)

    revised_positive_novelty_evaluations: List[str] = Field(default_factory=list)
    revised_neutral_novelty_evaluations: List[str] = Field(default_factory=list)
    revised_negative_novelty_evaluations: List[str] = Field(default_factory=list)

    revised_final_review_text: str = Field(
        description=(
            "Render the revised final review strictly in the grouped format: "
            "Positive Novelty Evaluations / Neutral Novelty Evaluations / Negative Novelty Evaluations."
        )
    )


# =========================================================
# 工具函数
# =========================================================

def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def flatten_string_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        s = x.strip()
        return [s] if s else []
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str):
                s = item.strip()
                if s:
                    out.append(s)
            elif isinstance(item, dict):
                for k in ["text", "sentence", "claim", "content"]:
                    if k in item and item[k]:
                        out.append(str(item[k]).strip())
                        break
        return out
    return []


def normalize_string_list(x: Any) -> List[str]:
    vals = flatten_string_list(x)
    out = []
    seen = set()
    for v in vals:
        s = safe_str(v)
        if s:
            key = s.lower()
            if key not in seen:
                seen.add(key)
                out.append(s)
    return out


def normalize_type_list(x: Any) -> List[str]:
    vals = flatten_string_list(x)
    out = []
    for v in vals:
        v = v.lower().strip()
        if v in ALLOWED_TYPES and v not in out:
            out.append(v)
    return out


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()

    protected = {
        "e.g.": "eg<dot>",
        "i.e.": "ie<dot>",
        "et al.": "et al<dot>",
        "Fig.": "Fig<dot>",
        "Tab.": "Tab<dot>",
        "Eq.": "Eq<dot>",
        "Sec.": "Sec<dot>",
        "No.": "No<dot>"
    }
    for k, v in protected.items():
        text = text.replace(k, v)

    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])', text)

    sents = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        for k, v in protected.items():
            p = p.replace(v, k)
        p = re.sub(r"\s+", " ", p).strip()
        if len(p.split()) >= 4:
            sents.append(p)
    return sents


def tokenize(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", text.lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def token_overlap_score(a: str, b: str) -> float:
    sa = set(tokenize(a))
    sb = set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def cosine_similarity(a, b) -> float:
    if np is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        return json.loads(text)
    except Exception:
        return None


def card_text_for_ranking(card: Dict[str, Any]) -> str:
    parts = []
    for k in ["source_title", "summary", "card_text", "intro_excerpt", "source_abstract"]:
        v = safe_str(card.get(k, ""))
        if v:
            parts.append(v)
    for s in flatten_string_list(card.get("key_contribution_sentences", [])):
        parts.append(s)
    return normalize_text(" ".join(parts))


def render_grouped_novelty_review(
    positive_items: List[str],
    neutral_items: List[str],
    negative_items: List[str],
) -> str:
    def render_block(title: str, items: List[str]) -> str:
        lines = [title]
        cleaned = normalize_string_list(items)
        if cleaned:
            for x in cleaned:
                lines.append(f"- {x}")
        else:
            lines.append("-")
        return "\n".join(lines)

    blocks = [
        render_block("Positive Novelty Evaluations", positive_items),
        render_block("Neutral Novelty Evaluations", neutral_items),
        render_block("Negative Novelty Evaluations", negative_items),
    ]
    return "\n\n".join(blocks).strip()


# =========================================================
# vLLM engine
# =========================================================

class VLLMMultiAgentEngine:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        trust_remote_code: bool = True,
        dtype: str = "auto",
        temperature: float = 0.0,
        max_tokens: int = 512,
        enable_prefix_caching: bool = True,
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            trust_remote_code=trust_remote_code,
            dtype=dtype,
            enable_prefix_caching=enable_prefix_caching,
        )

    def make_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return f"System: {system_prompt}\n\nUser: {user_prompt}\n\nAssistant:"

    def generate_structured_batch(
        self,
        prompts: List[str],
        schema_model: type[BaseModel],
        max_tokens: Optional[int] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        schema = schema_model.model_json_schema()
        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=1.0,
            max_tokens=max_tokens or self.max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            obj = safe_json_loads(text)
            results.append(obj)
        return results


# =========================================================
# Agent prompts
# =========================================================

SELECTOR_SYSTEM_PROMPT = """You are a background-selection agent for novelty evaluation.
Your task is to decide which background cards are genuinely relevant to a target novelty claim.

Rules:
1. Prefer cards that are directly comparable to the target claim.
2. Prefer background cards that most strongly challenge the novelty of the target claim.
3. Distinguish direct relevance from weak lexical similarity.
4. selected=true only when the card is useful for judging novelty.
5. Output JSON only.
"""

JUDGE_SYSTEM_PROMPT = """You are a peer-review expert specializing in novelty evaluation of academic papers.
You will be given one originality-related sentence from the introduction of a paper, together with related background papers.

Your task is to judge the novelty of this specific claim relative to the provided background papers.

Label definitions:
- positive:
  the claim is clearly novel and substantially distinct from prior work.
- neutral:
  the claim shows some improvement or some originality, but is only moderately new relative to prior work.
- negative:
  the claim lacks substantial novelty, is only weakly differentiated from prior work, or is mainly incremental in nature.

In addition to the local novelty_label, you must also determine the claim's overall novelty implication:
- strengthens_overall_novelty
- mixed
- limits_overall_novelty

Rules:
1. novelty_label must be one of: positive, neutral, negative.
2. overlap_level must be one of: low, medium, high.
3. overall_novelty_implication must be one of:
   - strengthens_overall_novelty
   - mixed
   - limits_overall_novelty
4. Judge the provided claim itself rather than rewriting it into an abstract summary.
5. Stay close to the wording and technical content of the original claim.
6. evaluation_sentence must be a single concise reviewer-style evaluation sentence.
7. evaluation_sentence must preserve important technical terms from the original claim whenever possible.
8. evaluation_sentence must lightly evaluate the claim rather than simply copying it.
9. Do not mention names of prior methods, datasets, or papers in evaluation_sentence.
10. Do not produce paper-level summaries in evaluation_sentence.
11. Base your judgment on comparison with the provided background papers.
12. Output JSON only.
"""

CRITIC_SYSTEM_PROMPT = """You are a critic agent for a novelty-evaluation pipeline.
Your task is to audit the grouped novelty evaluations using the original novelty descriptions and point-level judgments.

Rules:
1. Do not redo the whole task from scratch.
2. Check whether any important claim is missing from the grouped output.
3. Check whether the grouped output contradicts the point-level judgments.
4. Check whether the grouped output drifts too far from the original novelty descriptions.
5. Output JSON only.
"""


# =========================================================
# Pipeline
# =========================================================

class MultiAgentNoveltyPipeline:
    def __init__(
        self,
        llm_engine: VLLMMultiAgentEngine,
        embed_model=None,
        selector_prefilter_k: int = 12,
        selector_final_k: int = 5,
    ):
        self.llm_engine = llm_engine
        self.embed_model = embed_model
        self.selector_prefilter_k = selector_prefilter_k
        self.selector_final_k = selector_final_k

    # -------------------------
    # Planner: preserve original novelty sentences
    # -------------------------
    def run_planner(self, target_paper: Dict[str, Any]) -> Dict[str, Any]:
        raw_novelty = target_paper.get("paper_novelty", "")
        novelty_texts = flatten_string_list(raw_novelty)

        sentences = []
        if novelty_texts:
            for item in novelty_texts:
                item = safe_str(item)
                if not item:
                    continue
                split_sents = split_sentences(item)
                if len(split_sents) <= 1:
                    sentences.append(item)
                else:
                    sentences.extend(split_sents)
        else:
            abstract = safe_str(target_paper.get("abstract", ""))
            sentences = split_sentences(abstract)

        cleaned = []
        seen = set()
        for s in sentences:
            s = safe_str(s)
            if not s:
                continue
            key = s.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(s)

        cleaned = cleaned[:8]

        final_types = normalize_type_list(target_paper.get("novelty_type", []))
        prelim_types = normalize_type_list(target_paper.get("novelty_type_preliminary", []))
        point_types = final_types or prelim_types or ["method"]

        novelty_points = []
        for i, s in enumerate(cleaned, start=1):
            novelty_points.append({
                "point_id": f"P{i}",
                "claim": s,
                "novelty_type": point_types[:3],
                "importance": 3
            })

        return {"novelty_points": novelty_points}

    # -------------------------
    # Prefilter
    # -------------------------
    def prefilter_cards(
        self,
        point_claim: str,
        point_types: List[str],
        background_cards: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        scored = []

        card_texts = [card_text_for_ranking(c) for c in background_cards]
        point_vec = None
        card_vecs = None

        if self.embed_model is not None and SentenceTransformer is not None and np is not None:
            try:
                point_vec = self.embed_model.encode([point_claim], convert_to_numpy=True, show_progress_bar=False)[0]
                card_vecs = self.embed_model.encode(card_texts, convert_to_numpy=True, show_progress_bar=False)
            except Exception:
                point_vec = None
                card_vecs = None

        for idx, card in enumerate(background_cards):
            ctext = card_texts[idx]
            tok = token_overlap_score(point_claim, ctext)
            sem = 0.0
            if point_vec is not None and card_vecs is not None:
                sem = cosine_similarity(point_vec, card_vecs[idx])

            card_types = normalize_type_list(card.get("novelty_type_guess", []))
            type_bonus = 1.0 if set(point_types) & set(card_types) else 0.0

            retrieval_prior = card.get("retrieval_score", 0.0)
            try:
                retrieval_prior = float(retrieval_prior)
            except Exception:
                retrieval_prior = 0.0

            if math.isnan(retrieval_prior) or math.isinf(retrieval_prior):
                retrieval_prior = 0.0
            retrieval_prior = max(0.0, min(1.0, retrieval_prior if retrieval_prior <= 1 else retrieval_prior / 10.0))

            score = 0.70 * sem + 0.15 * tok + 0.10 * type_bonus + 0.05 * retrieval_prior
            scored.append((score, card))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:self.selector_prefilter_k]]

    # -------------------------
    # Selector prompt builder
    # -------------------------
    def _build_selector_prompt(
        self,
        target_paper: Dict[str, Any],
        point: Dict[str, Any],
        candidate_cards: List[Dict[str, Any]],
    ) -> str:
        card_lines = []
        for i, c in enumerate(candidate_cards, start=1):
            card_lines.append(
                f"""[{i}]
Title: {safe_str(c.get("source_title", ""))}
Year: {c.get("source_year", "")}
Venue: {safe_str(c.get("source_venue", ""))}
Novelty Types: {normalize_type_list(c.get("novelty_type_guess", []))}
Summary: {safe_str(c.get("summary", ""))}
Key Contributions: {" | ".join(flatten_string_list(c.get("key_contribution_sentences", []))[:4])}
"""
            )
        card_block = "\n".join(card_lines)

        user_prompt = f"""
Target Paper Title:
{safe_str(target_paper.get("title", ""))}

Original Introduction Novelty Sentence:
{point["claim"]}

Target Novelty Types:
{point.get("novelty_type", [])}

Candidate Background Cards:
{card_block}

For each candidate card, decide whether it should be selected for novelty judgment.
Keep only genuinely useful and comparable cards.
""".strip()

        return self.llm_engine.make_prompt(SELECTOR_SYSTEM_PROMPT, user_prompt)

    # -------------------------
    # Selector batch
    # -------------------------
    def run_selector_batch(
        self,
        target_paper: Dict[str, Any],
        points: List[Dict[str, Any]],
        prefiltered_cards_per_point: List[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        prompts = []
        meta = []

        for point, candidate_cards in zip(points, prefiltered_cards_per_point):
            prompts.append(
                self._build_selector_prompt(
                    target_paper=target_paper,
                    point=point,
                    candidate_cards=candidate_cards,
                )
            )
            meta.append({
                "point": point,
                "candidate_cards": candidate_cards,
            })

        raw_results = self.llm_engine.generate_structured_batch(
            prompts=prompts,
            schema_model=SelectorOutput,
            max_tokens=768,
        )

        outputs = []
        for info, result in zip(meta, raw_results):
            point = info["point"]
            candidate_cards = info["candidate_cards"]

            selected_cards = []
            if result and "decisions" in result:
                title_to_card = {safe_str(c.get("source_title", "")): c for c in candidate_cards}
                for d in result["decisions"]:
                    if d.get("selected", False):
                        title = safe_str(d.get("source_title", ""))
                        if title in title_to_card:
                            card = dict(title_to_card[title])
                            card["_selector_decision"] = d
                            selected_cards.append(card)

            if not selected_cards:
                selected_cards = candidate_cards[:self.selector_final_k]

            selected_cards = selected_cards[:self.selector_final_k]

            outputs.append({
                "point_id": point["point_id"],
                "claim": point["claim"],
                "novelty_type": point.get("novelty_type", []),
                "selected_background_cards": selected_cards
            })

        return outputs

    # -------------------------
    # Judge prompt builder
    # -------------------------
    def _build_judge_prompt(
        self,
        target_paper: Dict[str, Any],
        point: Dict[str, Any],
        selected_cards: List[Dict[str, Any]],
    ) -> str:
        card_lines = []
        for i, c in enumerate(selected_cards, start=1):
            card_lines.append(
                f"""[{i}]
Title: {safe_str(c.get("source_title", ""))}
Year: {c.get("source_year", "")}
Venue: {safe_str(c.get("source_venue", ""))}
Summary: {safe_str(c.get("summary", ""))}
Key Contributions: {" | ".join(flatten_string_list(c.get("key_contribution_sentences", []))[:4])}
"""
            )
        card_block = "\n".join(card_lines)

        user_prompt = f"""
Target Paper Title:
{safe_str(target_paper.get("title", ""))}

Target Abstract:
{safe_str(target_paper.get("abstract", ""))}

Original Introduction Novelty Sentence:
{point["claim"]}

Target Novelty Types:
{point.get("novelty_type", [])}

Selected Background Cards:
{card_block}

Judge the novelty of this original introduction claim relative to the selected background cards.
""".strip()

        return self.llm_engine.make_prompt(JUDGE_SYSTEM_PROMPT, user_prompt)

    # -------------------------
    # Judge batch
    # -------------------------
    def run_judge_batch(
        self,
        target_paper: Dict[str, Any],
        selector_outputs: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        prompts = []
        meta = []

        for sel in selector_outputs:
            point = {
                "point_id": sel["point_id"],
                "claim": sel["claim"],
                "novelty_type": sel.get("novelty_type", []),
            }
            selected_cards = sel["selected_background_cards"]

            prompts.append(
                self._build_judge_prompt(
                    target_paper=target_paper,
                    point=point,
                    selected_cards=selected_cards,
                )
            )
            meta.append({
                "point": point,
                "selected_cards": selected_cards,
            })

        raw_results = self.llm_engine.generate_structured_batch(
            prompts=prompts,
            schema_model=JudgeOutput,
            max_tokens=768,
        )

        outputs = []
        for info, result in zip(meta, raw_results):
            point = info["point"]
            selected_cards = info["selected_cards"]

            if result is None:
                result = {
                    "novelty_label": "neutral",
                    "overlap_level": "medium",
                    "is_incremental": True,
                    "overall_novelty_implication": "mixed",
                    "rationale": "The claim appears only moderately novel relative to the provided background.",
                    "evidence_from_background": [
                        safe_str(c.get("source_title", ""))
                        for c in selected_cards[:3]
                        if safe_str(c.get("source_title", ""))
                    ],
                    "evaluation_sentence": f"{point['claim']} This appears only moderately novel."
                }

            outputs.append({
                "point_id": point["point_id"],
                "claim": point["claim"],
                "novelty_type": point.get("novelty_type", []),
                "selected_background_cards": selected_cards,
                "judgment": result
            })

        return outputs

    # -------------------------
    # Writer: only group judge-generated evaluation sentences
    # -------------------------
    def run_writer(
        self,
        target_paper: Dict[str, Any],
        planner_output: Dict[str, Any],
        judgments: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        pos, neu, neg = [], [], []

        for j in judgments:
            sent = safe_str(j["judgment"].get("evaluation_sentence", ""))
            label = safe_str(j["judgment"].get("novelty_label", "neutral")).lower()

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
            "overall_summary": "Claim-level novelty evaluations organized by polarity.",
            "positive_novelty_evaluations": pos,
            "neutral_novelty_evaluations": neu,
            "negative_novelty_evaluations": neg,
            "final_review_text": render_grouped_novelty_review(pos, neu, neg),
        }

    # -------------------------
    # Critic: analysis only
    # -------------------------
    def run_critic(
        self,
        target_paper: Dict[str, Any],
        planner_output: Dict[str, Any],
        judgments: List[Dict[str, Any]],
        writer_output: Dict[str, Any],
    ) -> Dict[str, Any]:
        paper_novelty_list = flatten_string_list(target_paper.get("paper_novelty", ""))
        paper_novelty = "\n".join(paper_novelty_list) if paper_novelty_list else safe_str(target_paper.get("paper_novelty", ""))

        point_blocks = []
        for j in judgments:
            jd = j["judgment"]
            point_blocks.append(
                f"""Original Novelty Sentence: {j["claim"]}
Novelty Label: {jd.get("novelty_label", "")}
Overlap Level: {jd.get("overlap_level", "")}
Incremental: {jd.get("is_incremental", False)}
Overall Novelty Implication: {jd.get("overall_novelty_implication", "")}
Rationale: {jd.get("rationale", "")}
Evaluation Sentence: {jd.get("evaluation_sentence", "")}
Evidence: {" | ".join(normalize_string_list(jd.get("evidence_from_background", [])))}
"""
            )
        point_block = "\n".join(point_blocks)

        writer_block = f"""
Positive Novelty Evaluations:
{" | ".join(normalize_string_list(writer_output.get("positive_novelty_evaluations", [])))}

Neutral Novelty Evaluations:
{" | ".join(normalize_string_list(writer_output.get("neutral_novelty_evaluations", [])))}

Negative Novelty Evaluations:
{" | ".join(normalize_string_list(writer_output.get("negative_novelty_evaluations", [])))}

Final Review Text:
{writer_output.get("final_review_text", "")}
""".strip()

        user_prompt = f"""
Target Paper Title:
{safe_str(target_paper.get("title", ""))}

Abstract:
{safe_str(target_paper.get("abstract", ""))}

Paper Novelty Description:
{paper_novelty}

Sentence-level Judgments:
{point_block}

Current Grouped Output:
{writer_block}

Check whether the grouped output misses important claims, contradicts the judgments,
or drifts too far from the original novelty descriptions.
""".strip()

        prompt = self.llm_engine.make_prompt(CRITIC_SYSTEM_PROMPT, user_prompt)
        result = self.llm_engine.generate_structured_batch(
            prompts=[prompt],
            schema_model=CriticOutput,
            max_tokens=1024,
        )[0]

        if result is None:
            writer_pos = normalize_string_list(writer_output.get("positive_novelty_evaluations", []))
            writer_neu = normalize_string_list(writer_output.get("neutral_novelty_evaluations", []))
            writer_neg = normalize_string_list(writer_output.get("negative_novelty_evaluations", []))

            result = {
                "overall_consistency": "medium",
                "missing_points": [],
                "unsupported_judgments": [],
                "contradictions": [],
                "paraphrase_warnings": [],
                "revision_suggestions": [],
                "revised_positive_novelty_evaluations": writer_pos,
                "revised_neutral_novelty_evaluations": writer_neu,
                "revised_negative_novelty_evaluations": writer_neg,
                "revised_final_review_text": render_grouped_novelty_review(
                    writer_pos, writer_neu, writer_neg
                )
            }
        else:
            pos = normalize_string_list(result.get("revised_positive_novelty_evaluations", []))
            neu = normalize_string_list(result.get("revised_neutral_novelty_evaluations", []))
            neg = normalize_string_list(result.get("revised_negative_novelty_evaluations", []))

            result["revised_positive_novelty_evaluations"] = pos
            result["revised_neutral_novelty_evaluations"] = neu
            result["revised_negative_novelty_evaluations"] = neg

            if not safe_str(result.get("revised_final_review_text", "")):
                result["revised_final_review_text"] = render_grouped_novelty_review(pos, neu, neg)

        return result

    # -------------------------
    # Full pipeline for one paper
    # -------------------------
    def run_for_paper(
        self,
        target_paper: Dict[str, Any],
        background_cards: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        planner_output = self.run_planner(target_paper)
        novelty_points = planner_output.get("novelty_points", [])

        if not novelty_points:
            empty_writer = {
                "overall_summary": "",
                "positive_novelty_evaluations": [],
                "neutral_novelty_evaluations": [],
                "negative_novelty_evaluations": [],
                "final_review_text": render_grouped_novelty_review([], [], [])
            }
            empty_critic = {
                "overall_consistency": "low",
                "missing_points": [],
                "unsupported_judgments": [],
                "contradictions": [],
                "paraphrase_warnings": [],
                "revision_suggestions": [],
                "revised_positive_novelty_evaluations": [],
                "revised_neutral_novelty_evaluations": [],
                "revised_negative_novelty_evaluations": [],
                "revised_final_review_text": render_grouped_novelty_review([], [], [])
            }
            return {
                "planner": {"novelty_points": []},
                "selector": [],
                "judge": [],
                "writer": empty_writer,
                "critic": empty_critic,
            }

        prefiltered_cards_per_point = []
        for point in novelty_points:
            prefiltered = self.prefilter_cards(
                point_claim=point["claim"],
                point_types=point.get("novelty_type", []),
                background_cards=background_cards
            )
            prefiltered_cards_per_point.append(prefiltered)

        selector_outputs = self.run_selector_batch(
            target_paper=target_paper,
            points=novelty_points,
            prefiltered_cards_per_point=prefiltered_cards_per_point,
        )

        judge_outputs = self.run_judge_batch(
            target_paper=target_paper,
            selector_outputs=selector_outputs,
        )

        writer_output = self.run_writer(
            target_paper=target_paper,
            planner_output=planner_output,
            judgments=judge_outputs,
        )

        critic_output = self.run_critic(
            target_paper=target_paper,
            planner_output=planner_output,
            judgments=judge_outputs,
            writer_output=writer_output,
        )

        return {
            "planner": planner_output,
            "selector": selector_outputs,
            "judge": judge_outputs,
            "writer": writer_output,
            "critic": critic_output,
        }


# =========================================================
# IO orchestration
# =========================================================

def process_one_paper(
    target_paper: Dict[str, Any],
    background_card_root: Path,
    output_root: Path,
    pipeline: MultiAgentNoveltyPipeline,
    overwrite: bool = False,
) -> None:
    paper_id = safe_str(target_paper.get("paper_id", ""))
    if not paper_id:
        print("[跳过] 缺少 paper_id")
        return

    bg_path = background_card_root / paper_id / "background_cards.json"
    if not bg_path.exists():
        print(f"[跳过] {paper_id} 缺少 background_cards.json")
        return

    out_dir = output_root / paper_id
    complete_flag = out_dir / "_complete.flag"
    final_path = out_dir / "final_output.json"

    if complete_flag.exists() and final_path.exists() and not overwrite:
        print(f"[跳过] 已完成: {paper_id}")
        return

    try:
        background_cards = read_json(bg_path)
    except Exception as e:
        print(f"[跳过] {paper_id} 读取 background_cards.json 失败: {e}")
        return

    if not isinstance(background_cards, list) or len(background_cards) == 0:
        print(f"[跳过] {paper_id} background_cards 为空")
        return

    result = pipeline.run_for_paper(target_paper, background_cards)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "target_paper.json", target_paper)
    write_json(out_dir / "planner.json", result["planner"])
    write_json(out_dir / "selector.json", result["selector"])
    write_json(out_dir / "judge.json", result["judge"])
    write_json(out_dir / "writer.json", result["writer"])
    write_json(out_dir / "critic.json", result["critic"])

    writer_pos = normalize_string_list(result["writer"].get("positive_novelty_evaluations", []))
    writer_neu = normalize_string_list(result["writer"].get("neutral_novelty_evaluations", []))
    writer_neg = normalize_string_list(result["writer"].get("negative_novelty_evaluations", []))

    final_positive = writer_pos
    final_neutral = writer_neu
    final_negative = writer_neg

    final_review_text = render_grouped_novelty_review(
        final_positive,
        final_neutral,
        final_negative,
    )

    write_json(out_dir / "final_output.json", {
        "paper_id": paper_id,
        "title": target_paper.get("title", ""),
        "planner": result["planner"],
        "judge": result["judge"],
        "writer": result["writer"],
        "critic": result["critic"],
        "positive_novelty_evaluations": final_positive,
        "neutral_novelty_evaluations": final_neutral,
        "negative_novelty_evaluations": final_negative,
        "final_review_text": final_review_text,
    })

    with open(complete_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[完成] {paper_id}")


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

    embed_model = None
    if SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(args.embed_model)
            print(f"[信息] 已加载 embedding 模型: {args.embed_model}")
        except Exception as e:
            print(f"[警告] embedding 模型加载失败，将仅使用规则粗排: {e}")
            embed_model = None
    else:
        print("[警告] 未安装 sentence-transformers，将仅使用规则粗排")

    print(f"[信息] 加载 vLLM 模型: {args.model}")
    llm_engine = VLLMMultiAgentEngine(
        model_name=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_tokens=args.max_new_tokens,
        enable_prefix_caching=True,
    )

    pipeline = MultiAgentNoveltyPipeline(
        llm_engine=llm_engine,
        embed_model=embed_model,
        selector_prefilter_k=args.selector_prefilter_k,
        selector_final_k=args.selector_final_k,
    )

    print(f"[信息] target papers 数量: {len(target_papers)}")
    for paper in target_papers:
        try:
            process_one_paper(
                target_paper=paper,
                background_card_root=background_card_root,
                output_root=output_root,
                pipeline=pipeline,
                overwrite=args.overwrite,
            )
        except Exception as e:
            pid = safe_str(paper.get("paper_id", "UNKNOWN"))
            print(f"[错误] {pid} 处理失败: {e}")

    print("[全部完成]")


if __name__ == "__main__":
    main()