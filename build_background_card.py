#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_background_cards.py

将 background_raw_store/{paper_id}/background_papers.json 中的背景论文全文
压缩成结构化 background cards，供后续 novelty point-level matching 使用。

输入目录结构:
background_raw_store/{paper_id}/
  - target_paper.json
  - background_papers.json
  - _build_complete.flag

输出目录结构:
background_card_store/{paper_id}/
  - target_paper.json
  - background_cards.json
  - _build_complete.flag

每条 background card 大致格式:
{
  "source_title": "...",
  "source_year": 2023,
  "source_venue": "EMNLP",
  "retrieval_score": 0.60,
  "source_url": "...",
  "pdf_file": "...pdf",
  "source_abstract": "...",
  "intro_excerpt": "...",
  "candidate_contribution_sentences": ["...", "..."],
  "summary": "...",
  "key_contribution_sentences": ["...", "...", "..."],
  "novelty_type_guess": ["method", "result"],
  "card_text": "..."
}

运行示例:
python build_background_cards.py \
  --input_dir background_raw_store \
  --output_dir background_card_store \
  --card_model Qwen/Qwen2.5-7B-Instruct \
  --embed_model sentence-transformers/all-MiniLM-L6-v2 \
  --tensor_parallel_size 1 \
  --max_model_len 8192 \
  --max_new_tokens 256 \
  --llm_batch_size 8 \
  --trust_remote_code
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Literal

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

CARD_TYPES = ["method", "dataset", "task", "application", "theory", "result"]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "with", "by",
    "at", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its", "their", "our", "we", "they",
    "he", "she", "his", "her", "them", "such", "than", "also", "can", "could",
    "may", "might", "will", "would", "should", "into", "about", "over", "under",
    "between", "after", "before", "during", "through", "using", "use", "used"
}

SECTION_PATTERNS = {
    "abstract": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*abstract\s*$", re.I),
    "introduction": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*introduction\s*$", re.I),
    "related work": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*(related work|background|preliminaries)\s*$", re.I),
    "method": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*(method|methods|methodology|approach|model|models)\s*$", re.I),
    "experiment": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*(experiments?|experimental setup|evaluation|results?|analysis|discussion)\s*$", re.I),
    "conclusion": re.compile(r"^\s*(?:\d+(?:\.\d+)*)?\s*(conclusion|conclusions|limitations?)\s*$", re.I),
    "references": re.compile(r"^\s*(?:references|bibliography)\s*$", re.I),
    "appendix": re.compile(r"^\s*(?:appendix|appendices)\s*$", re.I),
}

TYPE_PATTERNS = {
    "method": [
        r"\b(method|approach|framework|model|architecture|algorithm|pipeline|training strategy|retrieval strategy)\b",
        r"\bwe propose\b",
        r"\bwe present\b",
        r"\bwe introduce\b",
    ],
    "dataset": [
        r"\b(dataset|corpus|benchmark|resource|annotation|annotated|data collection)\b",
        r"\bwe release\b",
        r"\bnew dataset\b",
    ],
    "task": [
        r"\b(task|problem|formulation|setting|objective)\b",
        r"\bnew task\b",
        r"\btask formulation\b",
    ],
    "application": [
        r"\b(application|applied|real-world|practical|domain-specific|clinical|biomedical|education|legal|financial)\b",
    ],
    "theory": [
        r"\b(theory|theoretical|proof|theorem|bound|formal analysis|analysis of)\b",
    ],
    "result": [
        r"\b(improve|outperform|state-of-the-art|sota|gain|better performance|empirical results?)\b",
        r"\bachieves?\b",
    ],
}

CONTRIB_PATTERNS = [
    r"\bwe propose\b",
    r"\bwe present\b",
    r"\bwe introduce\b",
    r"\bwe develop\b",
    r"\bwe study\b",
    r"\bthis paper\b",
    r"\bour contributions?\b",
    r"\bnovel\b",
    r"\bnew\b",
    r"\bfirst\b",
    r"\bstate-of-the-art\b",
    r"\boutperform\b",
    r"\bimprove\b",
    r"\bbenchmark\b",
    r"\bdataset\b",
    r"\btask\b",
    r"\bframework\b",
    r"\bmethod\b",
    r"\bmodel\b",
]


# =========================================================
# 数据结构
# =========================================================

class BackgroundCardSchema(BaseModel):
    summary: str = Field(
        description="One-sentence high-level summary of the paper's main contribution."
    )
    key_contribution_sentences: List[str] = Field(
        min_length=1,
        max_length=4,
        description="1-4 concise contribution statements focusing on the paper's own contributions."
    )
    novelty_type_guess: List[Literal["method", "dataset", "task", "application", "theory", "result"]] = Field(
        min_length=1,
        max_length=3,
        description="Choose 1-3 high-level novelty types only from the allowed set."
    )


# =========================================================
# 基础工具函数
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
    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def get_first_existing(d: Dict[str, Any], keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] not in [None, "", []]:
            return d[k]
    return default


def split_sentences(text: str) -> List[str]:
    """
    轻量级英文句子切分，不依赖 nltk/spacy。
    """
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


def jaccard_sent(a: str, b: str) -> float:
    return token_overlap_score(a, b)


def cosine_similarity(a, b) -> float:
    if np is None:
        return 0.0
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def batched(xs: List[Any], batch_size: int):
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


# =========================================================
# section 抽取
# =========================================================

def find_heading_lines(lines: List[str]) -> List[Tuple[int, str]]:
    hits = []
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        for name, pat in SECTION_PATTERNS.items():
            if pat.match(s):
                hits.append((i, name))
                break
    return hits


def extract_section_by_name(full_text: str, section_name: str, fallback_chars: int = 3000) -> str:
    text = normalize_text(full_text)
    if not text:
        return ""

    lines = text.splitlines()
    headings = find_heading_lines(lines)

    start_idx = None
    for i, name in headings:
        if name == section_name:
            start_idx = i
            break

    if start_idx is None:
        if section_name == "introduction":
            return text[:fallback_chars]
        return ""

    end_idx = len(lines)
    for i, name in headings:
        if i > start_idx:
            end_idx = i
            break

    sec_text = "\n".join(lines[start_idx + 1:end_idx]).strip()
    if not sec_text:
        sec_text = "\n".join(lines[start_idx:end_idx]).strip()
    return normalize_text(sec_text)


# =========================================================
# novelty type 粗预测
# =========================================================

def infer_novelty_types(sentences: List[str]) -> List[str]:
    scores = {k: 0 for k in TYPE_PATTERNS}
    joined = " ".join(sentences).lower()

    for t, patterns in TYPE_PATTERNS.items():
        for p in patterns:
            if re.search(p, joined, re.I):
                scores[t] += 1

    ranked = [k for k, v in sorted(scores.items(), key=lambda x: x[1], reverse=True) if v > 0]
    return ranked[:3] if ranked else ["method"]


# =========================================================
# 候选贡献句抽取
# =========================================================

def sentence_quality_score(sent: str) -> float:
    s = sent.strip()
    if not s:
        return -1.0

    wc = len(s.split())
    if wc < 6 or wc > 60:
        return -1.0

    score = 0.0

    if 10 <= wc <= 35:
        score += 1.0
    elif 36 <= wc <= 45:
        score += 0.5

    for p in CONTRIB_PATTERNS:
        if re.search(p, s, re.I):
            score += 0.6

    if re.search(r"\bet al\.\b", s) and re.search(r"\b\d{4}\b", s):
        score -= 0.5
    if re.search(r"\bappendix\b|\breferences\b", s, re.I):
        score -= 1.0

    return score


def select_top_sentences(
    abstract_text: str,
    intro_text: str,
    query_text: str,
    retrieval_score: float = 0.0,
    embed_model=None,
    top_n: int = 6
) -> List[str]:
    candidates = []
    for sent in split_sentences(abstract_text):
        candidates.append(("abstract", sent))
    for sent in split_sentences(intro_text):
        candidates.append(("introduction", sent))

    if not candidates:
        return []

    query_vec = None
    sent_vecs = None
    if embed_model is not None and SentenceTransformer is not None and np is not None:
        texts = [s for _, s in candidates]
        sent_vecs = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        query_vec = embed_model.encode([query_text], convert_to_numpy=True, show_progress_bar=False)[0]

    scored = []
    for idx, (section, sent) in enumerate(candidates):
        base = sentence_quality_score(sent)
        if base < 0:
            continue

        score = base

        if section == "abstract":
            score += 1.0
        else:
            score += 0.4

        overlap = token_overlap_score(sent, query_text)
        score += 1.5 * overlap

        if query_vec is not None and sent_vecs is not None:
            sim = cosine_similarity(sent_vecs[idx], query_vec)
            score += 2.0 * sim

        score += 0.2 * float(retrieval_score)
        scored.append((score, sent))

    scored.sort(key=lambda x: x[0], reverse=True)

    selected = []
    for score, sent in scored:
        duplicated = False
        for ss in selected:
            if jaccard_sent(sent, ss) >= 0.75:
                duplicated = True
                break
        if not duplicated:
            selected.append(sent)
        if len(selected) >= top_n:
            break

    return selected


# =========================================================
# target paper query 构造
# =========================================================

def flatten_string_list(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, str):
        return [x.strip()] if x.strip() else []
    if isinstance(x, list):
        out = []
        for item in x:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
            elif isinstance(item, dict):
                txt = get_first_existing(item, ["text", "sentence", "claim", "content"], "")
                if txt:
                    out.append(str(txt).strip())
        return out
    return []


def get_target_novelty_sentences(target_data: Dict[str, Any]) -> List[str]:
    cand_keys = [
        "intro_novelty_sentences",
        "introduction_novelty_sentences",
        "novelty_sentences",
        "target_novelty_sentences",
        "intro_sentences",
    ]
    for k in cand_keys:
        if k in target_data:
            sents = flatten_string_list(target_data[k])
            if sents:
                return sents

    if "target_paper" in target_data and isinstance(target_data["target_paper"], dict):
        for k in cand_keys:
            if k in target_data["target_paper"]:
                sents = flatten_string_list(target_data["target_paper"][k])
                if sents:
                    return sents

    return []


def build_target_query_text(target_data: Dict[str, Any]) -> str:
    parts = []

    title = get_first_existing(target_data, ["title", "target_title"], "")
    abstract = get_first_existing(target_data, ["abstract", "target_abstract"], "")
    novelty_sents = get_target_novelty_sentences(target_data)

    if not title and "target_paper" in target_data and isinstance(target_data["target_paper"], dict):
        title = get_first_existing(target_data["target_paper"], ["title", "target_title"], "")
    if not abstract and "target_paper" in target_data and isinstance(target_data["target_paper"], dict):
        abstract = get_first_existing(target_data["target_paper"], ["abstract", "target_abstract"], "")

    if title:
        parts.append(str(title))
    if abstract:
        parts.append(str(abstract))
    if novelty_sents:
        parts.append(" ".join(novelty_sents[:8]))

    return normalize_text(" ".join(parts))


# =========================================================
# vLLM 输出解析
# =========================================================

def clean_json_text(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    return text.strip()


def safe_parse_card_json(text: str) -> Optional[Dict[str, Any]]:
    try:
        cleaned = clean_json_text(text)
        obj = json.loads(cleaned)
        return obj
    except Exception:
        return None


def normalize_card_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    summary = str(obj.get("summary", "")).strip()

    key_sents = obj.get("key_contribution_sentences", [])
    if not isinstance(key_sents, list):
        key_sents = []
    key_sents = [str(x).strip() for x in key_sents if str(x).strip()]
    key_sents = key_sents[:4]

    types = obj.get("novelty_type_guess", [])
    if not isinstance(types, list):
        types = []
    norm_types = []
    for t in types:
        t = str(t).strip().lower()
        if t in CARD_TYPES and t not in norm_types:
            norm_types.append(t)
    norm_types = norm_types[:3]

    return {
        "summary": summary,
        "key_contribution_sentences": key_sents,
        "novelty_type_guess": norm_types,
    }


# =========================================================
# vLLM Builder
# =========================================================

class VLLMCardBuilder:
    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.90,
        max_model_len: int = 8192,
        max_tokens: int = 256,
        trust_remote_code: bool = True,
        dtype: str = "auto",
    ):
        self.model_name = model_name
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
        )

        schema = BackgroundCardSchema.model_json_schema()
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            structured_outputs=StructuredOutputsParams(json=schema),
        )

    def build_prompt(
        self,
        source_title: str,
        source_abstract: str,
        intro_excerpt: str,
        candidate_contribution_sentences: List[str],
    ) -> str:
        cand_text = "\n".join(
            [f"{i+1}. {s}" for i, s in enumerate(candidate_contribution_sentences)]
        ).strip()

        user_content = f"""
Title:
{source_title}

Abstract:
{source_abstract}

Introduction Excerpt:
{intro_excerpt}

Candidate Contribution Sentences:
{cand_text}

Return a JSON object with fields:
- summary
- key_contribution_sentences
- novelty_type_guess

Rules:
1. Focus on the paper's own contribution, not general background.
2. Keep only high-level contributions.
3. novelty_type_guess must be chosen only from:
["method", "dataset", "task", "application", "theory", "result"]
4. key_contribution_sentences should be concise and not redundant.
5. Do not copy long passages verbatim unless necessary.
6. Output JSON only.
""".strip()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert NLP researcher. "
                    "Your task is to build a structured background card for a paper."
                ),
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # 某些 tokenizer 可能没有 chat_template，做个兼容
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = (
                "System: You are an expert NLP researcher. "
                "Your task is to build a structured background card for a paper.\n\n"
                f"User:\n{user_content}\n\nAssistant:"
            )
        return prompt

    def generate_batch(self, inputs: List[Dict[str, Any]]) -> List[Optional[Dict[str, Any]]]:
        prompts = []
        for x in inputs:
            prompts.append(
                self.build_prompt(
                    source_title=x["source_title"],
                    source_abstract=x["source_abstract"],
                    intro_excerpt=x["intro_excerpt"],
                    candidate_contribution_sentences=x["candidate_contribution_sentences"],
                )
            )

        outputs = self.llm.generate(prompts, self.sampling_params)

        results = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            obj = safe_parse_card_json(text)
            if obj is None:
                results.append(None)
            else:
                results.append(normalize_card_output(obj))
        return results


# =========================================================
# 单篇背景论文预处理
# =========================================================

def prepare_single_card_input(
    bg_paper: Dict[str, Any],
    target_query_text: str,
    embed_model=None
) -> Dict[str, Any]:
    source_title = safe_str(bg_paper.get("source_title", ""))
    source_year = bg_paper.get("source_year", None)
    source_venue = safe_str(bg_paper.get("source_venue", ""))
    retrieval_score = float(bg_paper.get("retrieval_score", 0.0) or 0.0)
    source_url = safe_str(bg_paper.get("source_url", ""))
    source_abstract = safe_str(bg_paper.get("source_abstract", ""))
    pdf_file = safe_str(bg_paper.get("pdf_file", ""))
    paper_text = safe_str(bg_paper.get("paper_text", ""))

    intro_text = extract_section_by_name(paper_text, "introduction", fallback_chars=3500)

    candidate_sents = select_top_sentences(
        abstract_text=source_abstract,
        intro_text=intro_text,
        query_text=target_query_text,
        retrieval_score=retrieval_score,
        embed_model=embed_model,
        top_n=6,
    )

    intro_excerpt_sents = split_sentences(intro_text)[:5]
    intro_excerpt = " ".join(intro_excerpt_sents).strip()

    if not candidate_sents:
        candidate_sents = split_sentences(source_abstract)[:3]
    if not candidate_sents:
        candidate_sents = split_sentences(intro_excerpt)[:3]

    return {
        "source_title": source_title,
        "source_year": source_year,
        "source_venue": source_venue,
        "retrieval_score": retrieval_score,
        "source_url": source_url,
        "pdf_file": pdf_file,
        "source_abstract": source_abstract,
        "intro_excerpt": intro_excerpt,
        "candidate_contribution_sentences": candidate_sents,
    }


def finalize_card(
    raw_input: Dict[str, Any],
    llm_result: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    fallback_key_sents = raw_input["candidate_contribution_sentences"][:3]
    fallback_types = infer_novelty_types(fallback_key_sents)

    if llm_result is None:
        summary = " ".join(fallback_key_sents[:2]).strip()
        key_sents = fallback_key_sents
        novelty_types = fallback_types
    else:
        summary = llm_result.get("summary", "").strip()
        key_sents = llm_result.get("key_contribution_sentences", [])
        novelty_types = llm_result.get("novelty_type_guess", [])

        if not summary:
            summary = " ".join(fallback_key_sents[:2]).strip()
        if not key_sents:
            key_sents = fallback_key_sents
        if not novelty_types:
            novelty_types = fallback_types

    card_text_parts = []
    if raw_input["source_title"]:
        card_text_parts.append(f"Title: {raw_input['source_title']}")
    if raw_input["source_year"]:
        card_text_parts.append(f"Year: {raw_input['source_year']}")
    if raw_input["source_venue"]:
        card_text_parts.append(f"Venue: {raw_input['source_venue']}")
    if novelty_types:
        card_text_parts.append(f"Novelty Types: {', '.join(novelty_types)}")
    if summary:
        card_text_parts.append(f"Summary: {summary}")
    if key_sents:
        card_text_parts.append("Key Contributions: " + " ".join(key_sents))

    card_text = " | ".join(card_text_parts).strip()

    return {
        "source_title": raw_input["source_title"],
        "source_year": raw_input["source_year"],
        "source_venue": raw_input["source_venue"],
        "retrieval_score": raw_input["retrieval_score"],
        "source_url": raw_input["source_url"],
        "pdf_file": raw_input["pdf_file"],
        "source_abstract": raw_input["source_abstract"],
        "intro_excerpt": raw_input["intro_excerpt"],
        "candidate_contribution_sentences": raw_input["candidate_contribution_sentences"],
        "summary": summary,
        "key_contribution_sentences": key_sents,
        "novelty_type_guess": novelty_types,
        "card_text": card_text,
    }


# =========================================================
# 主流程
# =========================================================

def process_one_paper_dir(
    input_paper_dir: Path,
    output_paper_dir: Path,
    embed_model=None,
    card_builder: Optional[VLLMCardBuilder] = None,
    overwrite: bool = False,
    llm_batch_size: int = 8,
) -> None:
    target_path = input_paper_dir / "target_paper.json"
    bg_path = input_paper_dir / "background_papers.json"

    if not target_path.exists() or not bg_path.exists():
        print(f"[跳过] 缺少 target_paper.json 或 background_papers.json: {input_paper_dir}")
        return

    out_flag = output_paper_dir / "_build_complete.flag"
    out_cards = output_paper_dir / "background_cards.json"

    if out_flag.exists() and out_cards.exists() and not overwrite:
        print(f"[跳过] 已完成: {input_paper_dir.name}")
        return

    try:
        target_data = read_json(target_path)
    except Exception as e:
        print(f"[跳过] 读取 target_paper.json 失败: {input_paper_dir.name} | {e}")
        return

    try:
        bg_data = read_json(bg_path)
    except Exception as e:
        print(f"[跳过] 读取 background_papers.json 失败: {input_paper_dir.name} | {e}")
        return

    if not isinstance(bg_data, list):
        print(f"[跳过] background_papers.json 不是 list: {input_paper_dir.name}")
        return

    target_query_text = build_target_query_text(target_data)
    if not target_query_text:
        target_query_text = json.dumps(target_data, ensure_ascii=False)[:2000]

    prepared_inputs = []
    for idx, bg_paper in enumerate(bg_data):
        try:
            prepared = prepare_single_card_input(
                bg_paper=bg_paper,
                target_query_text=target_query_text,
                embed_model=embed_model,
            )
            prepared_inputs.append(prepared)
        except Exception as e:
            print(f"[警告] {input_paper_dir.name} 第 {idx} 篇背景论文预处理失败: {e}")

    llm_results: List[Optional[Dict[str, Any]]] = []
    if card_builder is not None and prepared_inputs:
        for batch in batched(prepared_inputs, llm_batch_size):
            try:
                batch_out = card_builder.generate_batch(batch)
                llm_results.extend(batch_out)
            except Exception as e:
                print(f"[警告] {input_paper_dir.name} 某个 vLLM batch 失败，回退规则结果: {e}")
                llm_results.extend([None] * len(batch))
    else:
        llm_results = [None] * len(prepared_inputs)

    cards = []
    for raw_input, llm_result in zip(prepared_inputs, llm_results):
        cards.append(finalize_card(raw_input, llm_result))

    output_paper_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_paper_dir / "target_paper.json", target_data)
    write_json(output_paper_dir / "background_cards.json", cards)

    with open(out_flag, "w", encoding="utf-8") as f:
        f.write("ok\n")

    print(f"[完成] {input_paper_dir.name}: {len(cards)} 张背景卡片")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, required=True,
                        help="background_raw_store 根目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="background_card_store 根目录")

    parser.add_argument("--card_model", type=str, required=True,
                        help="用于生成背景卡片的 vLLM 模型名称或本地路径")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=8192)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--llm_batch_size", type=int, default=8)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dtype", type=str, default="auto")

    parser.add_argument("--embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="用于候选贡献句排序的句向量模型；如果未安装 sentence-transformers，则退化为 token overlap")
    parser.add_argument("--overwrite", action="store_true",
                        help="是否覆盖已生成结果")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    embed_model = None
    if SentenceTransformer is not None:
        try:
            embed_model = SentenceTransformer(args.embed_model)
            print(f"[信息] 已加载 embedding 模型: {args.embed_model}")
        except Exception as e:
            print(f"[警告] embedding 模型加载失败，退化为 token overlap: {e}")
            embed_model = None
    else:
        print("[警告] 未安装 sentence-transformers，退化为 token overlap")

    print(f"[信息] 加载 vLLM 模型: {args.card_model}")
    card_builder = VLLMCardBuilder(
        model_name=args.card_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
    )

    paper_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    paper_dirs = sorted(paper_dirs, key=lambda x: x.name)

    print(f"[信息] 待处理论文数: {len(paper_dirs)}")

    for pdir in paper_dirs:
        process_one_paper_dir(
            input_paper_dir=pdir,
            output_paper_dir=output_dir / pdir.name,
            embed_model=embed_model,
            card_builder=card_builder,
            overwrite=args.overwrite,
            llm_batch_size=args.llm_batch_size,
        )

    print("[全部完成]")


if __name__ == "__main__":
    main()