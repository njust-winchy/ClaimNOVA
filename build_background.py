import re
import json
import shutil
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any

import requests
import fitz  # PyMuPDF
from tqdm import tqdm


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# =========================
# Config
# =========================
CANDIDATE_DIR = Path("/gpfs/scratch/qc25257/retrieved_candidates")
OUTPUT_ROOT = Path("/gpfs/scratch/qc25257/background_raw_store")
TEMP_ROOT = Path("/gpfs/scratch/qc25257/tmp_pdfs_background_raw")


# =========================
# Text cleaning
# =========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = ''.join(
        ch for ch in text
        if unicodedata.category(ch)[0] != 'C' or ch in '\n\t '
    )
    text = unicodedata.normalize('NFKC', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def safe_filename(name: str, max_len: int = 180) -> str:
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name).strip()
    if len(name) > max_len:
        name = name[:max_len]
    return name


# =========================
# Download
# =========================
def looks_like_pdf_response(response: requests.Response) -> bool:
    content_type = response.headers.get("Content-Type", "").lower()
    if "pdf" in content_type:
        return True
    return response.content[:5] == b"%PDF-"


def normalize_possible_pdf_url(url: str) -> str:
    if not url:
        return url

    url = url.strip()

    # arXiv abs -> pdf
    m = re.match(r"https?://arxiv\.org/abs/([0-9]+\.[0-9]+)(v\d+)?", url)
    if m:
        paper_id = m.group(1)
        version = m.group(2) or ""
        return f"https://arxiv.org/pdf/{paper_id}{version}.pdf"

    # ACL Anthology page -> pdf
    m = re.match(r"https?://aclanthology\.org/([^/]+)/?$", url)
    if m and not url.endswith(".pdf"):
        anthology_id = m.group(1)
        return f"https://aclanthology.org/{anthology_id}.pdf"

    return url


def download_pdf(url: str, save_path: Path, timeout: int = 30) -> bool:
    url = normalize_possible_pdf_url(url)

    try:
        response = requests.get(
            url,
            timeout=timeout,
            headers={"User-Agent": "Mozilla/5.0"}
        )
    except Exception as e:
        logger.warning(f"Request failed: {url} | {e}")
        return False

    if response.status_code != 200:
        logger.warning(f"Download failed ({response.status_code}): {url}")
        return False

    if not looks_like_pdf_response(response):
        logger.warning(f"Not a PDF response: {url}")
        return False

    try:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.warning(f"Failed to save PDF {save_path}: {e}")
        return False


# =========================
# Dual-column PDF parsing
# =========================
def is_likely_header_or_footer(text: str, y0: float, y1: float, page_height: float) -> bool:
    low = text.lower().strip()
    if not low:
        return True

    near_top = y1 < page_height * 0.08
    near_bottom = y0 > page_height * 0.92

    footer_signals = [
        "association for computational linguistics",
        "proceedings of",
        "pages ",
        "doi",
        "copyright",
        "licensed under",
        "creative commons",
        "arxiv:",
        "aclanthology.org",
    ]

    if (near_top or near_bottom) and any(sig in low for sig in footer_signals):
        return True

    if (near_top or near_bottom) and re.fullmatch(r"[\d\-–]+", low):
        return True

    return False


def merge_nearby_blocks(blocks: List[Dict], y_gap_threshold: float = 6.0) -> List[Dict]:
    if not blocks:
        return []

    merged = []
    current = blocks[0].copy()

    for blk in blocks[1:]:
        same_column_close = abs(blk["x0"] - current["x0"]) < 30
        vertical_close = blk["y0"] - current["y1"] <= y_gap_threshold

        if same_column_close and vertical_close:
            current["text"] = current["text"].rstrip() + "\n" + blk["text"].lstrip()
            current["x0"] = min(current["x0"], blk["x0"])
            current["y0"] = min(current["y0"], blk["y0"])
            current["x1"] = max(current["x1"], blk["x1"])
            current["y1"] = max(current["y1"], blk["y1"])
        else:
            merged.append(current)
            current = blk.copy()

    merged.append(current)
    return merged


def sort_blocks_dual_column(page, blocks_raw) -> List[str]:
    page_width = page.rect.width
    page_height = page.rect.height
    mid_x = page_width / 2.0

    parsed_blocks = []

    for blk in blocks_raw:
        if len(blk) < 5:
            continue

        x0, y0, x1, y1, text = blk[:5]
        if not isinstance(text, str):
            continue

        text = clean_text(text)
        if not text or len(text) < 2:
            continue

        if is_likely_header_or_footer(text, y0, y1, page_height):
            continue

        parsed_blocks.append({
            "x0": float(x0),
            "y0": float(y0),
            "x1": float(x1),
            "y1": float(y1),
            "text": text
        })

    if not parsed_blocks:
        return []

    left_blocks = []
    right_blocks = []
    spanning_blocks = []

    gutter_margin = page_width * 0.04

    for blk in parsed_blocks:
        center_x = (blk["x0"] + blk["x1"]) / 2.0
        crosses_mid = blk["x0"] < mid_x - gutter_margin and blk["x1"] > mid_x + gutter_margin
        wide_block = (blk["x1"] - blk["x0"]) > page_width * 0.6

        if crosses_mid or wide_block:
            spanning_blocks.append(blk)
        elif center_x <= mid_x:
            left_blocks.append(blk)
        else:
            right_blocks.append(blk)

    spanning_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    left_blocks.sort(key=lambda b: (b["y0"], b["x0"]))
    right_blocks.sort(key=lambda b: (b["y0"], b["x0"]))

    spanning_blocks = merge_nearby_blocks(spanning_blocks)
    left_blocks = merge_nearby_blocks(left_blocks)
    right_blocks = merge_nearby_blocks(right_blocks)

    ordered_texts = []

    for blk in spanning_blocks:
        if blk["y0"] < page_height * 0.28:
            ordered_texts.append(blk["text"])

    for blk in left_blocks:
        ordered_texts.append(blk["text"])
    for blk in right_blocks:
        ordered_texts.append(blk["text"])

    for blk in spanning_blocks:
        if blk["y0"] >= page_height * 0.28:
            ordered_texts.append(blk["text"])

    return ordered_texts


def extract_text_from_pdf_dualcolumn(pdf_path: Path) -> str:
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        logger.warning(f"Failed to open PDF {pdf_path}: {e}")
        return ""

    all_pages_text = []

    try:
        for page_idx in range(len(doc)):
            try:
                page = doc[page_idx]
                blocks_raw = page.get_text("blocks")
                ordered_texts = sort_blocks_dual_column(page, blocks_raw)

                if not ordered_texts:
                    continue

                page_text = "\n\n".join(ordered_texts)
                page_text = clean_text(page_text)

                if len(page_text) < 50:
                    continue

                all_pages_text.append(page_text)

            except Exception as e:
                logger.warning(f"Failed on page {page_idx + 1} of {pdf_path.name}: {e}")
                continue
    finally:
        doc.close()

    merged_text = "\n\n".join(all_pages_text)
    merged_text = clean_text(merged_text)

    if not merged_text:
        return ""

    ref_pattern = re.compile(r'\n\s*(references|bibliography)\s*\n', flags=re.IGNORECASE)
    matches = list(ref_pattern.finditer(merged_text))
    if matches:
        text_len = len(merged_text)
        for m in matches:
            if m.start() > text_len * 0.55:
                merged_text = merged_text[:m.start()].strip()
                break

    appendix_pattern = re.compile(r'\n\s*(appendix|appendices)\s*\n', flags=re.IGNORECASE)
    matches = list(appendix_pattern.finditer(merged_text))
    if matches:
        text_len = len(merged_text)
        for m in matches:
            if m.start() > text_len * 0.7:
                merged_text = merged_text[:m.start()].strip()
                break

    return merged_text


# =========================
# Resume / completion
# =========================
def is_completed(output_dir: Path) -> bool:
    return (
        output_dir.exists()
        and (output_dir / "background_papers.json").exists()
        and (output_dir / "target_paper.json").exists()
        and (output_dir / "_build_complete.flag").exists()
    )


def save_completion_flag(output_dir: Path):
    (output_dir / "_build_complete.flag").write_text("ok", encoding="utf-8")


# =========================
# Main processing
# =========================
def process_single_candidate_file(
    candidate_file: Path,
    output_root: Path,
    temp_root: Path
):
    with open(candidate_file, "r", encoding="utf-8") as f:
        payload = json.load(f)

    target_paper = payload["target_paper"]
    background_candidates = payload["background_candidates"]

    paper_id = str(target_paper["paper_id"])
    title = str(target_paper["title"]).strip()

    output_dir = output_root / paper_id
    output_dir.mkdir(parents=True, exist_ok=True)

    if is_completed(output_dir):
        logger.info(f"Skip completed paper_id={paper_id}")
        return

    temp_dir = temp_root / paper_id
    temp_dir.mkdir(parents=True, exist_ok=True)

    processed_papers = []

    try:
        for paper in background_candidates:
            paper_title = paper["title"]
            paper_url = paper["url"]

            pdf_name = safe_filename(paper_title) + ".pdf"
            pdf_path = temp_dir / pdf_name

            ok = download_pdf(paper_url, pdf_path)
            if not ok:
                continue

            raw_text = extract_text_from_pdf_dualcolumn(pdf_path)
            raw_text = clean_text(raw_text)

            if not raw_text or len(raw_text) < 300:
                logger.warning(f"Empty/invalid extracted text: {pdf_name}")
                continue

            processed_papers.append({
                "source_title": paper_title,
                "source_year": paper.get("year"),
                "source_venue": paper.get("venue"),
                "retrieval_score": paper.get("score"),
                "source_url": paper_url,
                "source_abstract": paper.get("abstract", ""),
                "pdf_file": pdf_name,
                "paper_text": raw_text
            })

        if not processed_papers:
            logger.warning(f"No valid background papers collected for {paper_id}")

        with open(output_dir / "background_papers.json", "w", encoding="utf-8") as f:
            json.dump(processed_papers, f, ensure_ascii=False, indent=2)

        with open(output_dir / "target_paper.json", "w", encoding="utf-8") as f:
            json.dump(target_paper, f, ensure_ascii=False, indent=2)

        save_completion_flag(output_dir)
        logger.info(f"Finished paper_id={paper_id} ({title})")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    output_root = OUTPUT_ROOT
    temp_root = TEMP_ROOT

    output_root.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    candidate_files = sorted(CANDIDATE_DIR.glob("*.json"))
    if not candidate_files:
        raise FileNotFoundError("No candidate JSON files found in retrieved_candidates/")

    # candidate_files = candidate_files[:3]

    for candidate_file in tqdm(candidate_files, total=len(candidate_files)):
        try:
            process_single_candidate_file(
                candidate_file=candidate_file,
                output_root=output_root,
                temp_root=temp_root
            )
        except Exception as e:
            logger.exception(f"Unexpected error in {candidate_file.name}: {e}")


if __name__ == "__main__":
    main()