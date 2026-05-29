import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from sqlalchemy import create_engine
from sentence_transformers import SentenceTransformer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def get_candidate_year_range(target_year: int) -> Tuple[int, int]:
    if target_year == 2023:
        return 2019, 2022
    elif target_year == 2024:
        return 2020, 2023
    else:
        raise ValueError(f"Unsupported target year: {target_year}")


def get_engine():
    user = os.getenv("DB_USER", "root")
    password = os.getenv("DB_PASSWORD", "970515")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "3306")
    database = os.getenv("DB_NAME", "ACL")

    if not password:
        logger.warning("DB_PASSWORD is empty. Please set it via environment variables.")

    return create_engine(
        f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    )


def load_candidate_papers() -> pd.DataFrame:
    sql_query = """
    SELECT DISTINCT *
    FROM paper
    WHERE year BETWEEN 2019 AND 2023
      AND CHAR_LENGTH(abstract) > 10
      AND venue IN ('ACL', 'EMNLP', 'NAACL');
    """
    engine = get_engine()
    df = pd.read_sql(sql_query, engine)
    df = df.drop_duplicates(subset=["title"]).reset_index(drop=True)
    return df


class AbstractSimilaritySearch:
    def __init__(
        self,
        all_abstracts: List[str],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        index_path: str = "faiss_index_abstracts.bin",
        vec_path: str = "abstract_vectors.npy"
    ):
        self.all_abstracts = all_abstracts
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.vec_path = Path(vec_path)
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.abstract_vectors = None

    def build_index(self):
        logger.info("Generating abstract vectors...")
        self.abstract_vectors = self.model.encode(
            self.all_abstracts,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        faiss.normalize_L2(self.abstract_vectors)

        dim = self.abstract_vectors.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.abstract_vectors)

        faiss.write_index(self.index, str(self.index_path))
        np.save(str(self.vec_path), self.abstract_vectors)

        logger.info(f"Saved index to {self.index_path}")
        logger.info(f"Saved vectors to {self.vec_path}")

    def load_index(self):
        if not self.index_path.exists() or not self.vec_path.exists():
            raise FileNotFoundError("Index files not found. Run build_index() first.")
        self.index = faiss.read_index(str(self.index_path))
        self.abstract_vectors = np.load(str(self.vec_path))
        logger.info("Loaded abstract index.")

    def search(self, query_abstract: str, top_k: int = 150):
        if self.index is None:
            self.load_index()

        query_vector = self.model.encode([query_abstract], convert_to_numpy=True)
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_k)
        return indices[0].tolist(), distances[0].tolist()


def retrieve_background_candidates(
    candidate_df: pd.DataFrame,
    searcher: AbstractSimilaritySearch,
    target_title: str,
    target_abstract: str,
    target_year: int,
    final_top_k: int = 30,
    initial_top_k: int = 150
) -> List[Dict]:
    min_year, max_year = get_candidate_year_range(target_year)
    indices, scores = searcher.search(target_abstract, top_k=initial_top_k)

    retrieved = []
    seen_titles = set()

    for idx, score in zip(indices, scores):
        row = candidate_df.iloc[idx]

        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()
        url = str(row.get("url", "")).strip()
        venue = str(row.get("venue", "")).strip()
        year = row.get("year", None)

        if not title or not abstract or not url or pd.isna(year):
            continue

        year = int(year)

        if year < min_year or year > max_year:
            continue
        if title.lower() == target_title.lower():
            continue
        if title.lower() in seen_titles:
            continue

        seen_titles.add(title.lower())

        retrieved.append({
            "title": title,
            "abstract": abstract,
            "url": url,
            "venue": venue,
            "year": year,
            "score": float(score)
        })

        if len(retrieved) >= final_top_k:
            break

    return retrieved


def main():
    target_json = "Dataset_with_type.json"
    output_dir = Path("../retrieved_candidates")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(target_json, "r", encoding="utf-8") as f:
        target_data = json.load(f)

    df_target = pd.DataFrame(target_data)
    required_cols = {"paper_id", "title", "abstract", "year"}
    missing = required_cols - set(df_target.columns)
    if missing:
        raise ValueError(f"Missing required columns in target JSON: {missing}")

    candidate_df = load_candidate_papers()

    candidate_abstracts = candidate_df["abstract"].astype(str).tolist()
    searcher = AbstractSimilaritySearch(candidate_abstracts)

    if not searcher.index_path.exists() or not searcher.vec_path.exists():
        searcher.build_index()

    for _, row in tqdm(df_target.iterrows(), total=len(df_target)):
        paper_id = str(row["paper_id"])
        title = str(row["title"]).strip()
        abstract = str(row["abstract"]).strip()
        target_year = int(row["year"])

        out_file = output_dir / f"{paper_id}.json"
        if out_file.exists():
            logger.info(f"Skip existing candidates for {paper_id}")
            continue

        if not abstract or len(abstract) < 10:
            logger.warning(f"Skip {paper_id}: empty abstract")
            continue

        candidates = retrieve_background_candidates(
            candidate_df=candidate_df,
            searcher=searcher,
            target_title=title,
            target_abstract=abstract,
            target_year=target_year,
            final_top_k=30,
            initial_top_k=150
        )

        payload = {
            "target_paper": {
                "paper_id": paper_id,
                "title": title,
                "abstract": abstract,
                "year": target_year
            },
            "background_candidates": candidates
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Finished exporting retrieved candidates.")


if __name__ == "__main__":
    main()