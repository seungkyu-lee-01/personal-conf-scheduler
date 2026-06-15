"""Stage 1 — hybrid retrieval (sentence-BERT cosine + keyword count)."""

import hashlib
import logging
import re
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from .config import (
    EMBEDDING_MODEL,
    EMBEDDING_CACHE_DIR,
    HYBRID_EMBEDDING_WEIGHT,
    HYBRID_KEYWORD_WEIGHT,
    PRESENTATION_MODE_FILTER,
)

logger = logging.getLogger(__name__)

_model_cache = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[Retriever] Loading sentence-transformer: {model_name}")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def load_inperson_papers(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df[df["Presentation mode"] == PRESENTATION_MODE_FILTER].copy()
    df = df.dropna(subset=["Title", "Abstract"])
    df = df[df["Title"].str.strip().astype(bool) & df["Abstract"].str.strip().astype(bool)]
    df = df.reset_index(drop=True)
    logger.info(f"[Retriever] Loaded {len(df)}/{before} in-person papers with non-empty title+abstract")
    return df


def _corpus_text(df: pd.DataFrame) -> List[str]:
    return [f"{t.strip()}. {a.strip()}" for t, a in zip(df["Title"], df["Abstract"])]


def _cache_key(texts: List[str], model_name: str) -> str:
    h = hashlib.sha1()
    h.update(model_name.encode())
    h.update(str(len(texts)).encode())
    # Hash a sample of titles to detect content changes
    for t in texts[:: max(1, len(texts) // 20)]:
        h.update(t[:200].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


def embed_corpus(df: pd.DataFrame, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    texts = _corpus_text(df)
    Path(EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    key = _cache_key(texts, model_name)
    cache_path = Path(EMBEDDING_CACHE_DIR) / f"emb_{key}.npy"
    if cache_path.exists():
        logger.info(f"[Retriever] Loading cached embeddings: {cache_path}")
        return np.load(cache_path)
    logger.info(f"[Retriever] Encoding {len(texts)} papers (this takes ~30-60s)...")
    model = _get_model(model_name)
    emb = model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(cache_path, emb)
    logger.info(f"[Retriever] Cached embeddings to {cache_path}")
    return emb


def _normalize(scores: np.ndarray) -> np.ndarray:
    if scores.max() == scores.min():
        return np.zeros_like(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())


def score_embeddings(query: str, corpus_emb: np.ndarray, model_name: str = EMBEDDING_MODEL) -> np.ndarray:
    model = _get_model(model_name)
    q = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
    sims = corpus_emb @ q  # already L2-normalized → cosine
    return _normalize(sims)


def score_keywords(
    df: pd.DataFrame,
    keywords: List[str],
    negative_keywords: Optional[List[str]] = None,
) -> np.ndarray:
    if not keywords:
        return np.zeros(len(df))
    haystacks = [
        f"{(t or '').lower()} {(a or '').lower()}"
        for t, a in zip(df["Title"].fillna(""), df["Abstract"].fillna(""))
    ]
    pos_patterns = [re.compile(rf"\b{re.escape(k)}\b") for k in keywords]
    neg_patterns = [re.compile(rf"\b{re.escape(k)}\b") for k in (negative_keywords or [])]

    raw = np.zeros(len(haystacks), dtype=float)
    for i, text in enumerate(haystacks):
        pos_hits = sum(1 for p in pos_patterns if p.search(text))
        neg_hits = sum(1 for p in neg_patterns if p.search(text))
        raw[i] = max(0.0, pos_hits - 0.5 * neg_hits)
    return _normalize(raw)


def hybrid_rank(
    df: pd.DataFrame,
    refined_query: str,
    keywords: List[str],
    top_k: int,
    negative_keywords: Optional[List[str]] = None,
    embedding_weight: float = HYBRID_EMBEDDING_WEIGHT,
    keyword_weight: float = HYBRID_KEYWORD_WEIGHT,
) -> pd.DataFrame:
    corpus_emb = embed_corpus(df)
    emb_scores = score_embeddings(refined_query, corpus_emb)
    kw_scores = score_keywords(df, keywords, negative_keywords)
    combined = embedding_weight * emb_scores + keyword_weight * kw_scores
    out = df.copy()
    out["embedding_score"] = emb_scores
    out["keyword_score"] = kw_scores
    out["retrieval_score"] = combined
    out = out.sort_values("retrieval_score", ascending=False).head(top_k).reset_index(drop=True)
    logger.info(
        f"[Retriever] Top-{top_k} retrieved: emb_w={embedding_weight} kw_w={keyword_weight} "
        f"score range [{out['retrieval_score'].min():.3f}, {out['retrieval_score'].max():.3f}]"
    )
    return out
