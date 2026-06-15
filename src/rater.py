"""Stage 2 — LLM batch rater (10 papers/batch, parallel calls)."""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import pandas as pd

from .llm_provider import call_llm_json_schema
from .config import (
    OPENAI_REASONING_MODEL,
    LLM_MAX_TOKENS_REASONING,
    RATING_BATCH_SIZE,
    RATING_MAX_WORKERS,
    ABSTRACT_TRUNCATE_CHARS,
)

logger = logging.getLogger(__name__)


_SCHEMA = {
    "type": "object",
    "properties": {
        "ratings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "score": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 5,
                        "description": "Relevance to the user's research interest. 0=unrelated, 5=highly relevant.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One concise sentence justifying the score.",
                    },
                },
                "required": ["paper_id", "score", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["ratings"],
    "additionalProperties": False,
}


def _build_system_prompt(research_interest: str) -> str:
    return f"""\
You are a research-paper triage assistant. The user's research interest is:

\"\"\"
{research_interest.strip()}
\"\"\"

You will receive a batch of papers (id, title, abstract). For each paper, output an
integer relevance score on a 0–5 scale and one concise sentence of justification.

Scoring rubric:
- 5: directly tackles the user's core topic
- 4: closely related; methodologically or topically adjacent
- 3: shares a meaningful sub-area, worth a closer look
- 2: tangentially related
- 1: distantly related
- 0: unrelated / off-topic

Be strict — most papers should score 0–2. Output JSON only, matching the schema.
Return one entry per paper, preserving the given paper_id exactly.
"""


def _format_batch(papers: List[Dict]) -> str:
    parts = []
    for p in papers:
        abstract = (p.get("abstract") or "")[:ABSTRACT_TRUNCATE_CHARS]
        parts.append(f"--- paper_id: {p['paper_id']} ---\nTitle: {p['title']}\nAbstract: {abstract}\n")
    return "\n".join(parts)


def rate_batch(papers: List[Dict], research_interest: str) -> List[Dict]:
    """Rate up to RATING_BATCH_SIZE papers in one LLM call."""
    user_prompt = _format_batch(papers)
    raw = call_llm_json_schema(
        model_name=OPENAI_REASONING_MODEL,
        max_tokens=LLM_MAX_TOKENS_REASONING,
        system_prompt=_build_system_prompt(research_interest),
        user_prompt=user_prompt,
        json_schema=_SCHEMA,
        schema_name="PaperRelevanceRatings",
        temperature=0.0,
    )
    parsed = json.loads(raw)
    return parsed.get("ratings", [])


def _chunk(items: List, size: int) -> List[List]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def rate_all(
    candidates: pd.DataFrame,
    research_interest: str,
    batch_size: int = RATING_BATCH_SIZE,
    max_workers: int = RATING_MAX_WORKERS,
) -> pd.DataFrame:
    """Rate every candidate; merge score/reason into the dataframe."""
    records = [
        {
            "paper_id": str(row["Paper number"]),
            "title": row["Title"],
            "abstract": row["Abstract"],
        }
        for _, row in candidates.iterrows()
    ]
    batches = _chunk(records, batch_size)
    logger.info(f"[Rater] Rating {len(records)} papers in {len(batches)} batches (workers={max_workers})")

    results: Dict[str, Dict] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_idx = {
            ex.submit(rate_batch, batch, research_interest): i for i, batch in enumerate(batches)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                ratings = fut.result()
                for r in ratings:
                    results[str(r["paper_id"])] = {"score": r["score"], "reason": r["reason"]}
                logger.info(f"[Rater] Batch {idx + 1}/{len(batches)} done ({len(ratings)} ratings)")
            except Exception as e:
                logger.error(f"[Rater] Batch {idx + 1} failed: {e}")

    out = candidates.copy()
    out["llm_score"] = out["Paper number"].astype(str).map(lambda pid: results.get(pid, {}).get("score"))
    out["llm_reason"] = out["Paper number"].astype(str).map(lambda pid: results.get(pid, {}).get("reason", ""))
    missing = out["llm_score"].isna().sum()
    if missing:
        logger.warning(f"[Rater] {missing} papers received no rating (likely batch failure)")
        out["llm_score"] = out["llm_score"].fillna(-1).astype(int)
    else:
        out["llm_score"] = out["llm_score"].astype(int)
    return out
