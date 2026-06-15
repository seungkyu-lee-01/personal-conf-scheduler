"""Pipeline entry point: expand interest → hybrid retrieve → batch-rate → write outputs."""

import logging
import re
import sys
from pathlib import Path

from .config import (
    RESEARCH_INTEREST,
    PAPERS_CSV_PATH,
    OUTPUT_CSV_PATH,
    OUTPUT_MARKDOWN_PATH,
    CANDIDATE_POOL_SIZE,
    RELEVANCE_SCORE_THRESHOLD,
)
from .interest_expander import expand_interest
from .retriever import load_inperson_papers, hybrid_rank
from .rater import rate_all


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("src_openai.main")


_OUTPUT_COLUMNS = [
    "Paper number",
    "Title",
    "Authors Names",
    "llm_score",
    "llm_reason",
    "retrieval_score",
    "embedding_score",
    "keyword_score",
    "Session",
    "Underline/Whova Session Name",
    "Session Date",
    "Session time",
    "Room Location",
    "Presentation mode",
    "Abstract",
]


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _session_sort_key(date_str: str, time_str: str) -> tuple:
    """Parse "Sun. July 5" + "09:00-10:30" into a sortable (month, day, start_minutes) tuple."""
    date_str = (date_str or "").strip()
    time_str = (time_str or "").strip()

    month = day = 10**6
    m = re.search(r"([A-Za-z]+)\.?\s+(\d{1,2})", date_str)
    if m:
        mon_key = m.group(1)[:3].lower()
        if mon_key in _MONTHS:
            month = _MONTHS[mon_key]
            day = int(m.group(2))

    minutes = 10**6
    t = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if t:
        minutes = int(t.group(1)) * 60 + int(t.group(2))

    return (month, day, minutes)


def _to_markdown(df) -> str:
    if df.empty:
        return "# No papers above threshold.\n"
    lines = ["# ACL 2026 — Relevant In-Person Papers", ""]
    sessions = []
    for sess, sub in df.groupby("Underline/Whova Session Name", dropna=False, sort=False):
        date = sub["Session Date"].iloc[0] if "Session Date" in sub.columns else ""
        time = sub["Session time"].iloc[0] if "Session time" in sub.columns else ""
        sessions.append((_session_sort_key(date, time), sess, sub))
    sessions.sort(key=lambda x: x[0])

    for _, sess, sub in sessions:
        lines.append(f"## {sess if sess else '(no session)'}")
        for _, r in sub.iterrows():
            date = r.get("Session Date", "")
            time = r.get("Session time", "")
            room = r.get("Room Location", "")
            lines.append(f"- **[{r['llm_score']}/5] {r['Title']}**  ")
            lines.append(f"  _{date} {time} · {room}_  ")
            lines.append(f"  {r['llm_reason']}")
            lines.append("")
    return "\n".join(lines)


def main() -> int:
    logger.info("=" * 70)
    logger.info("Stage 0 — expanding research interest via LLM")
    expanded = expand_interest(RESEARCH_INTEREST)
    logger.info(f"Refined query: {expanded['refined_query']}")
    logger.info(f"Keywords ({len(expanded['keywords'])}): {expanded['keywords']}")
    logger.info(f"Negative keywords ({len(expanded['negative_keywords'])}): {expanded['negative_keywords']}")

    logger.info("=" * 70)
    logger.info("Stage 1 — hybrid retrieval")
    df = load_inperson_papers(PAPERS_CSV_PATH)
    candidates = hybrid_rank(
        df,
        refined_query=expanded["refined_query"],
        keywords=expanded["keywords"],
        negative_keywords=expanded["negative_keywords"],
        top_k=CANDIDATE_POOL_SIZE,
    )

    logger.info("=" * 70)
    logger.info(f"Stage 2 — LLM batch rating ({len(candidates)} papers)")
    rated = rate_all(candidates, research_interest=RESEARCH_INTEREST)

    relevant = rated[rated["llm_score"] >= RELEVANCE_SCORE_THRESHOLD].sort_values(
        ["llm_score", "retrieval_score"], ascending=[False, False]
    )
    logger.info(
        f"Filter score >= {RELEVANCE_SCORE_THRESHOLD}: {len(relevant)}/{len(rated)} papers kept"
    )

    cols = [c for c in _OUTPUT_COLUMNS if c in relevant.columns]
    out_csv = Path(OUTPUT_CSV_PATH)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    relevant[cols].to_csv(out_csv, index=False)
    logger.info(f"Wrote {out_csv}")

    out_md = Path(OUTPUT_MARKDOWN_PATH)
    out_md.write_text(_to_markdown(relevant[cols]))
    logger.info(f"Wrote {out_md}")

    # Also write the full rated pool (useful for tuning the threshold)
    full_path = out_csv.with_name(out_csv.stem + "_all_rated.csv")
    rated.sort_values("llm_score", ascending=False)[cols].to_csv(full_path, index=False)
    logger.info(f"Wrote {full_path} (all {len(rated)} rated candidates)")

    print(f"\n=== Summary ===")
    print(f"In-person papers loaded:     {len(df)}")
    print(f"Stage 1 candidate pool:      {len(candidates)}")
    print(f"Stage 2 papers rated:        {len(rated)}")
    print(f"Relevant (score >= {RELEVANCE_SCORE_THRESHOLD}):       {len(relevant)}")
    print(f"\nOutputs:")
    print(f"  {out_csv}")
    print(f"  {out_md}")
    print(f"  {full_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
