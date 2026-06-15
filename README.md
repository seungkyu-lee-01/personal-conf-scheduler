# Conference Paper Search

LLM-assisted triage of accepted conference papers (currently ACL 2026) against a free-form
description of your research interest. The pipeline retrieves a candidate pool with hybrid
semantic + keyword search, then asks an LLM to rate each candidate on a 0–5 relevance
scale and emits a ranked CSV + Markdown report grouped by session.

## How it works

The pipeline runs in three stages, orchestrated by `src/main.py`:

1. **Interest expansion** (`src/interest_expander.py`) — A fast LLM (Claude Sonnet) turns
   the free-form `RESEARCH_INTEREST` text from `src/config.py` into a refined semantic
   query, a list of 8–15 positive keywords, and an optional list of negative keywords.
2. **Hybrid retrieval** (`src/retriever.py`) — Loads the accepted papers CSV, filters to
   in-person presentations with non-empty title and abstract, embeds the corpus with
   `sentence-transformers/all-mpnet-base-v2` (cached on disk under `.cache/`), and ranks
   papers by a weighted combination of cosine similarity (0.6) and keyword count (0.4).
   The top `CANDIDATE_POOL_SIZE` (default 100) papers move on.
3. **LLM rating** (`src/rater.py`) — Sends the candidates to a reasoning LLM
   (Claude Opus) in batches of 10 with up to 5 parallel workers, asking for a strict
   0–5 score and a one-sentence reason per paper, returned as structured JSON.

Papers scoring at or above `RELEVANCE_SCORE_THRESHOLD` (default 3) are kept and written
to disk; the full rated pool is also saved so you can tune the threshold without
re-running the pipeline.

## Repository layout

```
conference_paper_search/
├── src/
│   ├── main.py               # pipeline entry point
│   ├── config.py             # research interest, paths, weights, model names
│   ├── interest_expander.py  # stage 0: free-form text → query + keywords
│   ├── retriever.py          # stage 1: hybrid embedding + keyword ranking
│   ├── rater.py              # stage 2: parallel LLM batch rating
│   └── llm_provider.py       # LLM Provider for OpenAI
├── acl2026/
│   ├── accepted_papers.csv          # input: full accepted-papers list
│   ├── relevant_papers.csv          # output: filtered ranked list
│   ├── relevant_papers.md           # output: human-readable report by session
│   └── relevant_papers_all_rated.csv# output: every rated candidate (for tuning)
├── .cache/                   # cached sentence-transformer embeddings
└── .env                      # credentials (OPENAI_API_KEY, not committed)
```

## Setup

1. Install dependencies:
   ```bash
   pip install openai python-dotenv pandas numpy sentence-transformers
   ```
2. Place OPENAI_API_KEY in a `.env` file at the repo root
   ```bash
   echo "OPENAI_API_KEY=sk-..." >> .env
   ```
3. Drop the conference's accepted-papers CSV at the path configured in
   `src/config.py` (default `acl2026/accepted_papers.csv`). Required columns:
   `Paper number`, `Title`, `Authors Names`, `Abstract`, `Presentation mode`,
   plus the session/location fields used in the Markdown output.

## Usage

Edit `RESEARCH_INTEREST` in `src/config.py` to describe what you care about, then run:

```bash
python -m src.main
```

Outputs land alongside the input CSV:

- `relevant_papers.csv` — filtered, ranked by `llm_score` then `retrieval_score`
- `relevant_papers.md` — same papers grouped by session for at-a-glance reading
- `relevant_papers_all_rated.csv` — every candidate the LLM scored


## Configuration knobs

All in `src/config.py`:

| Setting | Purpose |
|---|---|
| `RESEARCH_INTEREST` | Free-form description of your interests |
| `PRESENTATION_MODE_FILTER` | Restrict to e.g. `In-Person` papers |
| `CANDIDATE_POOL_SIZE` | How many papers stage 2 rates (cost/coverage tradeoff) |
| `HYBRID_EMBEDDING_WEIGHT` / `HYBRID_KEYWORD_WEIGHT` | Stage 1 score blend |
| `RATING_BATCH_SIZE`, `RATING_MAX_WORKERS` | Stage 2 throughput |
| `RELEVANCE_SCORE_THRESHOLD` | Minimum LLM score kept in the filtered output |
| `REASONING_MODEL`, `FAST_MODEL` | Which LLMs to use for rating vs. expansion |
