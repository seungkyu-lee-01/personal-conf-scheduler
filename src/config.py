"""Configuration for ACL 2026 paper search pipeline (OpenAI variant)."""

# ==============================================================================
# LLM Configuration (OpenAI)
# ==============================================================================

# Reasoning model: complex analysis (used for batch rating)
OPENAI_REASONING_MODEL = "gpt-4o"

# Fast model: lightweight tasks (interest expansion)
OPENAI_FAST_MODEL = "gpt-4o-mini"

# Token Limits
LLM_MAX_TOKENS_REASONING = 16000
LLM_MAX_TOKENS_FAST = 8000

# Temperature
LLM_TEMPERATURE = 0.1

# ==============================================================================
# Research Interest (free-form text — the LLM will refine this into query + keywords)
# ==============================================================================

RESEARCH_INTEREST = """\
My research interests include how agents can understand user intent clearly, reason with minimal overhead, and act adaptively in ambiguous situations. My goal is to develop AI systems that combine structured reasoning with personalized behavior, enabling accurate and cost-effective collaboration with humans.
I envision an ecosystem of specialized agents that interface with diverse services to accomplish complex goals. As these agents collaborate, the context they share---dialogue history, intermediate reasoning, and environmental states---will inevitably grow richer. To filter out the irrelevant information and reduce unnecessary computation, effective context management will become increasingly important. Furthermore, since users will not specify every detail in each query, agents must be able to remember past interactions, infer unstated preferences, and deliver responses aligned with their priorities.
To advance this vision, my research aims to develop (1) efficient agents that reason and act with minimal overhead, and (2) personalized agents that adapt their behavior to individual users.
"""

# ==============================================================================
# Paper Search Pipeline
# ==============================================================================

# Paths
PAPERS_CSV_PATH = "acl2026/accepted_papers.csv"
OUTPUT_CSV_PATH = "acl2026/relevant_papers.csv"
OUTPUT_MARKDOWN_PATH = "acl2026/relevant_papers.md"
EMBEDDING_CACHE_DIR = ".cache"

# Filter
PRESENTATION_MODE_FILTER = "In-Person"  # only physical presentations

# Stage 1 — hybrid retrieval
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
CANDIDATE_POOL_SIZE = 100
HYBRID_EMBEDDING_WEIGHT = 0.6
HYBRID_KEYWORD_WEIGHT = 0.4

# Stage 2 — LLM rating
RATING_BATCH_SIZE = 10
RATING_MAX_WORKERS = 5  # parallel calls
RELEVANCE_SCORE_THRESHOLD = 3  # keep papers scored >= this (0-5 scale)
ABSTRACT_TRUNCATE_CHARS = 1500
