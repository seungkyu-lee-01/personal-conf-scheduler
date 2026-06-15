"""Expand free-form research interest into a refined query + keyword list via LLM."""

import json
import logging
from typing import Dict, List

from .llm_provider import call_llm_json_schema
from .config import OPENAI_FAST_MODEL, LLM_MAX_TOKENS_FAST

logger = logging.getLogger(__name__)


_SCHEMA = {
    "type": "object",
    "properties": {
        "refined_query": {
            "type": "string",
            "description": "A concise paraphrase of the user's research interest, optimized as a dense semantic query for retrieving relevant academic papers.",
        },
        "keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "8-15 specific technical terms, method names, or topic phrases the user is likely interested in. Lowercased, no duplicates.",
        },
        "negative_keywords": {
            "type": "array",
            "items": {"type": "string"},
            "description": "0-8 terms the user explicitly excluded or that are clearly off-topic. Lowercased.",
        },
    },
    "required": ["refined_query", "keywords", "negative_keywords"],
    "additionalProperties": False,
}


_SYSTEM_PROMPT = """\
You are a research assistant helping a researcher triage NLP/ML conference papers.
Given a free-form description of their research interests, produce:
1. A refined semantic query (1-3 sentences) suitable for embedding-based retrieval.
2. A list of specific technical keywords likely to appear in relevant paper titles/abstracts.
   Prefer multi-word phrases over single common words. Include method names, task names,
   model families, and well-known acronyms when appropriate.
3. A list of negative keywords for topics the user excluded.

Output JSON only, matching the schema.
"""


def expand_interest(interest_text: str) -> Dict[str, List[str]]:
    """Return {'refined_query': str, 'keywords': [str], 'negative_keywords': [str]}."""
    user_prompt = f"Research interest description:\n\n{interest_text.strip()}"
    raw = call_llm_json_schema(
        model_name=OPENAI_FAST_MODEL,
        max_tokens=LLM_MAX_TOKENS_FAST,
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_schema=_SCHEMA,
        schema_name="ResearchInterestExpansion",
        temperature=0.0,
    )
    parsed = json.loads(raw)
    parsed["keywords"] = [k.strip().lower() for k in parsed.get("keywords", []) if k.strip()]
    parsed["negative_keywords"] = [k.strip().lower() for k in parsed.get("negative_keywords", []) if k.strip()]
    logger.info(
        f"[InterestExpander] refined_query={parsed['refined_query'][:80]}... "
        f"keywords={len(parsed['keywords'])} neg={len(parsed['negative_keywords'])}"
    )
    return parsed
