"""LLM Provider for OpenAI (chat completions + structured JSON schema).

Exposes the same two functions the pipeline depends on:
    - call_llm(...)              : text completion
    - call_llm_json_schema(...)  : structured JSON-schema completion
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Load OPENAI_API_KEY from the repo-root .env before instantiating the client.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from openai import OpenAI

from .config import LLM_TEMPERATURE

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _client() -> OpenAI:
    return OpenAI()


def call_llm(
    model_name: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    temperature: float = LLM_TEMPERATURE,
) -> str:
    response = _client().chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.choices[0].message.content


def call_llm_json_schema(
    model_name: str,
    max_tokens: int,
    system_prompt: str,
    user_prompt: str,
    json_schema: Dict[str, Any],
    schema_name: str,
    temperature: float = 0.0,
) -> str:
    response = _client().chat.completions.create(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            },
        },
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        raw = raw.rsplit("```", 1)[0].strip()
    return raw
