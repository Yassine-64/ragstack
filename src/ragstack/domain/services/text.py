from __future__ import annotations

import re

_WS = re.compile(r"\s+")


def normalize_whitespace(text: str) -> str:
    return _WS.sub(" ", text).strip()


def estimate_tokens(text: str) -> int:
    """~4 chars per token for English/French prose. ponytail: swap for a tokenizer if budgets get tight."""
    return max(1, len(text) // 4)
