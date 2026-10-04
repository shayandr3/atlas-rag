"""Cache key builders (spec §9): every layer versioned by corpus version + config fingerprint.

Bumping CORPUS_VERSION or changing models/prompts invalidates logically with no flush.
"""

import hashlib
import json
from typing import Any

from atlas.config import Settings
from atlas.llm.prompts import ANSWER_SYSTEM


def norm_query(query: str) -> str:
    return " ".join(query.lower().split())


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def config_fingerprint(settings: Settings) -> str:
    payload: dict[str, Any] = {
        "provider": settings.llm_provider,
        "cheap": settings.llm_cheap_model,
        "strong": settings.llm_strong_model,
        "embed": settings.embed_model,
        "sparse": settings.sparse_model,
        "top_k": settings.retrieval_top_k,
        "max_context": settings.max_context_tokens,
        "max_output": settings.max_output_tokens,
        "prompt": _sha(ANSWER_SYSTEM)[:16],
        "v": 1,
    }
    return _sha(json.dumps(payload, sort_keys=True))[:16]


class CacheKeys:
    def __init__(self, settings: Settings) -> None:
        self.corpus_version = settings.corpus_version
        self.cfg = config_fingerprint(settings)
        self.embed_model = settings.embed_model

    def embedding(self, query: str) -> str:
        return f"emb:{self.embed_model}:{_sha(norm_query(query))}"

    def response(self, query: str) -> str:
        return f"resp:{self.corpus_version}:{self.cfg}:{_sha(norm_query(query))}"

    def retrieval(self, query: str, limit: int) -> str:
        # filters arrive with M3; limit is part of the identity
        return f"ret:{self.corpus_version}:{_sha(norm_query(query) + f':limit={limit}')}"

    def semantic(self) -> str:
        return f"sem:{self.corpus_version}:{self.cfg}"

    def route(self, query: str) -> str:
        return f"route:{_sha(norm_query(query))}"

    def lock(self, query: str) -> str:
        return f"lock:{self.corpus_version}:{self.cfg}:{_sha(norm_query(query))}"
