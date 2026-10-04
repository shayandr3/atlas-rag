"""L2 semantic cache index (spec §9): brute-force cosine over cached answer vectors.

Kept in-process (numpy) because the candidate set is capped (~2k entries, ~3MB).
The Redis hash is the durable copy; the local index is (re)built lazily from it.
"""

import base64
from typing import Any

import numpy as np
from orjson import loads


def encode_vector(dense: list[float]) -> bytes:
    return np.asarray(dense, dtype=np.float32).tobytes()


def decode_vector(blob: bytes) -> Any:
    return np.frombuffer(blob, dtype=np.float32)


class SemanticIndex:
    def __init__(self, threshold: float, max_entries: int) -> None:
        self._threshold = threshold
        self._max_entries = max_entries
        self._vecs: dict[str, Any] = {}
        self._order: list[str] = []

    def load(self, raw: dict[bytes, bytes]) -> None:
        """(Re)build the local index from a Redis HGETALL of field -> {vec, answer}."""
        self._vecs.clear()
        self._order.clear()
        for field, blob in raw.items():
            try:
                entry = loads(blob)
                vec = np.frombuffer(base64.b64decode(entry["vec"]), dtype=np.float32)
                self._vecs[field.decode()] = vec
                self._order.append(field.decode())
            except Exception:
                continue

    def lookup(self, vec: Any) -> str | None:
        """Return the answer-ref key of the closest entry above threshold, else None."""
        if not self._order or vec is None:
            return None
        ids = list(self._order)
        matrix = np.stack([self._vecs[i] for i in ids])
        query = np.asarray(vec, dtype=np.float32)
        matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        query = query / (np.linalg.norm(query) + 1e-9)
        scores = matrix @ query
        best = int(np.argmax(scores))
        return ids[best] if float(scores[best]) >= self._threshold else None

    def add(self, field: str, vec_blob: bytes) -> str | None:
        """Insert; if over capacity, evict the oldest and return its field for HDEL."""
        if field in self._vecs:
            return None
        self._vecs[field] = decode_vector(vec_blob)
        self._order.append(field)
        if len(self._order) <= self._max_entries:
            return None
        oldest = self._order.pop(0)
        self._vecs.pop(oldest, None)
        return oldest
