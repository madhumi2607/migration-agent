"""Hybrid search over doc chunks: keyword (BM25) + semantic (ChromaDB), merged by rank.

The agent mostly searches for exact identifiers from errors and old code (ModelField,
orm_mode, __fields__); the guide uses exactly those identifiers, so keyword ranking is the
backbone. Semantic search adds recall for questions phrased in words. Measured on 16
agent-style queries (mcp_servers/docs_search/eval.py), right section in the top 3:
    semantic only 8/16 · keyword only 15/16 · equal-weight fusion 15/16 · keyword x2 fusion 16/16
Equal weighting let semantic hits that are "close but generic" outrank the one chunk that
names the identifier, so the keyword ranking counts double.
"""

import json
import math
import re
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mcp_servers.docs_search.corpus import Chunk

COLLECTION = "migration-docs"
# Reciprocal-rank-fusion settings, chosen with eval.py (see the module docstring). The constant
# matters little here (k=5 and k=60 tie); the weight does.
RRF_K = 5
KEYWORD_WEIGHT = 2.0
SEMANTIC_WEIGHT = 1.0

TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def tokenize(text: str) -> list[str]:
    """Identifiers stay whole (ModelField, orm_mode, __fields__), and dotted paths also yield
    their parts (pydantic.fields.ModelField -> pydantic, fields, ModelField)."""
    return [t.lower() for t in TOKEN.findall(text)]


class BM25:
    """Okapi BM25 over the chunks' title + text. Small corpus, so a plain in-memory index."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(c.values()) for c in self.docs]
        self.avg_len = sum(self.lengths) / max(len(self.docs), 1)
        df = Counter(term for c in self.docs for term in c)
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def rank(self, query: str) -> list[int]:
        """Doc indexes with a positive score, best first."""
        terms = set(tokenize(query))
        scores = []
        for i, counts in enumerate(self.docs):
            score = 0.0
            for t in terms:
                tf = counts.get(t, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_len)
                    score += self.idf[t] * tf * (self.k1 + 1) / norm
            if score > 0:
                scores.append((score, i))
        return [i for _, i in sorted(scores, reverse=True)]


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = RRF_K, weights: list[float] | None = None) -> list[int]:
    """Merge rankings: each item scores sum(weight / (k + rank)) over the lists it appears in.
    Uses ranks only, so the lists' raw scores (BM25 vs cosine) never need to be comparable."""
    weights = weights or [1.0] * len(rankings)
    scores: dict[int, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank + 1)
    return sorted(scores, key=lambda i: scores[i], reverse=True)


class DocsIndex:
    """Chunks on disk (chunks.json) + their embeddings in a persistent Chroma collection."""

    def __init__(self, path: Path, embedding_function: Any = None) -> None:
        import chromadb  # heavy import: only when an index is actually used

        self.path = path
        self.client = chromadb.PersistentClient(path=str(path / "chroma"))
        kwargs = {"embedding_function": embedding_function} if embedding_function is not None else {}
        self.collection = self.client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"}, **kwargs)
        chunks_file = path / "chunks.json"
        self.chunks = [Chunk(**c) for c in json.loads(chunks_file.read_text(encoding="utf-8"))] if chunks_file.is_file() else []
        self._bm25 = BM25([f"{c.title}\n{c.text}" for c in self.chunks])

    def build(self, chunks: list[Chunk]) -> None:
        """Replace the index contents with these chunks."""
        self.path.mkdir(parents=True, exist_ok=True)
        existing = self.collection.get(include=[])["ids"]
        if existing:
            self.collection.delete(ids=existing)
        if chunks:
            self.collection.add(
                ids=[c.id for c in chunks],
                documents=[f"{c.title}\n{c.text}" for c in chunks],
                metadatas=[{"library": c.library, "source": c.source} for c in chunks],
            )
        (self.path / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
        self.chunks = list(chunks)
        self._bm25 = BM25([f"{c.title}\n{c.text}" for c in self.chunks])

    def search(self, query: str, k: int = 4, library: str | None = None) -> list[Chunk]:
        """Top-k chunks for the query, by fused keyword + semantic rank."""
        if not self.chunks:
            return []
        allowed = [i for i, c in enumerate(self.chunks) if library is None or c.library == library]
        if not allowed:
            return []
        position = {c.id: i for i, c in enumerate(self.chunks)}
        n = min(len(allowed), max(k * 5, 20))
        result = self.collection.query(
            query_texts=[query], n_results=n, include=[],
            where={"library": library} if library else None,
        )
        semantic = [position[i] for i in result["ids"][0] if i in position]
        allowed_set = set(allowed)
        keyword = [i for i in self._bm25.rank(query) if i in allowed_set]
        fused = reciprocal_rank_fusion([keyword, semantic], weights=[KEYWORD_WEIGHT, SEMANTIC_WEIGHT])
        return [self.chunks[i] for i in fused[:k]]
