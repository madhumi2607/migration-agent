"""Retrieval evaluation: how often the right section is in the top k, per ranking strategy.

Queries are phrased the way the agent actually searches (identifiers from errors and old
code), plus a few questions in words. A hit = a top-k chunk whose heading path (target
"title:...") or text (target "text:...") contains the target.

    uv run python -m mcp_servers.docs_search.eval
"""

from mcp_servers.docs_search.build import index_path
from mcp_servers.docs_search.index import DocsIndex, reciprocal_rank_fusion

QUERIES: dict[str, str] = {
    # identifiers, as they appear in errors / old code
    "pydantic.typing is_classvar": "text:is_classvar",
    "pydantic.fields.ModelField": "text:ModelField",
    "pydantic.error_wrappers.ErrorWrapper": "text:ErrorWrapper",
    "orm_mode": "title:Changes to config",
    "allow_population_by_field_name": "text:populate_by_name",
    "__fields__": "text:model_fields",
    "parse_obj": "text:model_validate",
    "update_forward_refs": "text:model_rebuild",
    "json_encoders": "text:json_encoders",
    "GenericModel": "title:GenericModel",
    "BaseSettings": "title:BaseSettings",
    "conlist min_items": "title:Constrained types",
    "root_validator": "title:Changes to validators",
    "validator pre=True always=True": "title:Changes to validators",
    # questions in words
    "how do I make a field optional with a default": "title:Required, optional, and nullable",
    "my custom type uses __get_validators__": "title:Defining custom types",
}


def hit(chunk, target: str) -> bool:
    kind, _, want = target.partition(":")
    haystack = chunk.title if kind == "title" else chunk.text
    return want in haystack.replace("`", "")


def evaluate(index: DocsIndex, k: int = 3) -> dict[str, int]:
    position = {c.id: i for i, c in enumerate(index.chunks)}
    strategies = {
        "keyword": lambda kw, sem: kw,
        "semantic": lambda kw, sem: sem,
        "hybrid, equal weights": lambda kw, sem: reciprocal_rank_fusion([kw, sem]),
        "hybrid (as shipped)": None,  # keyword x2, see index.py
    }
    scores = {name: 0 for name in strategies}
    for query, target in QUERIES.items():
        sem = [position[i] for i in index.collection.query(query_texts=[query], n_results=20, include=[])["ids"][0]]
        kw = index._bm25.rank(query)
        for name, rank in strategies.items():
            top = index.search(query, k=k) if rank is None else [index.chunks[i] for i in rank(kw, sem)[:k]]
            if any(hit(c, target) for c in top):
                scores[name] += 1
            elif name == "hybrid (as shipped)":
                print(f"  miss: {query!r} (wanted {target}); got {[c.title.split(' › ')[-1] for c in top]}")
    return scores


def main() -> None:
    index = DocsIndex(index_path())
    for k in (1, 3):
        print(f"top-{k} hits out of {len(QUERIES)}:")
        for name, score in evaluate(index, k).items():
            print(f"  {name:22} {score}")


if __name__ == "__main__":
    main()
