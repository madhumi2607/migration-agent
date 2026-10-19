"""Download the sources and build the docs index. Run once (and after changing SOURCES):
    uv run python -m mcp_servers.docs_search.build
The first build also downloads Chroma's local embedding model (~80 MB, cached).
"""

import os
from pathlib import Path

from mcp_servers.docs_search.corpus import SOURCES, chunk_markdown, download
from mcp_servers.docs_search.index import DocsIndex

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX = PROJECT_ROOT / "data" / "docs_index"


def index_path() -> Path:
    return Path(os.environ.get("DOCS_INDEX_DIR", DEFAULT_INDEX))


def main() -> None:
    path = index_path()
    chunks = []
    for source in SOURCES:
        text = download(source, path / "sources")
        source_chunks = chunk_markdown(text, source)
        print(f"{source.id}: {len(text):,} chars -> {len(source_chunks)} chunks")
        chunks += source_chunks
    DocsIndex(path).build(chunks)
    print(f"Indexed {len(chunks)} chunks into {path}")


if __name__ == "__main__":
    main()
