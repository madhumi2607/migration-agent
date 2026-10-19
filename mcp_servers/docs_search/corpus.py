"""The documents docs_search knows about, and how they are split into searchable chunks.

Sources are pinned to a release so every run (and every benchmark experiment) searches
exactly the same text. Adding a source is one entry in SOURCES.
"""

import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    id: str
    library: str
    url: str  # raw markdown, pinned to a tag
    docs_url: str  # the rendered page, for links back to a section


SOURCES = [
    Source(
        id="pydantic-v2-migration",
        library="pydantic",
        url="https://raw.githubusercontent.com/pydantic/pydantic/v2.14.1/docs/migration.md",
        docs_url="https://docs.pydantic.dev/2.14/migration/",
    ),
]


@dataclass(frozen=True)
class Chunk:
    id: str
    source: str
    library: str
    title: str  # heading path, e.g. "Migration guide › Changes to config"
    url: str  # rendered docs URL with the section's anchor
    text: str


HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")


def slugify(heading: str) -> str:
    """Approximate the anchor mkdocs generates for a heading."""
    text = re.sub(r"[`*_]", "", heading).lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "-", text).strip("-")


def chunk_markdown(markdown: str, source: Source, max_chars: int = 2500) -> list[Chunk]:
    """Split a markdown document at its headings, ignoring '#' lines inside code blocks
    (the Pydantic guide's examples are full of '# V1' / '# V2' comments).

    Each chunk carries its heading path, so a hit reads as a location in the guide, not a
    loose paragraph. Sections longer than max_chars are split at paragraph boundaries.
    """
    sections: list[tuple[list[str], list[str]]] = []  # (heading path, lines)
    stack: list[tuple[int, str]] = []  # (level, heading) of the current section and its parents
    lines: list[str] = []
    in_code = False
    for line in markdown.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if match:
            if any(l.strip() for l in lines):
                sections.append(([h for _, h in stack], lines))
            level = len(match[1])
            # Pop back to this heading's parent. A stack (not path[:level-1]) handles documents
            # that start at "##" with no "#" title, like the Pydantic migration guide.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match[2].strip()))
            lines = []
        else:
            lines.append(line)
    if any(l.strip() for l in lines):
        sections.append(([h for _, h in stack], lines))

    chunks = []
    for path, body in sections:
        title = " › ".join(path) or source.id
        anchor = f"#{slugify(path[-1])}" if path else ""
        for i, piece in enumerate(_split(("\n".join(body)).strip(), max_chars)):
            chunks.append(Chunk(
                id=f"{source.id}:{len(chunks)}", source=source.id, library=source.library,
                title=title + (f" (part {i + 1})" if i else ""), url=source.docs_url + anchor,
                text=piece,
            ))
    return chunks


def _split(text: str, max_chars: int) -> list[str]:
    """Split at blank lines into pieces of at most ~max_chars, never inside a code block."""
    if len(text) <= max_chars:
        return [text]
    blocks, current, in_code = [], [], False
    for line in text.splitlines():
        if FENCE.match(line):
            in_code = not in_code
        current.append(line)
        if not in_code and line.strip() == "" and sum(len(l) + 1 for l in current) >= max_chars * 0.6:
            blocks.append("\n".join(current).strip())
            current = []
    if current:
        blocks.append("\n".join(current).strip())
    pieces, piece = [], ""
    for block in blocks:
        if piece and len(piece) + len(block) > max_chars:
            pieces.append(piece)
            piece = block
        else:
            piece = f"{piece}\n\n{block}" if piece else block
    if piece:
        pieces.append(piece)
    return pieces


def download(source: Source, cache_dir: Path) -> str:
    """The source's markdown, downloaded once into cache_dir."""
    path = cache_dir / f"{source.id}.md"
    if not path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(source.url, timeout=30) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8")
