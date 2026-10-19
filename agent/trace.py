"""Record a run to disk so it can be inspected, compared and replayed.

Layout of one run:
    runs/<run-id>/
        meta.json      who/what/when: target repo and commit, model, settings
        events.jsonl   one JSON object per event, in order (see agent/loop.py run_agent)
        result.json    the RunResult, written when the run ends
        diff.patch     the agent's changes to the target repo (git diff)

Events hold only what is new (one model turn, one tool result), not the whole conversation,
so the files grow linearly with the run; the conversation is rebuilt by replaying them.
"""

import json
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any


class RunTrace:
    """An event handler for run_agent that appends each event to events.jsonl."""

    def __init__(self, path: Path, repo: Path) -> None:
        self.path = path
        self.repo = repo
        self._events: IO[str] = (path / "events.jsonl").open("a", encoding="utf-8")

    @classmethod
    def create(cls, runs_dir: str | Path, repo: str | Path, meta: dict[str, Any]) -> "RunTrace":
        """Make a new run directory (named by start time and repo) and write meta.json."""
        repo = Path(repo).resolve()
        started = datetime.now(timezone.utc)
        run_id = f"{started:%Y%m%d-%H%M%S}-{repo.name}"
        path = Path(runs_dir) / run_id
        path.mkdir(parents=True, exist_ok=False)
        meta = {"run_id": run_id, "started_at": started.isoformat(), "repo": str(repo),
                "repo_commit": _git(repo, "rev-parse", "HEAD").strip() or None, **meta}
        (path / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        return cls(path, repo)

    def __call__(self, event: dict[str, Any]) -> None:
        self._events.write(json.dumps(event, default=str) + "\n")
        self._events.flush()  # so a live dashboard (or a crash) sees every event written so far

    def finish(self, result: Any) -> None:
        """Write result.json and the agent's diff, then close the events file."""
        data = asdict(result)
        data["finished_at"] = datetime.now(timezone.utc).isoformat()
        (self.path / "result.json").write_text(json.dumps(data, indent=1, default=str), encoding="utf-8")
        (self.path / "diff.patch").write_text(_git(self.repo, "diff"), encoding="utf-8")
        self._events.close()


def _git(repo: Path, *args: str) -> str:
    """Output of a git command in the target repo; empty if it isn't a git checkout."""
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, encoding="utf-8")
    return proc.stdout if proc.returncode == 0 else ""
