"""Automatic review checks on a run's final diff: catch migrations that pass the tests but are
still wrong. Findings are advisory (they don't fail the run); they feed the human review step
and the benchmark's failure categories.

Why: in one run the harness blocked a correct dependency bump, and the agent passed every
test by hand-rolling BaseSettings and pinning `pydantic>=1.10` (which still allows v1).

The "after" state is rebuilt from the task's commit plus the run's diff.patch in a temporary
clone, so the same code checks a live run and any past run.

Check a recorded run:
    uv run python -m agent.checks runs/<run-id>
"""

import json
import re
import shlex
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import Version

# Imports that keep the old API alive instead of migrating (forbidden by the prompt).
SHIM_PATTERNS = {"pydantic": [r"\bpydantic\.v1\b", r"from\s+pydantic\s+import\s+v1\b"]}
SKIP_MARKERS = re.compile(r"pytest\.mark\.(skip|skipif|xfail)\b|pytest\.(skip|xfail)\(")
TEST_PATH = re.compile(r"(^|/)(tests?/|conftest\.py$|test_[^/]*\.py$|[^/]*_test\.py$)")


@dataclass
class Finding:
    severity: str  # "error" | "warning"
    check: str
    message: str
    file: str | None = None


# ---------- dependency declarations ----------

def poetry_to_pep440(spec: str) -> str:
    """Translate Poetry's ^ / ~ / * shorthand into a PEP 440 specifier, e.g. "^1.2" -> ">=1.2,<2"."""
    out = []
    for part in (p.strip() for p in spec.split(",")):
        if part in ("", "*"):
            continue
        if part.startswith("^"):
            v = Version(part[1:])
            nums = list(v.release) + [0, 0]
            # Caret: bump the first non-zero component (^0.3 -> <0.4, ^1.2 -> <2).
            idx = next((i for i, n in enumerate(nums[:3]) if n != 0), 0)
            upper = nums[: idx + 1]
            upper[idx] += 1
            out += [f">={part[1:]}", f"<{'.'.join(map(str, upper))}"]
        elif part.startswith("~") and not part.startswith("~="):
            v = Version(part[1:])
            nums = list(v.release) + [0]
            upper = [nums[0], nums[1] + 1] if len(v.release) >= 2 else [nums[0] + 1]
            out += [f">={part[1:]}", f"<{'.'.join(map(str, upper))}"]
        else:
            out.append(part)
    return ",".join(out)


def read_dependencies(repo: Path) -> dict[str, SpecifierSet]:
    """Runtime dependencies declared by a repo: name -> version range. Combines pyproject.toml
    ([project] and [tool.poetry]) and top-level requirements files (not test/dev ones)."""
    deps: dict[str, SpecifierSet] = {}

    def add(requirement: str) -> None:
        try:
            req = Requirement(requirement)
        except InvalidRequirement:
            return
        deps[canonicalize_name(req.name)] = req.specifier

    pyproject = repo / "pyproject.toml"
    if pyproject.is_file():
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        for requirement in data.get("project", {}).get("dependencies", []):
            add(requirement)
        for name, spec in data.get("tool", {}).get("poetry", {}).get("dependencies", {}).items():
            if name.lower() == "python":
                continue
            version = spec.get("version", "") if isinstance(spec, dict) else str(spec)
            try:
                deps[canonicalize_name(name)] = SpecifierSet(poetry_to_pep440(version))
            except (InvalidSpecifier, ValueError):
                deps[canonicalize_name(name)] = SpecifierSet()
    for path in sorted(repo.glob("*.txt")):
        name = path.name.lower()
        if name.startswith("requirements") and not any(k in name for k in ("test", "dev", "doc", "lint")):
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith("-"):
                    add(line)
    return deps


def _minimum(spec: SpecifierSet) -> Version | None:
    """The lowest version a range allows, as far as its >= / == / ~= / > bounds say (None = unbounded)."""
    lows = [Version(s.version) for s in spec if s.operator in (">=", "==", "~=", ">") and "*" not in s.version]
    return max(lows) if lows else None


def check_pins(before: dict[str, SpecifierSet], after: dict[str, SpecifierSet], upgrade: str) -> list[Finding]:
    """Each upgraded package's declared range must start at or above the upgrade's minimum."""
    findings = []
    for requirement in shlex.split(upgrade):
        req = Requirement(requirement)
        name = canonicalize_name(req.name)
        required_min = _minimum(req.specifier)
        if name not in after:
            if name in before:
                findings.append(Finding("error", "pins", f"{req.name} was declared before the run but is missing now."))
            continue
        declared = after[name]
        declared_min = _minimum(declared)
        # Excluded = no version satisfies both ranges. Candidates are each range's lower bound:
        # if any version is allowed by both, the larger of the two lower bounds is (for >=/< ranges).
        candidates = [v for v in (required_min, declared_min) if v is not None]
        if candidates and not any(declared.contains(v) and req.specifier.contains(v) for v in candidates):
            findings.append(Finding("error", "pins", f"{req.name} is pinned to '{declared}', which excludes the "
                                                     f"upgrade target {req.specifier}."))
            continue
        if required_min is not None and (declared_min is None or declared_min < required_min):
            allowed = f"'{declared}'" if str(declared) else "any version"
            findings.append(Finding("warning", "pins", f"{req.name} is pinned to {allowed}, which still allows "
                                                       f"versions below {required_min} that the migrated code doesn't support."))
    return findings


def check_removed_dependencies(before: dict[str, SpecifierSet], after: dict[str, SpecifierSet]) -> list[Finding]:
    return [Finding("warning", "removed_dependency", f"Dependency '{name}' was removed. Make sure the code "
                                                     "that needed it wasn't replaced by a hand-rolled substitute.")
            for name in sorted(set(before) - set(after))]


# ---------- the diff itself ----------

def parse_diff(diff: str) -> dict[str, list[str]]:
    """Added lines per file ("b/" path) in a unified diff."""
    added: dict[str, list[str]] = {}
    current = None
    for line in diff.splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            current = path[2:] if path.startswith("b/") else None  # /dev/null for deletions
            if current is not None:
                added.setdefault(current, [])
        elif line.startswith("diff --git "):
            current = None
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                added.setdefault(parts[1], [])  # also records files that were deleted or renamed
        elif current is not None and line.startswith("+") and not line.startswith("+++"):
            added[current].append(line[1:])
    return added


def check_diff(diff: str, library: str) -> list[Finding]:
    findings = []
    patterns = [re.compile(p) for p in SHIM_PATTERNS.get(library, [])]
    for path, lines in parse_diff(diff).items():
        if TEST_PATH.search(path):
            findings.append(Finding("error", "test_integrity", "A test file was changed.", path))
        for line in lines:
            if any(p.search(line) for p in patterns):
                findings.append(Finding("error", "compat_shim", f"Compatibility shim instead of a migration: {line.strip()[:120]}", path))
            if SKIP_MARKERS.search(line):
                findings.append(Finding("error", "test_integrity", f"Skip/xfail marker added: {line.strip()[:120]}", path))
    return findings


# ---------- putting it together ----------

def run_checks(repo: Path, base_commit: str, diff: str, upgrade: str, library: str) -> list[Finding]:
    """All checks for one run: the repo at base_commit is "before"; base_commit + diff is "after"."""
    with tempfile.TemporaryDirectory(prefix="checks-") as tmp:
        work = Path(tmp) / "repo"
        _git(None, "clone", "--quiet", "--shared", "--no-checkout", str(repo), str(work))
        _git(work, "checkout", "--quiet", base_commit)
        before = read_dependencies(work)
        if diff.strip():
            patch = Path(tmp) / "run.patch"
            patch.write_text(diff, encoding="utf-8")
            _git(work, "apply", "--whitespace=nowarn", str(patch))
        after = read_dependencies(work)
    return [*check_pins(before, after, upgrade), *check_removed_dependencies(before, after), *check_diff(diff, library)]


def check_run(run_dir: Path) -> list[Finding]:
    """Run the checks for a recorded run (runs/<id>/) and write checks.json next to it."""
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    start = json.loads((run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()[0])
    diff_path = run_dir / "diff.patch"
    findings = run_checks(
        repo=Path(meta["repo"]),
        base_commit=meta["repo_commit"],
        diff=diff_path.read_text(encoding="utf-8") if diff_path.is_file() else "",
        upgrade=(meta.get("sandbox") or {}).get("SANDBOX_UPGRADE") or "",
        library=start["task"]["library"],
    )
    (run_dir / "checks.json").write_text(json.dumps([asdict(f) for f in findings], indent=1), encoding="utf-8")
    return findings


def _git(cwd: Path | None, *args: str) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout


def main(argv: list[str]) -> None:
    for run in argv:
        findings = check_run(Path(run))
        print(f"{run}: {len(findings)} finding(s)")
        for f in findings:
            print(f"  [{f.severity}] {f.check}: {f.message}" + (f" ({f.file})" if f.file else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
