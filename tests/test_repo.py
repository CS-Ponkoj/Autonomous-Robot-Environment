"""Repository hygiene checks."""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEXT_SUFFIXES = {".md", ".py", ".ps1", ".sh", ".xml", ".ini", ".txt", ".gitignore", ".gitattributes"}
SKIP_DIRS = {".git", ".venv", "qa_output", "_private", "__pycache__", ".pytest_cache"}
CONTROL = re.compile(rb"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def text_files():
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS.intersection(path.parts) and (
                path.suffix in TEXT_SUFFIXES or path.name in TEXT_SUFFIXES):
            yield path


def test_no_control_characters_in_text_files():
    """Regression: a stray backspace (U+0008) broke a README command."""
    bad = [str(p.relative_to(ROOT)) for p in text_files() if CONTROL.search(p.read_bytes())]
    assert not bad, bad


def published_text_files():
    """Text files that can be pushed: README.md is the only published Markdown file."""
    for path in text_files():
        if path.suffix == ".md" and path.name != "README.md":
            continue
        yield path


def test_published_files_contain_no_development_attribution():
    """Owner rule: published files contain no prohibited development attribution."""
    words = re.compile(("co" + "dex|cl" + "aude").encode(), re.IGNORECASE)
    bad = [str(p.relative_to(ROOT)) for p in published_text_files() if words.search(p.read_bytes())]
    assert not bad, bad


def test_readme_speed_level_table_matches_the_config():
    """The README table of speed levels is generated from SPEED_LEVELS and must not drift."""
    import os

    import pytest
    from robot_env import config as C
    if os.environ.get("ROBOT_CANDIDATE_LEVELS"):
        pytest.skip("candidate levels are validated before the README lists them")
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    start = text.index("<!-- speed-levels:")
    block = text[start:text.index("<!-- /speed-levels -->", start)]
    rows = [line.strip() for line in block.splitlines() if line.strip().startswith("| ") and "m/s" in line]
    expected = [f"| {i + 1}{' (default)' if i == C.DEFAULT_SPEED_LEVEL else ''} | {v:.2f} m/s | {w:.1f} rad/s |"
                for i, (v, w) in enumerate(C.SPEED_LEVELS)]
    assert rows == expected
