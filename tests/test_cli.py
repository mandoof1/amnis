"""Argument parsing — the 0.1 CLI swallowed flags into the payload."""

from __future__ import annotations

import pytest

from amnis.__main__ import build_parser


def test_flags_are_not_absorbed_into_the_fact():
    args = build_parser().parse_args(["remember", "I use fish", "--importance", "9", "--tags", "shell,setup"])
    assert args.fact == "I use fish"
    assert args.importance == 9
    assert args.tags == ["shell", "setup"]


def test_importance_is_range_checked():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["remember", "x", "--importance", "42"])


def test_every_documented_command_exists():
    parser = build_parser()
    for command in (
        "server",
        "web",
        "remember",
        "recall",
        "forget",
        "search",
        "index",
        "index-notes",
        "index-wiki",
        "reindex",
        "compile-wiki",
        "wiki-query",
        "wiki-lint",
        "consolidate",
        "prune",
        "episodic-log",
        "episodic-recall",
        "episodic-prune",
        "status",
        "init",
    ):
        assert parser.parse_args(
            [command]
            if command
            not in ("remember", "recall", "forget", "search", "index", "wiki-query", "episodic-log")
            else _minimal_args(command)
        )


def _minimal_args(command):
    return {
        "remember": ["remember", "a fact"],
        "recall": ["recall"],
        "forget": ["forget", "an-id"],
        "search": ["search", "a query"],
        "index": ["index", "/tmp/x.md"],
        "wiki-query": ["wiki-query", "a question"],
        "episodic-log": ["episodic-log", "s", "user", "content"],
    }[command]


def test_prune_dry_run_defaults_to_false():
    assert build_parser().parse_args(["prune"]).dry_run is False
    assert build_parser().parse_args(["prune", "--dry-run"]).dry_run is True
