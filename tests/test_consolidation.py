"""Extraction heuristics and importance scoring (no embeddings involved)."""

from __future__ import annotations

import pytest

from amnis.memory import consolidation


@pytest.mark.parametrize(
    "line",
    [
        "You prefer dark mode in every editor you open",
        "You use CachyOS as your daily driver operating system",
        "Your primary machine is a desktop with a discrete GPU",
    ],
)
def test_declarative_statements_are_kept(line):
    assert consolidation.looks_like_fact(line)


@pytest.mark.parametrize(
    "line",
    [
        "You should restart the daemon before trying that again",  # advice
        "Do you want me to go ahead and rebuild the whole index?",  # question
        "- you can also pass --verbose to see the full output here",  # list item
        "1. You open the settings panel and scroll to the bottom",  # numbered step
        "If you prefer dark mode, flip the toggle in preferences",  # conditional
        "```you prefer this to be treated as code, not as a fact```",  # code fence
        "You",  # too short
    ],
)
def test_non_facts_are_rejected(line):
    assert not consolidation.looks_like_fact(line)


def test_extract_candidates_filters_a_realistic_turn():
    turn = (
        "Let me check that for you.\n"
        "You prefer dark mode in all of your editors and terminals.\n"
        "You should restart the shell to pick that up.\n"
        "- you can also export the setting to a dotfile\n"
        "You use fish as your interactive login shell every day.\n"
    )
    facts = consolidation.extract_candidates(turn)
    assert len(facts) == 2
    assert all(f.startswith("You prefer") or f.startswith("You use") for f in facts)


def test_polarity_uses_whole_words():
    # "has" must not match inside "hash", "no" must not match inside "notes".
    assert consolidation._polarity("the hash of your notes") == 0.0
    assert consolidation._polarity("you love this") > 0
    assert consolidation._polarity("you hate this") < 0


def test_importance_keywords_come_from_config(amnis_env, monkeypatch):
    monkeypatch.setattr(consolidation.config, "importance_keywords", [])
    baseline = consolidation.compute_importance("You run cachyos on the desktop")
    monkeypatch.setattr(consolidation.config, "importance_keywords", ["cachyos"])
    boosted = consolidation.compute_importance("You run cachyos on the desktop")
    assert boosted == baseline + 1


def test_pluggable_extractor_replaces_the_heuristic():
    consolidation.set_extractor(lambda content: ["injected fact"])
    try:
        assert consolidation.extract_candidates("anything at all") == ["injected fact"]
    finally:
        consolidation.set_extractor(None)
