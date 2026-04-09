"""Step 2: expectations YAML load and archetype merge."""

from pathlib import Path

import pytest

from governance_engine.expectations import load_expectations, merged_bands_for_archetype


def test_load_expectations_default() -> None:
    cfg = load_expectations()
    assert cfg.get("version") == 1
    assert "defaults" in cfg
    assert "weights" in cfg
    assert "risk_level_cutoffs" in cfg


def test_merged_bands_llm_rag() -> None:
    cfg = load_expectations()
    b = merged_bands_for_archetype(cfg, "llm_rag")
    assert b["min_grounding_score"] >= cfg["defaults"]["min_grounding_score"]
    assert "min_context_retrieval_hit_rate" in b


def test_merged_bands_unknown_archetype_uses_defaults() -> None:
    cfg = load_expectations()
    b = merged_bands_for_archetype(cfg, "tabular_ml")
    assert b["min_accuracy_pct"] == cfg["defaults"]["min_accuracy_pct"]
