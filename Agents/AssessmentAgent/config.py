#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AssessmentAgent/config.py
details DetectionAgent/config.py：details（details/details）。

details：
- details Detection details JSON + RAG artifacts + details LOCAL_MODEL。
- details Assessment details，details/details。
"""

# details“details Agent details”：
# - details repo details（models/、RAG/）
# - details
from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """
    details repo details。
    details：details `models/` details `RAG/`。
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    return cur.parents[1] if len(cur.parents) > 1 else cur


_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_AGENT_DIR)

# RAG artifacts details（details/details/details）
RAG_ARTIFACTS_DIR = str(_REPO_ROOT / "RAG" / "artifacts")

# details（details）
LOCAL_LLM_MODEL_PATH = str(_REPO_ROOT / "models" / "LOCAL_MODEL" / "LOCAL_MODEL")

# details（3-class）
DAMAGE_LEVELS = ["minor", "moderate", "severe"]

# details RAG details
DEFAULT_GOV_TOP_K = 5
DEFAULT_HISTORY_TOP_K = 5

# Detection Step3 details（confirmed / uncertain / pseudo）details Assessment details
# - confirmed: details（details）
# - uncertain: details（details；details severe）
# - pseudo: details（details/details）
INCLUDE_UNCERTAIN = True
UNCERTAIN_WEIGHT = 0.3

# details：details“details LLM details”（details，details）
ENABLE_OVERALL_LLM_SUMMARY = False

# details（details，details/details；details/details）
MAX_NEW_TOKENS_PER_CHANGE = 400
TEMPERATURE = 0.2
TOP_P = 0.9

