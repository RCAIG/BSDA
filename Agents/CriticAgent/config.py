#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/config.py

details LLM details Critic details（v0.3）：
- details LOCAL_MODEL（details gov details）
- details/details
"""

from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    return cur.parents[1] if len(cur.parents) > 1 else cur


_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_AGENT_DIR)

# Version
CRITIC_VERSION = "v0.3"

# Default behavior
DEFAULT_USE_RAG = True
DEFAULT_RAG_TOP_K = 5

# Local LLM: use local LOCAL_MODEL instruct model
ENABLE_LLM_CRITIC = True
LOCAL_LLM_MODEL_PATH = str(_REPO_ROOT / "models" / "LOCAL_MODEL" / "LOCAL_MODEL")
MAX_NEW_TOKENS = 2000
# Critic details“details”，details
TEMPERATURE = 0.0
TOP_P = 0.9
DEVICE_MAP = "auto"

# Output constraints
ALLOWED_JUDGEMENTS = ["accept", "revise", "human_review"]
ALLOWED_SEVERITIES = ["minor", "major", "critical"]
ALLOWED_ISSUE_TYPES = [
    "misuse_of_uncertain",
    "misuse_of_pseudo",
    "grade_evidence_mismatch",
    "cross_agent_inconsistency",
    "confidence_mismatch",
    "other",
]
ALLOWED_GRADES = ["minor", "moderate", "severe"]
ALLOWED_CONFIDENCE_LABELS = ["low", "medium", "high"]

