#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/rag.py

Critic details RAG details（v0.1）：
- Critic details agent，details“details/details”，details grounded。
- details gov details（details history）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def build_critic_rag_query(
    *,
    hazard_type: str,
    overall_grade: str,
    overall_confidence_label: str,
    post_scene_summary: str,
    overall_reasoning: str,
) -> str:
    hz = safe_str(hazard_type) or "hurricane"
    return (
        "You are retrieving official guidance about disaster damage severity grading.\n"
        f"Hazard type: {hz}.\n"
        "Target grades: minor / moderate / severe.\n"
        f"Current system output: overall_grade={safe_str(overall_grade)}, confidence={safe_str(overall_confidence_label)}.\n"
        f"Scene summary: {safe_str(post_scene_summary)}\n"
        f"Assessment reasoning summary: {safe_str(overall_reasoning)}\n"
        "Retrieval goal: find criteria/thresholds/examples that distinguish minor vs moderate vs severe, "
        "and guidance about handling uncertain/ambiguous evidence and confidence calibration."
    ).strip()


def search_gov_rules_for_critic(query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Reuse existing AssessmentAgent gov RAG wrapper.
    """
    q = safe_str(query_text)
    if not q:
        return []
    try:
        from AssessmentAgent.rag import search_gov_rules  # type: ignore

        hits = search_gov_rules(q, top_k=int(top_k))
        return hits if isinstance(hits, list) else []
    except Exception:
        return []

