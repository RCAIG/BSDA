#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assessment Agent CLI (placed under AssessmentAgent/ as requested).

Usage:
  python AssessmentAgent/main.py --input_detection detection_output.json --output_assessment assessment_output.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict


def _ensure_repo_root_on_path() -> None:
    """
    details，details Agent details（details repo_root/Agents/AssessmentAgent）。
    details repo_root details repo_root/Agents details sys.path。
    """
    here = Path(__file__).resolve().parent
    # locate repo root by markers
    repo_root = None
    for p in [here] + list(here.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            repo_root = p
            break
    if repo_root is None:
        repo_root = here.parent  # fallback

    agents_root = repo_root / "Agents"
    for path in [repo_root, agents_root]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_json(path: Path) -> Dict[str, Any]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Input JSON must be an object/dict, got: {type(obj)}")
    return obj


def main() -> None:
    _ensure_repo_root_on_path()

    from AssessmentAgent.assessment_agent import AssessmentAgent, run_assessment  # noqa: WPS433
    from AssessmentAgent.config import DEFAULT_GOV_TOP_K, DEFAULT_HISTORY_TOP_K, MAX_NEW_TOKENS_PER_CHANGE  # noqa: WPS433

    p = argparse.ArgumentParser(description="Assessment Agent CLI")
    p.add_argument("--input_detection", type=str, required=True, help="Detection Agent details JSON details")
    p.add_argument("--output_assessment", type=str, default="", help="Assessment details JSON details（details）")
    p.add_argument("--hazard_type", type=str, default="hurricane", help="details（details query details）")
    p.add_argument("--gov_top_k", type=int, default=int(DEFAULT_GOV_TOP_K), help="details top_k")
    p.add_argument("--history_top_k", type=int, default=int(DEFAULT_HISTORY_TOP_K), help="details top_k")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=int(MAX_NEW_TOKENS_PER_CHANGE),
        help="details change details token details（details）",
    )
    args = p.parse_args()

    # Windows console encoding can be cp1252/gbk etc; ensure JSON (with non-ascii) can be printed.
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

    in_path = Path(args.input_detection)
    if not in_path.exists():
        raise FileNotFoundError(f"Input detection JSON not found: {in_path}")

    detection_result = _load_json(in_path)

    # Create a fresh agent for CLI runs so we can apply per-run overrides (token limits, etc.).
    agent = AssessmentAgent(max_new_tokens=int(args.max_new_tokens))
    assessment = run_assessment(
        detection_result,
        agent=agent,
        hazard_type=str(args.hazard_type),
        gov_top_k=int(args.gov_top_k),
        history_top_k=int(args.history_top_k),
    )

    out_text = json.dumps(assessment, ensure_ascii=False, indent=2)
    if args.output_assessment:
        out_path = Path(args.output_assessment)
        os.makedirs(str(out_path.parent), exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
    else:
        print(out_text)


if __name__ == "__main__":
    main()

