#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/main.py

details：
- 3 details demo case（accept / revise / human_review）
- orchestrator details
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """
    details，details Agent details（details repo_root/Agents/CriticAgent）。
    details repo_root details repo_root/Agents details sys.path。
    """
    here = Path(__file__).resolve().parent
    repo_root = None
    for p in [here] + list(here.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            repo_root = p
            break
    if repo_root is None:
        repo_root = here.parent

    agents_root = repo_root / "Agents"
    for path in [repo_root, agents_root]:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _load_json(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    obj = json.loads(raw)
    if not isinstance(obj, dict):
        raise ValueError(f"Input JSON must be an object/dict, got: {type(obj)}")
    return obj


def demo_accept_case():
    perception = {
        "pre_scene": {"summary": "street view"},
        "post_scene": {"summary": "some roof covering loss is visible"},
        "_meta": {"source": "perception_agent"},
    }
    detection = {
        "confirmed_disaster_damage": [{"id": "chg_1", "description": "missing shingles", "location": "roof"}],
        "likely_pseudo_change": [],
        "uncertain": [],
    }
    assessment = {
        "overall_area_damage_level": "moderate",
        "overall_area_confidence": 0.72,
        "overall_area_reasoning": "Roof covering loss suggests moderate damage.",
        "per_change_assessments": [
            {
                "tier": "confirmed_disaster_damage",
                "change": {"id": "chg_1"},
                "assessment": {"predicted_damage_level": "moderate", "confidence": 0.78, "reasoning_summary": "Moderate."},
            }
        ],
        "uncertain_change_assessments": [],
    }
    return perception, detection, assessment


def demo_revise_case_misuse_uncertain():
    perception = {
        "pre_scene": {"summary": "street view"},
        "post_scene": {"summary": "visibility is limited; no obvious collapse mentioned"},
        "_meta": {"source": "perception_agent"},
    }
    detection = {
        "confirmed_disaster_damage": [{"id": "chg_1", "description": "minor debris", "location": "road"}],
        "likely_pseudo_change": [],
        "uncertain": [{"id": "chg_u1", "description": "possible collapse / structural failure", "location": "building"}],
    }
    assessment = {
        "overall_area_damage_level": "severe",
        "overall_area_confidence": 0.83,
        "overall_area_reasoning": "Severe damage inferred though evidence is unclear.",
        "per_change_assessments": [
            {
                "tier": "confirmed_disaster_damage",
                "change": {"id": "chg_1"},
                "assessment": {"predicted_damage_level": "minor", "confidence": 0.6, "reasoning_summary": "Minor."},
            }
        ],
        "uncertain_change_assessments": [
            {
                "tier": "uncertain",
                "change": {"id": "chg_u1"},
                "assessment": {"predicted_damage_level": "severe", "confidence": 0.4, "reasoning_summary": "Possible collapse."},
                "_evidence_weight": 0.3,
            }
        ],
    }
    return perception, detection, assessment


def demo_human_review_case_cross_inconsistency():
    perception = {
        "pre_scene": {"summary": "street view"},
        "post_scene": {"summary": "building appears intact; no visible damage; facade unchanged"},
        "_meta": {"source": "perception_agent"},
    }
    detection = {
        "confirmed_disaster_damage": [
            {"id": "chg_1", "description": "roof collapsed", "location": "roof"},
            {"id": "chg_2", "description": "wall collapsed", "location": "facade"},
        ],
        "likely_pseudo_change": [],
        "uncertain": [],
    }
    assessment = {
        "overall_area_damage_level": "severe",
        "overall_area_confidence": 0.9,
        "overall_area_reasoning": "Severe collapse and structural failure.",
        "per_change_assessments": [
            {
                "tier": "confirmed_disaster_damage",
                "change": {"id": "chg_1"},
                "assessment": {"predicted_damage_level": "severe", "confidence": 0.85, "reasoning_summary": "Severe."},
            },
            {
                "tier": "confirmed_disaster_damage",
                "change": {"id": "chg_2"},
                "assessment": {"predicted_damage_level": "severe", "confidence": 0.85, "reasoning_summary": "Severe."},
            },
        ],
        "uncertain_change_assessments": [],
    }
    return perception, detection, assessment


def integration_example() -> str:
    return (
        "perception_output = run_perception(...)\n"
        "detection_output = run_detection(perception_output, ...)\n"
        "assessment_output = run_assessment(detection_output, ...)\n\n"
        "critic = CriticAgent(use_rag=True)\n"
        "critic_result = critic.run(\n"
        "    perception_output=perception_output,\n"
        "    detection_output=detection_output,\n"
        "    assessment_output=assessment_output,\n"
        "    rules_summary=None,\n"
        ")\n\n"
        "if critic_result['overall_judgement'] == 'accept':\n"
        "    final_output = assessment_output\n"
        "elif critic_result['overall_judgement'] == 'revise':\n"
        "    final_output = {'assessment': assessment_output, 'critic': critic_result}\n"
        "else:  # human_review\n"
        "    final_output = {'assessment': assessment_output, 'critic': critic_result, 'need_human_review': True}\n"
    )


def main() -> None:
    _ensure_repo_root_on_path()
    # Import after sys.path fix so running as a script works on Windows.
    from CriticAgent.critic_agent import CriticAgent  # noqa: WPS433

    p = argparse.ArgumentParser(description="Critic Agent CLI / Demo")
    p.add_argument("--input_assessment", type=str, default="", help="Assessment details JSON（detailsCLIdetails）")
    p.add_argument("--input_detection", type=str, default="", help="Detection details JSON（details）")
    p.add_argument("--input_perception", type=str, default="", help="Perception details JSON（details）")
    p.add_argument("--output_critic", type=str, default="", help="Critic details JSON（details；details）")
    p.add_argument("--use_rag", type=int, default=1, help="details gov details：1/0")
    p.add_argument("--rag_top_k", type=int, default=5, help="RAG top_k")
    p.add_argument("--use_llm", type=int, default=1, help="details LLM：1/0（0detailshuman_review）")
    args = p.parse_args()

    # If user provided an assessment file, run CLI mode.
    if args.input_assessment:
        # Windows console encoding can be cp1252/gbk etc; ensure JSON can be printed.
        try:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            if hasattr(sys.stderr, "reconfigure"):
                sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

        assessment = _load_json(Path(args.input_assessment))
        detection = _load_json(Path(args.input_detection)) if args.input_detection else {}
        perception = _load_json(Path(args.input_perception)) if args.input_perception else {}

        critic = CriticAgent(use_rag=bool(int(args.use_rag)), rag_top_k=int(args.rag_top_k), use_llm=bool(int(args.use_llm)))
        res = critic.run(perception_output=perception, detection_output=detection, assessment_output=assessment, rules_summary=None)

        out_text = json.dumps(res, ensure_ascii=False, indent=2)
        if args.output_critic:
            out_path = Path(args.output_critic)
            os.makedirs(str(out_path.parent), exist_ok=True)
            out_path.write_text(out_text, encoding="utf-8")
        else:
            print(out_text)
        return

    # Otherwise: demo mode
    critic = CriticAgent(use_rag=False)
    cases = [
        ("accept", demo_accept_case()),
        ("revise", demo_revise_case_misuse_uncertain()),
        ("human_review", demo_human_review_case_cross_inconsistency()),
    ]
    for name, (pp, dd, aa) in cases:
        res = critic.run(perception_output=pp, detection_output=dd, assessment_output=aa, rules_summary=None)
        print("=" * 88)
        print(f"CASE: {name}")
        print(json.dumps(res, ensure_ascii=False, indent=2))
    print("=" * 88)
    print("Integration example (pseudo-code):")
    print(integration_example())


if __name__ == "__main__":
    main()

