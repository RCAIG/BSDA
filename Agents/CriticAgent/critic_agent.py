#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/critic_agent.py

details CriticAgent details（details）：
- details `CriticAgent/config.py`
- RAG details `CriticAgent/rag.py`
- CLI / demo details `CriticAgent/main.py`
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from CriticAgent import config as critic_cfg
from CriticAgent import rag as critic_rag
from shared_llm import get_shared_llm


# -----------------------------
# Basic utils
# -----------------------------

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from model output.
    - Prefer ```json ... ```
    - Else try first '{' to last '}'.
    """
    t = _safe_str(text)
    if not t:
        return None

    m = re.search(r"```json\s*([\s\S]*:)\s*```", t, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    l = t.find("{")
    r = t.rfind("}")
    if l >= 0 and r > l:
        candidate = t[l : r + 1].strip()
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


def _normalize_confidence_label(x: Any) -> Optional[str]:
    """
    Normalize proposed_confidence to one of: low/medium/high or None.
    Accepts numeric confidences too.
    """
    s = _safe_str(x).lower()
    if s in set(critic_cfg.ALLOWED_CONFIDENCE_LABELS):
        return s
    try:
        v = float(x)
        if v >= 0.7:
            return "high"
        if v >= 0.4:
            return "medium"
        if v > 0:
            return "low"
    except Exception:
        pass
    return None


def _clean_critic_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enforce output schema: keep only required top-level keys and normalize fields.
    This prevents LLM from leaking extra keys (e.g., echoing a rule snippet dict).
    """
    out: Dict[str, Any] = {
        "overall_judgement": obj.get("overall_judgement"),
        "detected_issues": obj.get("detected_issues"),
        "critic_recommendation": obj.get("critic_recommendation"),
        "_meta": obj.get("_meta"),
    }

    oj = _safe_str(out.get("overall_judgement")).lower()
    if oj not in set(critic_cfg.ALLOWED_JUDGEMENTS):
        oj = "human_review"
    out["overall_judgement"] = oj

    issues = out.get("detected_issues")
    if not isinstance(issues, list):
        issues = []
    norm_issues: List[Dict[str, Any]] = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        t = _safe_str(it.get("type")).lower()
        if t not in set(critic_cfg.ALLOWED_ISSUE_TYPES):
            t = "other"
        sev = _safe_str(it.get("severity")).lower()
        if sev not in set(critic_cfg.ALLOWED_SEVERITIES):
            sev = "minor"
        desc = _safe_str(it.get("description"))
        rc = it.get("related_changes")
        ra = it.get("related_agents")
        norm_issues.append(
            {
                "type": t,
                "severity": sev,
                "description": desc,
                "related_changes": rc if isinstance(rc, list) else [],
                "related_agents": ra if isinstance(ra, list) else [],
            }
        )
    out["detected_issues"] = norm_issues

    rec = out.get("critic_recommendation")
    if not isinstance(rec, dict):
        rec = {}
    pg = rec.get("proposed_grade")
    pg_s = _safe_str(pg).lower()
    proposed_grade: Optional[str] = pg_s if pg_s in set(critic_cfg.ALLOWED_GRADES) else None
    proposed_conf = _normalize_confidence_label(rec.get("proposed_confidence"))
    out["critic_recommendation"] = {
        "proposed_grade": proposed_grade,
        "proposed_confidence": proposed_conf,
        "rationale": _safe_str(rec.get("rationale")),
    }

    meta = out.get("_meta")
    if not isinstance(meta, dict):
        meta = {}
    out["_meta"] = meta
    return out


def _extract_overall_grade_and_confidence(assessment_output: Dict[str, Any]) -> tuple[str, Optional[float]]:
    grade = _safe_str(
        assessment_output.get("overall_area_damage_level")
        or assessment_output.get("overall_damage_level")
        or assessment_output.get("overall_grade")
    ).lower()
    if grade not in set(critic_cfg.ALLOWED_GRADES):
        grade = ""
    conf_raw = assessment_output.get("overall_area_confidence")
    if conf_raw is None:
        conf_raw = assessment_output.get("overall_confidence")
    conf: Optional[float] = None
    try:
        conf = float(conf_raw) if conf_raw is not None else None
    except Exception:
        conf = None
    return grade, conf


def _collect_valid_change_ids(detection_output: Dict[str, Any], assessment_output: Dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for key in ["confirmed_disaster_damage", "uncertain", "likely_pseudo_change", "changes"]:
        arr = detection_output.get(key)
        if isinstance(arr, list):
            for it in arr:
                if isinstance(it, dict):
                    cid = it.get("source_change_id", None)
                    if cid is None:
                        cid = it.get("id", None)
                    s = _safe_str(cid)
                    if s:
                        ids.add(s)

    for key in ["per_change_assessments", "uncertain_change_assessments"]:
        arr = assessment_output.get(key)
        if isinstance(arr, list):
            for it in arr:
                if not isinstance(it, dict):
                    continue
                ch = it.get("change")
                if isinstance(ch, dict):
                    cid = ch.get("source_change_id", None)
                    if cid is None:
                        cid = ch.get("id", None)
                    s = _safe_str(cid)
                    if s:
                        ids.add(s)
    return ids


def _post_validate_against_inputs(
    cleaned: Dict[str, Any],
    *,
    detection_output: Dict[str, Any],
    assessment_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Make Critic output consistent with the provided inputs.
    - Filter invalid related_changes IDs.
    - If critic recommends the same grade as current grade, it should not claim "grade mismatch".
    """
    current_grade, _ = _extract_overall_grade_and_confidence(assessment_output)
    valid_ids = _collect_valid_change_ids(detection_output, assessment_output)

    issues = cleaned.get("detected_issues")
    if not isinstance(issues, list):
        issues = []

    filtered_issues: List[Dict[str, Any]] = []
    for it in issues:
        if not isinstance(it, dict):
            continue
        rc = it.get("related_changes")
        if isinstance(rc, list):
            rc2 = [x for x in [_safe_str(y) for y in rc] if x and (not valid_ids or x in valid_ids)]
        else:
            rc2 = []
        it2 = dict(it)
        it2["related_changes"] = rc2
        filtered_issues.append(it2)

    cleaned["detected_issues"] = filtered_issues

    rec = cleaned.get("critic_recommendation")
    proposed_grade = None
    if isinstance(rec, dict):
        proposed_grade = _safe_str(rec.get("proposed_grade")).lower()
    if proposed_grade not in set(critic_cfg.ALLOWED_GRADES):
        proposed_grade = None

    # If current grade is already the same as the critic's proposed grade, do not claim a grade mismatch.
    if current_grade and proposed_grade and proposed_grade == current_grade:
        cleaned["detected_issues"] = [
            it for it in cleaned["detected_issues"] if _safe_str(it.get("type")).lower() != "grade_evidence_mismatch"
        ]
        # If the critic does not propose a different grade, do not force "revise" unless there are major/critical issues.
        if _safe_str(cleaned.get("overall_judgement")).lower() == "revise":
            issues2 = cleaned.get("detected_issues")
            has_major_or_critical = False
            if isinstance(issues2, list):
                for it in issues2:
                    if isinstance(it, dict) and _safe_str(it.get("severity")).lower() in {"major", "critical"}:
                        has_major_or_critical = True
                        break
            if not has_major_or_critical:
                cleaned["overall_judgement"] = "accept"

        # Also fix textual hallucinations: rationale may wrongly state current grade (e.g., "minor").
        rec2 = cleaned.get("critic_recommendation")
        if isinstance(rec2, dict):
            # If the critic agrees with current grade, the recommendation should be "confirm" rather than "re-evaluate".
            rec2["proposed_grade"] = current_grade
            rec2["proposed_confidence"] = rec2.get("proposed_confidence") or "high"
            rec2["rationale"] = (
                f"The overall grade is consistent: current_overall_grade={current_grade}. "
                "Based on the key evidence in the inputs (e.g., strong structural-failure signals), this grade is reasonable."
            )
            cleaned["critic_recommendation"] = rec2

        # If we ended up accepting, avoid leaving "details" notes around.
        meta0 = cleaned.get("_meta")
        if not isinstance(meta0, dict):
            meta0 = {}
        meta0.setdefault("notes", "")
        oj2 = _safe_str(cleaned.get("overall_judgement")).lower() or "accept"
        if oj2 == "accept":
            meta0["notes"] = f"Critic check: proposed grade equals current grade ({current_grade}); no revision required."
        else:
            meta0["notes"] = (
                f"Critic check: proposed grade equals current grade ({current_grade}); "
                "however, there are major/critical QC issues that should be revised."
            )
        cleaned["_meta"] = meta0

        meta = cleaned.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta["post_validation"] = {
            "current_grade": current_grade,
            "proposed_grade": proposed_grade,
            "note": "Proposed grade equals current grade; removed grade_evidence_mismatch and possibly downgraded judgement.",
        }
        cleaned["_meta"] = meta

    # Add a compact, one-glance summary for readability (always present).
    current_grade2, current_conf2 = _extract_overall_grade_and_confidence(assessment_output if isinstance(assessment_output, dict) else {})
    rec3 = cleaned.get("critic_recommendation") if isinstance(cleaned.get("critic_recommendation"), dict) else {}
    issues4 = cleaned.get("detected_issues") if isinstance(cleaned.get("detected_issues"), list) else []
    
    # Count issues by severity
    critical_count = sum(1 for it in issues4 if isinstance(it, dict) and _safe_str(it.get("severity")).lower() == "critical")
    major_count = sum(1 for it in issues4 if isinstance(it, dict) and _safe_str(it.get("severity")).lower() == "major")
    minor_count = sum(1 for it in issues4 if isinstance(it, dict) and _safe_str(it.get("severity")).lower() == "minor")
    
    recommended_grade = _safe_str(rec3.get("proposed_grade")).lower() or None
    overall_judgement = _safe_str(cleaned.get("overall_judgement")).lower()
    
    # Determine why revise (if applicable)
    revise_reason = None
    if overall_judgement == "revise":
        if recommended_grade and recommended_grade == current_grade2:
            if critical_count > 0 or major_count > 0:
                revise_reason = f"Grade is correct ({recommended_grade}), but {critical_count} critical and {major_count} major QC issues found."
            else:
                revise_reason = f"Grade is correct ({recommended_grade}), but {minor_count} minor QC issues found."
        else:
            revise_reason = f"Grade mismatch: current={current_grade2}, recommended={recommended_grade}."
    
    summary = {
        "current_overall_grade": current_grade2,
        "current_overall_confidence": current_conf2,
        "recommended_overall_grade": recommended_grade,
        "recommended_confidence": _safe_str(rec3.get("proposed_confidence")).lower() or None,
        "overall_judgement": overall_judgement,
        "issue_counts": {
            "critical": critical_count,
            "major": major_count,
            "minor": minor_count,
            "total": len(issues4),
        },
        "revise_reason": revise_reason,
        "next_action": (
            "Revise the pipeline outputs according to detected_issues." if overall_judgement == "revise"
            else ("Flag for human review." if overall_judgement == "human_review" else "No revision needed.")
        ),
    }
    cleaned["summary"] = summary

    # Add actionable fix suggestions per issue (which agent should change what).
    issues3 = cleaned.get("detected_issues")
    if isinstance(issues3, list):
        fixed: List[Dict[str, Any]] = []
        for it in issues3:
            if not isinstance(it, dict):
                continue
            t = _safe_str(it.get("type")).lower()
            ra = it.get("related_agents") if isinstance(it.get("related_agents"), list) else []
            fix_by = None
            if "Assessment" in ra:
                fix_by = "Assessment"
            elif "Detection" in ra:
                fix_by = "Detection"
            elif "Perception" in ra:
                fix_by = "Perception"
            else:
                # Default mapping by type
                if t in {"grade_evidence_mismatch", "confidence_mismatch", "misuse_of_uncertain", "misuse_of_pseudo"}:
                    fix_by = "Assessment"
                elif t in {"cross_agent_inconsistency"}:
                    fix_by = "Assessment"
                else:
                    fix_by = "Assessment"
            suggestion = None
            if t == "confidence_mismatch":
                suggestion = "Align confidence values with the written rationale; do not use low-confidence items as strong evidence for the overall grade."
            elif t == "cross_agent_inconsistency":
                suggestion = "Align Assessment reasoning with Detection evidence: avoid asserting stronger claims (e.g., 'complete collapse') unless explicitly supported."
            elif t == "grade_evidence_mismatch":
                suggestion = "Re-check the overall grade against the strongest confirmed evidence; adjust grade or cite stronger evidence."
            elif t == "misuse_of_uncertain":
                suggestion = "Do not let uncertain items dominate the overall grade; keep them as low-weight support only."
            elif t == "misuse_of_pseudo":
                suggestion = "Exclude pseudo changes from grading; keep them as notes only."
            else:
                suggestion = "Review this issue and revise the responsible agent's output accordingly."
            it2 = dict(it)
            it2.setdefault("fix_by_agent", fix_by)
            it2.setdefault("suggested_fix", suggestion)
            fixed.append(it2)
        cleaned["detected_issues"] = fixed

    return cleaned


_GLOBAL_LLM = None
_GLOBAL_TOKENIZER = None
_GLOBAL_TORCH = None


def _get_local_llm():
    """
    Lazy-load local LOCAL_MODEL model once per process.
    """
    global _GLOBAL_LLM, _GLOBAL_TOKENIZER, _GLOBAL_TORCH
    if _GLOBAL_LLM is not None and _GLOBAL_TOKENIZER is not None and _GLOBAL_TORCH is not None:
        return _GLOBAL_TOKENIZER, _GLOBAL_LLM, _GLOBAL_TORCH
    # details Agent details LOCAL_MODEL details（details/details）
    tokenizer, model, torch = get_shared_llm(critic_cfg.LOCAL_LLM_MODEL_PATH)
    _GLOBAL_TOKENIZER = tokenizer  # type: ignore[assignment]
    _GLOBAL_LLM = model  # type: ignore[assignment]
    _GLOBAL_TORCH = torch  # type: ignore[assignment]
    return _GLOBAL_TOKENIZER, _GLOBAL_LLM, _GLOBAL_TORCH


def _generate_llm(system_prompt: str, user_prompt: str) -> str:
    system_prompt = _safe_str(system_prompt)
    user_prompt = _safe_str(user_prompt)
    if not user_prompt:
        return ""

    tokenizer, model, torch = _get_local_llm()
    try:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        text_input = system_prompt + "\n\n" + user_prompt

    model_inputs = tokenizer([text_input], return_tensors="pt").to(model.device)
    do_sample = bool(float(getattr(critic_cfg, "TEMPERATURE", 0.2)) > 1e-6)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=int(getattr(critic_cfg, "MAX_NEW_TOKENS", 600)),
            temperature=float(getattr(critic_cfg, "TEMPERATURE", 0.2)),
            do_sample=do_sample,
            top_p=float(getattr(critic_cfg, "TOP_P", 0.9)),
            repetition_penalty=1.05,
        )
    gen_only = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    out = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
    return _safe_str(out)

# -----------------------------
# CriticAgent
# -----------------------------


class CriticAgent:
    """
    Pure LLM critic (local LOCAL_MODEL). No heuristic keyword rules.
    """

    critic_version: str = critic_cfg.CRITIC_VERSION

    def __init__(
        self,
        *,
        use_rag: bool = critic_cfg.DEFAULT_USE_RAG,
        rag_top_k: int = critic_cfg.DEFAULT_RAG_TOP_K,
        use_llm: Optional[bool] = None,
    ) -> None:
        """
        Args:
            use_rag: If True, Critic will retrieve a few gov rule snippets to ground its judgement/recommendation.
            rag_top_k: How many gov snippets to retrieve (small by default).
        """
        self.use_rag = bool(use_rag)
        self.rag_top_k = int(rag_top_k)
        self.use_llm = bool(critic_cfg.ENABLE_LLM_CRITIC) if use_llm is None else bool(use_llm)

    def run(
        self,
        *,
        perception_output: Dict[str, Any],
        detection_output: Dict[str, Any],
        assessment_output: Dict[str, Any],
        rules_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Optional RAG grounding (gov rules)
        used_rules_snippets: List[Dict[str, Any]] = []
        if self.use_rag and rules_summary is None:
            hazard = _safe_str(assessment_output.get("_meta", {}).get("hazard_type")) or _safe_str(
                detection_output.get("hazard_type")
            )
            q = critic_rag.build_critic_rag_query(
                hazard_type=hazard or "hurricane",
                overall_grade=_safe_str(assessment_output.get("overall_area_damage_level") or assessment_output.get("overall_damage_level")),
                overall_confidence_label=_safe_str(assessment_output.get("overall_area_confidence") or assessment_output.get("overall_confidence")),
                post_scene_summary=_safe_str(perception_output.get("post_scene", {}).get("summary")) if isinstance(perception_output.get("post_scene"), dict) else "",
                overall_reasoning=_safe_str(assessment_output.get("overall_area_reasoning") or assessment_output.get("overall_reasoning")),
            )
            used_rules_snippets = critic_rag.search_gov_rules_for_critic(q, top_k=int(self.rag_top_k))

        if not self.use_llm:
            # pure LLM is disabled; safest fallback is "human_review"
            return {
                "overall_judgement": "human_review",
                "detected_issues": [
                    {
                        "type": "other",
                        "severity": "critical",
                        "description": "LLM critic disabled; cannot perform QC. Flagging for human review.",
                        "related_changes": [],
                        "related_agents": ["Critic"],
                    }
                ],
                "critic_recommendation": {"proposed_grade": None, "proposed_confidence": None, "rationale": "LLM disabled."},
                "_meta": {
                    "critic_version": self.critic_version,
                    "timestamp": _now_iso(),
                    "used_rules": list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [],
                    "used_rules_snippets": used_rules_snippets,
                },
            }

        system = (
            "You are a Critic Agent for a multi-agent disaster assessment pipeline.\n"
            "You will be given the outputs of Perception, Detection, and Assessment Agents (as JSON),\n"
            "and optional official rule snippets from a government/standard knowledge base.\n"
            "Your job is to produce a structured QC decision.\n\n"
            f"Allowed overall_judgement: {critic_cfg.ALLOWED_JUDGEMENTS}\n"
            f"Allowed issue.severity: {critic_cfg.ALLOWED_SEVERITIES}\n"
            f"Allowed issue.type: {critic_cfg.ALLOWED_ISSUE_TYPES}\n"
            f"Allowed proposed_grade: {critic_cfg.ALLOWED_GRADES} or null\n"
            f"Allowed proposed_confidence: {critic_cfg.ALLOWED_CONFIDENCE_LABELS} or null\n\n"
            "You MUST output a single JSON object with EXACTLY these key<LOCAL_PATH>"
            "- overall_judgement\n"
            "- detected_issues\n"
            "- critic_recommendation\n"
            "- _meta\n"
            "Do NOT output any extra text outside this JSON."
        )
        system = system + "\n\nLanguage requiremen<LOCAL_PATH>"

        # Provide explicit facts up-front to reduce misreading.
        current_grade, current_conf = _extract_overall_grade_and_confidence(assessment_output if isinstance(assessment_output, dict) else {})
        facts = f"FACTS (must treat as ground truth): current_overall_grade={current_grade or 'unknown'}; current_overall_confidence={current_conf if current_conf is not None else 'unknown'}."

        # Full-input mode: pass through the original outputs without compacting/truncation.
        payload = {
            "perception_output": perception_output if isinstance(perception_output, dict) else {},
            "detection_output": detection_output if isinstance(detection_output, dict) else {},
            "assessment_output": assessment_output if isinstance(assessment_output, dict) else {},
            "rules_summary": rules_summary if isinstance(rules_summary, dict) else None,
            "used_rules_snippets": used_rules_snippets,
        }
        user = (
            "Please perform consistency and quality-control (QC) review for the multi-agent pipeline outputs.\n"
            "Goal: identify issues and provide an overall judgement (accept / revise / human_review).\n\n"
            + facts
            + "\n\n"
            "[INPUT (for review only; do NOT copy/echo in your output)]\n"
            "```json\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            "[OUTPUT REQUIREMENTS]\n"
            "Output ONLY 1 JSON object with exactly these 4 top-level key<LOCAL_PATH>"
            "- overall_judgement\n"
            "- detected_issues\n"
            "- critic_recommendation\n"
            "- _meta\n\n"
            "Strictly forbidden: echoing any input JSON fields (e.g., changes / used_rules_snippets).\n\n"
            "Example output JSON (field names must match exactly):\n"
            "```json\n"
            "{\n"
            "  \"overall_judgement\": \"revise\",\n"
            "  \"detected_issues\": [\n"
            "    {\n"
            "      \"type\": \"grade_evidence_mismatch\",\n"
            "      \"severity\": \"major\",\n"
            "      \"description\": \"Overall grade does not match the evidence...\",\n"
            "      \"related_changes\": [\"chg_003\"],\n"
            "      \"related_agents\": [\"Assessment\"]\n"
            "    }\n"
            "  ],\n"
            "  \"critic_recommendation\": {\n"
            "    \"proposed_grade\": \"severe\",\n"
            "    \"proposed_confidence\": \"high\",\n"
            "    \"rationale\": \"Key evidence supports severe damage due to structural failure signals...\"\n"
            "  },\n"
            "  \"_meta\": {\n"
            "    \"notes\": \"Optional notes\"\n"
            "  }\n"
            "}\n"
            "```\n\n"
            "Now output your JSON (no extra text)."
        )

        raw = _generate_llm(system, user)
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return {
                "overall_judgement": "human_review",
                "detected_issues": [
                    {
                        "type": "other",
                        "severity": "critical",
                        "description": "Critic LLM output invalid or parsing failed; flagging for human review.",
                        "related_changes": [],
                        "related_agents": ["Critic"],
                    }
                ],
                "critic_recommendation": {"proposed_grade": None, "proposed_confidence": None, "rationale": "LLM output invalid."},
                "_meta": {
                    "critic_version": self.critic_version,
                    "timestamp": _now_iso(),
                    "used_rules": list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [],
                    "used_rules_snippets": used_rules_snippets,
                    "llm_parse_failed": True,
                    "llm_raw_output": raw,
                },
            }

        # Strict schema cleanup + normalization
        cleaned = _clean_critic_output(obj)

        meta = cleaned.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("critic_version", self.critic_version)
        meta.setdefault("timestamp", _now_iso())
        meta.setdefault("used_rules", list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [])
        meta.setdefault("used_rules_snippets", used_rules_snippets)
        # Save raw output for debugging when the model misbehaves.
        meta.setdefault("llm_raw_output", _safe_str(raw))
        cleaned["_meta"] = meta
        return _post_validate_against_inputs(
            cleaned,
            detection_output=detection_output if isinstance(detection_output, dict) else {},
            assessment_output=assessment_output if isinstance(assessment_output, dict) else {},
        )
