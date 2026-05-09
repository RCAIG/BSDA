from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    return cur


def ensure_repo_root_on_path(hint_file: Path) -> Path:
    repo_root = find_repo_root(hint_file)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def load_text_or_json(path: Path) -> Tuple[str, Any]:
    """
    Return (raw_text, parsed_obj_or_fallback_dict).
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    try:
        obj = json.loads(raw)
        return raw, obj
    except Exception:
        return raw, {"raw_text": raw}


def extract_judgement(critic_output: Dict[str, Any]) -> str:
    """
    Return lowercased judgement: accept/revise/flag_for_human, default "accept".
    """
    j = safe_str(critic_output.get("summary", {}).get("overall_judgement") or critic_output.get("overall_judgement")).lower()
    return j or "accept"


def issues_related_agents(critic_output: Dict[str, Any]) -> List[str]:
    """
    Best-effort: extract related_agents values from Critic output.
    """
    out: List[str] = []
    issues = critic_output.get("detected_issues", [])
    if not isinstance(issues, list):
        return out
    for it in issues:
        if not isinstance(it, dict):
            continue
        ra = it.get("related_agents")
        if isinstance(ra, list):
            out.extend([safe_str(x) for x in ra if safe_str(x)])
        # Backward compat (some versions might output fix_by_agent)
        fba = safe_str(it.get("fix_by_agent"))
        if fba:
            out.append(fba)
        ta = safe_str(it.get("target_agent"))
        if ta:
            out.append(ta)
    # normalize
    return [x.strip() for x in out if x.strip()]


def apply_critic_feedback_to_detection(
    detection_output: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    revised = dict(detection_output)
    meta = revised.setdefault("_meta", {})
    history = meta.setdefault("critic_feedback_history", [])
    if isinstance(history, list):
        history.append({"summary": critic_output.get("summary", {}), "issues": critic_output.get("detected_issues", [])})
    meta["critic_detection_issues"] = [
        it
        for it in (critic_output.get("detected_issues", []) or [])
        if isinstance(it, dict)
        and (
            safe_str(it.get("fix_by_agent")) == "Detection"
            or "Detection" in (it.get("related_agents") or [])
            or safe_str(it.get("target_agent")) == "Detection"
        )
    ]
    return revised


def apply_critic_feedback_to_assessment(
    assessment_output: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    revised = dict(assessment_output)
    meta = revised.setdefault("_meta", {})
    history = meta.setdefault("critic_feedback_history", [])
    if isinstance(history, list):
        history.append(
            {
                "summary": critic_output.get("summary", {}),
                "recommendation": critic_output.get("critic_recommendation", {}),
                "issues": critic_output.get("detected_issues", []),
            }
        )
    meta["critic_last_overall_judgement"] = critic_output.get("summary", {}).get("overall_judgement") or critic_output.get("overall_judgement")
    meta["critic_last_revise_reason"] = critic_output.get("summary", {}).get("revise_reason")
    rec = critic_output.get("critic_recommendation", {}) if isinstance(critic_output.get("critic_recommendation"), dict) else {}
    meta["critic_last_recommended_grade"] = rec.get("proposed_grade")
    meta["critic_last_recommended_confidence"] = rec.get("proposed_confidence")
    meta["critic_assessment_issues"] = [
        it
        for it in (critic_output.get("detected_issues", []) or [])
        if isinstance(it, dict)
        and (
            safe_str(it.get("fix_by_agent")) == "Assessment"
            or "Assessment" in (it.get("related_agents") or [])
            or safe_str(it.get("target_agent")) == "Assessment"
        )
    ]
    return revised


def build_revision_prompt_suffix(critic_output: Dict[str, Any]) -> str:
    """
    A compact suffix to append to VLM prompt so the model focuses on fixing omissions.
    """
    summary = critic_output.get("summary", {}) if isinstance(critic_output.get("summary"), dict) else {}
    revise_reason = safe_str(summary.get("revise_reason"))
    issues = critic_output.get("detected_issues", [])
    bullets: List[str] = []
    if isinstance(issues, list):
        for it in issues[:8]:
            if not isinstance(it, dict):
                continue
            desc = safe_str(it.get("description"))
            sev = safe_str(it.get("severity"))
            t = safe_str(it.get("type"))
            if desc:
                bullets.append(f"- ({sev or 'unknown'}|{t or 'issue'}) {desc}")
    extra = "\n".join(bullets).strip()
    parts = [
        "",
        "REVISION NOTE (Critic feedback):",
        revise_reason or "",
        extra or "",
        "Please correct omissions/inaccuracies and output ONLY a single JSON object following the same schema.",
    ]
    return "\n".join([p for p in parts if p]).strip()


