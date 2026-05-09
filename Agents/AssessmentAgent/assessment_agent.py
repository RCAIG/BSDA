#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assessment Agent

Input: DetectionAgent output dict (JSON-loaded)
Output: per-change assessments + overall scene/area-level assessment (for the street-view region)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from AssessmentAgent import config as cfg
from .rag import search_gov_rules, search_history_cases
from shared_llm import get_shared_llm


SYSTEM_PROMPT = (
    "You are an Assessment Agent that determines the damage severity of components after a disaster.\n"
    "\n"
    "Input<LOCAL_PATH>"
    "- One detected change (component, change summary, and whether it is likely disaster-related).\n"
    "- Several official rule snippets from a government/standard knowledge base that describe damage states and severity categories.\n"
    "- Several historical case snippets, each with a known damage level label, describing similar observed damage.\n"
    "\n"
    "Your goal<LOCAL_PATH>"
    "1. Use the rule snippets to understand how each damage level is defined (\"minor\", \"moderate\", \"severe\").\n"
    "2. Use the historical cases as empirical calibration (how similar descriptions were labeled in the past).\n"
    "3. Combine the detected change, rule definitions, and historical examples to assign a single damage level to this change.\n"
    "4. Explain clearly why this level was chosen, citing the most relevant rule phrases and the most similar historical patterns.\n"
    "\n"
    "Allowed damage level<LOCAL_PATH>"
    "\"minor\", \"moderate\", \"severe\".\n"
    "\n"
    "Borderline/insufficient-information rul<LOCAL_PATH>"
    "- If evidence is clearly insufficient or the case is borderline, you must still choose one of the allowed levels,\n"
    "  but set LOW confidence (e.g., 0.1–0.3) and explicitly explain the uncertainty or missing evidence.\n"
    "\n"
    "Language requiremen<LOCAL_PATH>"
    "- Write reasoning_summary and notes in English.\n"
    "\n"
    "You MUST respond with a single JSON object with EXACTLY these field<LOCAL_PATH>"
    "- predicted_damage_level: one of [\"minor\", \"moderate\", \"severe\"]\n"
    "- confidence: a float between 0.05 and 0.99 (never output 0.0)\n"
    "- reasoning_summary: a concise natural language explanation (3–5 sentences)\n"
    "- notes: optional extra remarks (or an empty string if none)\n"
    "\n"
    "Do NOT include any extra text outside this JSON object."
)


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


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


def _normalize_level(level: str) -> str:
    lv = _safe_str(level).lower()
    if not lv:
        return ""
    aliases = {
        "mild": "minor",
        "slight": "minor",
        "light": "minor",
        "low": "minor",
        "minor_damage": "minor",
        "moderate_damage": "moderate",
        "medium": "moderate",
        "major": "severe",
        "heavy": "severe",
        "severe_damage": "severe",
    }
    return aliases.get(lv, lv)


def _is_uncertain_change(change: Dict[str, Any]) -> bool:
    """
    Heuristic: some items may appear under confirmed_disaster_damage in detection output
    but are actually uncertain/possible. We re-route them to the uncertain tier so that
    they don't dominate overall_level/overall_conf.
    """
    why_uncertain = _safe_str(change.get("why_uncertain"))
    if why_uncertain:
        return True
    dr = _safe_str(change.get("disaster_related")).lower()
    if dr in {"possible", "uncertain", "maybe"}:
        return True
    return False


def _rule_based_override(change: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    High-precision overrides for obvious cases (e.g., complete collapse -> severe).
    Returns an assessment dict compatible with the normal output schema, or None.
    """
    text = " ".join(
        [
            _safe_str(change.get("damage_type")),
            _safe_str(change.get("description")),
            _safe_str(change.get("change_summary")),
            _safe_str(change.get("change_description")),
            _safe_str(change.get("reason")),
        ]
    ).lower()

    # Collapse / structural failure -> severe (strong evidence).
    # Keep multilingual keywords for recall, but ALWAYS output details.
    collapse_kw = [
        "details",
        "details",
        "details",
        "details",
        "collapsed",
        "collapse",
        "structural failure",
        "rubble",
        "debris pile",
    ]
    if any(k.lower() in text for k in collapse_kw):
        return {
            "predicted_damage_level": "severe",
            "confidence": 0.85,
            "reasoning_summary": "The description indicates structural collapse/structural failure (a strong structural-failure signal). By damage grading guidelines, this should be rated as severe.",
            "notes": "Rule override: structural collapse is strong evidence. The LLM is optional only for additional explanation.",
            "_rule_override": True,
        }
    return None


_GLOBAL_AGENT: Optional["AssessmentAgent"] = None


def get_assessment_agent() -> "AssessmentAgent":
    """
    Lazy-load a global AssessmentAgent (avoids re-loading local LOCAL_MODEL every call).
    """
    global _GLOBAL_AGENT
    if _GLOBAL_AGENT is None:
        _GLOBAL_AGENT = AssessmentAgent(model_path=cfg.LOCAL_LLM_MODEL_PATH)
    return _GLOBAL_AGENT


def _allowed_output_levels() -> List[str]:
    return list(cfg.DAMAGE_LEVELS)


def _level_rank(level: str) -> int:
    order = ["minor", "moderate", "severe"]
    lv = _normalize_level(level)
    if lv in order:
        return order.index(lv)
    return -1


def _build_query_text(change: Dict[str, Any], *, hazard_type: str = "") -> str:
    hz = _safe_str(change.get("hazard_type")) or _safe_str(hazard_type)
    component = (
        _safe_str(change.get("component"))
        or _safe_str(change.get("component_name"))
        or _safe_str(change.get("affected_feature"))
    )
    location = _safe_str(change.get("location"))
    change_summary = (
        _safe_str(change.get("change_summary"))
        or _safe_str(change.get("change_desc"))
        or _safe_str(change.get("change_description"))
        or _safe_str(change.get("reason"))
    )
    after_state = _safe_str(change.get("post_state")) or _safe_str(change.get("after"))

    parts: List[str] = []
    if hz:
        parts.append(f"Hazard type: {hz}.")
    if component:
        parts.append(f"Component: {component}.")
    if location:
        parts.append(f"Location: {location}.")
    if change_summary:
        parts.append(f"Detected change summary (pre vs post): {change_summary}.")
    if after_state:
        parts.append(f"Post-disaster state details: {after_state}.")
    parts.append("This is a description of changes observed between pre-disaster and post-disaster conditions.")
    parts.append(
        "Retrieval goal: find guidance, criteria, thresholds, and examples that help classify the observed damage severity as minor vs moderate vs severe."
    )
    return " ".join(parts).strip()


def _build_user_prompt(
    *,
    change: Dict[str, Any],
    gov_rules: List[Dict[str, Any]],
    history_cases: List[Dict[str, Any]],
    detection_tier: str = "confirmed_disaster_damage",
) -> str:
    # Keep prompts compact to reduce token length (critical for CPU memory safety).
    change_json = json.dumps(change, ensure_ascii=False, separators=(",", ":"))
    gov_json = json.dumps(gov_rules, ensure_ascii=False, separators=(",", ":"))
    hist_json = json.dumps(history_cases, ensure_ascii=False, separators=(",", ":"))

    return (
        "You are given one detected building change and supporting context.\n\n"
        f"[Detection tier]\n{_safe_str(detection_tier)}\n\n"
        "Interpretation rule<LOCAL_PATH>"
        "- confirmed_disaster_damage: strong evidence; can support moderate/severe if matched to official criteria.\n"
        "- uncertain: weak evidence; it may support a decision, but MUST NOT alone justify a severe grade.\n\n"
        "[Official rule snippets from government/standard knowledge base]\n\n"
        "```json\n"
        f"{gov_json}\n"
        "```\n\n"
        "[Historical case snippets with known damage levels]\n\n"
        "```json\n"
        f"{hist_json}\n"
        "```\n\n"
        # Put the detected change near the end so that left-truncation keeps it (critical under token caps).
        "[Detected change]\n\n"
        "```json\n"
        f"{change_json}\n"
        "```\n\n"
        "Tas<LOCAL_PATH>"
        "1. Carefully read the detected change.\n"
        "2. Use the rule snippets to map the observed damage to one of the allowed severity levels.\n"
        "3. Use the historical cases as additional calibration.\n"
        "4. If evidence suggests limited impact or information is insufficient, you must still choose a level but use LOW confidence (0.1–0.3) and explain the uncertainty.\n\n"
        "Return ONLY a JSON object with the structur<LOCAL_PATH>"
        "{\n"
        "  \"predicted_damage_level\": \"...\",\n"
        "  \"confidence\": 0.1,\n"
        "  \"reasoning_summary\": \"...\",\n"
        "  \"notes\": \"...\"\n"
        "}\n\n"
        "Do not add any other text."
    )


@dataclass
class AssessmentAgent:
    model_path: str = cfg.LOCAL_LLM_MODEL_PATH
    max_new_tokens: int = cfg.MAX_NEW_TOKENS_PER_CHANGE
    temperature: float = cfg.TEMPERATURE
    top_p: float = cfg.TOP_P

    def __post_init__(self) -> None:
        # details LOCAL_MODEL details（details）
        tokenizer, model, torch = get_shared_llm(self.model_path)
        self._torch = torch  # type: ignore[assignment]
        self._tokenizer = tokenizer  # type: ignore[assignment]
        self._model = model  # type: ignore[assignment]

    def generate_json_assessment(self, *, system_prompt: str, user_prompt: str) -> str:
        system_prompt = _safe_str(system_prompt)
        user_prompt = _safe_str(user_prompt)
        if not user_prompt:
            return ""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text_input = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text_input = system_prompt + "\n\n" + user_prompt

        model_inputs = self._tokenizer([text_input], return_tensors="pt").to(self._model.device)
        do_sample = bool(float(self.temperature) > 1e-6)
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=int(self.max_new_tokens),
                temperature=float(self.temperature),
                do_sample=do_sample,
                top_p=float(self.top_p),
                repetition_penalty=1.05,
            )

        gen_only = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        out = self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
        return _safe_str(out)


def _coerce_confidence(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v != v:
        return 0.0
    if v < 0:
        return 0.0
    if v > 1:
        return 1.0
    # Avoid meaningless 0.0 confidences: treat as "very low but non-zero"
    if v == 0.0:
        return 0.05
    return v


def _default_invalid_response() -> Dict[str, Any]:
    return {
        "predicted_damage_level": "minor",
        "confidence": 0.05,
        "reasoning_summary": "Model output was invalid or parsing failed; falling back to 'minor' with very low confidence.",
        "notes": "",
    }


def assess_one_change(
    agent: AssessmentAgent,
    change: Dict[str, Any],
    *,
    hazard_type: str = "",
    gov_top_k: int = cfg.DEFAULT_GOV_TOP_K,
    history_top_k: int = cfg.DEFAULT_HISTORY_TOP_K,
    detection_tier: str = "confirmed_disaster_damage",
) -> Dict[str, Any]:
    # High-precision rule-based overrides first (improves consistency and avoids obvious misgrades).
    override = _rule_based_override(change)
    if isinstance(override, dict):
        out = {
            "predicted_damage_level": _normalize_level(override.get("predicted_damage_level", "")),
            "confidence": _coerce_confidence(override.get("confidence", 0.0)),
            "reasoning_summary": _safe_str(override.get("reasoning_summary", "")),
            "notes": _safe_str(override.get("notes", "")),
        }
        out["_rag"] = {"gov_rules_count": 0, "history_cases_count": 0}
        out["_query_text"] = _build_query_text(change, hazard_type=hazard_type)
        out["_raw_model_output"] = ""
        out["_rule_override"] = True
        return out

    query_text = _build_query_text(change, hazard_type=hazard_type)
    try:
        gov_rules = search_gov_rules(query_text, top_k=int(gov_top_k)) or []
    except Exception:
        gov_rules = []
    try:
        history_cases = search_history_cases(query_text, top_k=int(history_top_k)) or []
    except Exception:
        history_cases = []

    user_prompt = _build_user_prompt(
        change=change,
        gov_rules=gov_rules,
        history_cases=history_cases,
        detection_tier=_safe_str(detection_tier) or "confirmed_disaster_damage",
    )
    raw = agent.generate_json_assessment(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    obj = _extract_json_object(raw)
    if not isinstance(obj, dict):
        out = _default_invalid_response()
    else:
        lvl = _normalize_level(obj.get("predicted_damage_level", ""))
        conf = _coerce_confidence(obj.get("confidence", 0.0))
        reasoning = _safe_str(obj.get("reasoning_summary", ""))
        notes = _safe_str(obj.get("notes", ""))

        allowed = set(_allowed_output_levels())
        if lvl not in allowed:
            out = _default_invalid_response()
        else:
            out = {
                "predicted_damage_level": lvl,
                "confidence": conf,
                "reasoning_summary": reasoning,
                "notes": notes,
            }

    out["_rag"] = {"gov_rules_count": len(gov_rules), "history_cases_count": len(history_cases)}
    out["_query_text"] = query_text
    out["_raw_model_output"] = raw
    return out


def _extract_changes_from_detection(detection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract the list of changes that should be assessed.

    IMPORTANT (updated for your new 3-tier design):
    - Assessment must mainly rely on confirmed_disaster_damage for overall conclusions.
    - uncertain items can be assessed as "weighted reference" but MUST NOT be the sole basis
      for calling a case severe; if most items are uncertain, lower confidence and say so.

    Supported inputs:
    - detection_result["confirmed_disaster_damage"] (preferred)
    - Legacy: detection_result["verified_hurricane_damages"] / ["verified_damages"] / ["verified_changes"]
    """

    confirmed = detection_result.get("confirmed_disaster_damage")
    if isinstance(confirmed, list):
        return [it for it in confirmed if isinstance(it, dict)]

    for key in ["verified_hurricane_damages", "verified_damages", "verified_changes"]:
        v = detection_result.get(key)
        if isinstance(v, list):
            return [it for it in v if isinstance(it, dict)]

    v = detection_result.get("changes")
    if isinstance(v, list):
        return [it for it in v if isinstance(it, dict)]

    return []


def _extract_uncertain_from_detection(detection_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    v = detection_result.get("uncertain")
    if isinstance(v, list):
        return [it for it in v if isinstance(it, dict)]
    return []


def _compute_overall(per_change: List[Dict[str, Any]]) -> Tuple[str, float, str]:
    levels = [_normalize_level(x.get("predicted_damage_level", "")) for x in per_change]
    known = [lv for lv in levels if lv in {"minor", "moderate", "severe"}]
    overall_level = max(known, key=_level_rank) if known else "minor"

    # Confidence should reflect the evidence that determines the overall_level,
    # rather than a simple average across all items (which can be dragged down by uncertain/minor items).
    confs: List[float] = []
    for x in per_change:
        lv = _normalize_level(x.get("predicted_damage_level", ""))
        if lv not in {"minor", "moderate", "severe"}:
            continue
        if lv == overall_level:
            confs.append(_coerce_confidence(x.get("confidence", 0.0)))
    # Fallback: if no item matches overall_level (shouldn't happen), average all valid ones.
    if not confs:
        for x in per_change:
            lv = _normalize_level(x.get("predicted_damage_level", ""))
            if lv in {"minor", "moderate", "severe"}:
                confs.append(_coerce_confidence(x.get("confidence", 0.0)))
    overall_conf = sum(confs) / len(confs) if confs else 0.0

    ranked = sorted(per_change, key=lambda d: _level_rank(d.get("predicted_damage_level", "")), reverse=True)
    key_summaries: List[str] = []
    for item in ranked:
        lv = _normalize_level(item.get("predicted_damage_level", ""))
        if lv not in {"minor", "moderate", "severe"}:
            continue
        rs = _safe_str(item.get("reasoning_summary", ""))
        if rs:
            key_summaries.append(f"[{lv}] {rs}")
        if len(key_summaries) >= 3:
            break
    overall_reasoning = "\n".join(key_summaries).strip()
    return overall_level, float(overall_conf), overall_reasoning


def run_assessment(
    detection_result: Dict[str, Any],
    *,
    agent: Optional[AssessmentAgent] = None,
    hazard_type: Optional[str] = None,
    gov_top_k: int = cfg.DEFAULT_GOV_TOP_K,
    history_top_k: int = cfg.DEFAULT_HISTORY_TOP_K,
) -> Dict[str, Any]:
    hz = _safe_str(hazard_type) or _safe_str(detection_result.get("hazard_type")) or "hurricane"
    # Partition confirmed into strong-confirmed vs uncertain-like items (possible/has why_uncertain)
    raw_confirmed = _extract_changes_from_detection(detection_result)
    raw_uncertain = _extract_uncertain_from_detection(detection_result)
    confirmed_changes: List[Dict[str, Any]] = []
    uncertain_changes: List[Dict[str, Any]] = list(raw_uncertain)
    for ch in raw_confirmed:
        if isinstance(ch, dict) and _is_uncertain_change(ch):
            uncertain_changes.append(ch)
        else:
            confirmed_changes.append(ch)
    agent = agent or get_assessment_agent()

    per_change_assessments: List[Dict[str, Any]] = []
    for ch in confirmed_changes:
        assessment = assess_one_change(
            agent,
            ch,
            hazard_type=hz,
            gov_top_k=int(gov_top_k),
            history_top_k=int(history_top_k),
            detection_tier="confirmed_disaster_damage",
        )
        per_change_assessments.append({"tier": "confirmed_disaster_damage", "change": ch, "assessment": assessment})

    # uncertain details“details”：details，details overall_level details
    uncertain_change_assessments: List[Dict[str, Any]] = []
    include_uncertain = bool(getattr(cfg, "INCLUDE_UNCERTAIN", True))
    uncertain_weight = float(getattr(cfg, "UNCERTAIN_WEIGHT", 0.3))
    if include_uncertain:
        for ch in uncertain_changes:
            assessment = assess_one_change(
                agent,
                ch,
                hazard_type=hz,
                gov_top_k=int(gov_top_k),
                history_top_k=int(history_top_k),
                detection_tier="uncertain",
            )
            assessment["_evidence_weight"] = float(uncertain_weight)
            uncertain_change_assessments.append({"tier": "uncertain", "change": ch, "assessment": assessment})

    flattened = [
        x["assessment"]
        for x in per_change_assessments
        if isinstance(x, dict) and isinstance(x.get("assessment"), dict)
    ]
    overall_level, overall_conf, overall_reasoning = _compute_overall(flattened)

    # --- uncertain details“details”details ---
    # 1) uncertain details severe
    # 2) uncertain details confirmed=minor details，details moderate（details）
    if include_uncertain and uncertain_change_assessments:
        unc_flat = [
            x["assessment"]
            for x in uncertain_change_assessments
            if isinstance(x, dict) and isinstance(x.get("assessment"), dict)
        ]
        unc_level, unc_conf, unc_reason = _compute_overall(unc_flat) if unc_flat else ("minor", 0.0, "")

        # details confirmed：details conservative（minor）details
        if len(confirmed_changes) == 0 and len(uncertain_changes) > 0:
            overall_level = "minor"
            overall_conf = min(float(overall_conf), 0.25)
            overall_reasoning = (overall_reasoning + "\nMost evidence is UNCERTAIN; overall grade is conservative.").strip()
        else:
            # minor -> moderate details（details uncertain details moderate）
            if overall_level == "minor" and unc_level == "moderate":
                overall_level = "moderate"
                overall_conf = min(float(overall_conf), 0.35)
                overall_reasoning = (overall_reasoning + "\nModerate is suggested only by UNCERTAIN evidence; confidence is lowered.").strip()

            # details：uncertain details overall details severe
            if overall_level == "severe" and _compute_overall(flattened)[0] != "severe":
                overall_level = _compute_overall(flattened)[0]
                overall_conf = min(float(overall_conf), 0.6)
                overall_reasoning = (overall_reasoning + "\nUNCERTAIN evidence cannot alone justify severe.").strip()

            if unc_reason:
                overall_reasoning = (overall_reasoning + "\n\n[uncertain_support]\n" + unc_reason).strip()

    # --- overall_conf：details uncertain details ---
    # NOTE: do NOT let uncertain evidence drag down confirmed confidence when we have confirmed changes.
    # Only use uncertain-weighted confidence when there are no confirmed changes.
    if include_uncertain and uncertain_change_assessments and len(confirmed_changes) == 0:
        conf_sum = 0.0
        w_sum = 0.0
        for item in uncertain_change_assessments:
            if not isinstance(item, dict):
                continue
            a = item.get("assessment")
            if not isinstance(a, dict):
                continue
            lvl = _normalize_level(a.get("predicted_damage_level", ""))
            if lvl not in {"minor", "moderate", "severe"}:
                continue
            conf_sum += float(uncertain_weight) * _coerce_confidence(a.get("confidence", 0.0))
            w_sum += float(uncertain_weight)
        if w_sum > 0:
            overall_conf = float(conf_sum / w_sum)

    # NOTE: "overall" refers to the street-view scene/area covered by the description,
    # not a single building instance.
    out = {
        "overall_area_damage_level": overall_level,
        "overall_area_confidence": overall_conf,
        "overall_area_reasoning": overall_reasoning,
        "per_change_assessments": per_change_assessments,
        "uncertain_change_assessments": uncertain_change_assessments,
        "_meta": {
            "hazard_type": hz,
            "gov_top_k": int(gov_top_k),
            "history_top_k": int(history_top_k),
            "allowed_levels": _allowed_output_levels(),
            "tiers": ["confirmed_disaster_damage", "uncertain", "likely_pseudo_change"],
            "include_uncertain": include_uncertain,
            "uncertain_weight": float(uncertain_weight),
        },
    }
    # Backward-compatible aliases (deprecated): previous naming implied building-level.
    out["overall_damage_level"] = out["overall_area_damage_level"]
    out["overall_confidence"] = out["overall_area_confidence"]
    out["overall_reasoning"] = out["overall_area_reasoning"]
    return out

