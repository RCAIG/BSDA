#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AssessmentAgent/assessment_agent_ragws.py

details RAG-WS (RAG with Web Search) details AssessmentAgent details。

RAG-WS details：
1. Internal RAG: details + details
2. External Web Search: details（details Fallback）

details：
1. details
2. RAG-WS details
3. details
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# details tools details
_AGENTS_ROOT = Path(__file__).resolve().parent.parent
_ASSESSMENT_DIR = Path(__file__).resolve().parent

if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
if str(_ASSESSMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_ASSESSMENT_DIR))

from tools.rag_ws import (
    RAGWS,
    RAGResult,
    RAGWSResult,
    create_assessment_ragws,
)

# details AssessmentAgent details
try:
    from AssessmentAgent.assessment_agent import (
        AssessmentAgent,
        SYSTEM_PROMPT,
        _safe_str,
        _extract_json_object,
        _normalize_level,
        _is_uncertain_change,
        _rule_based_override,
        _allowed_output_levels,
        _level_rank,
        _build_query_text,
        _build_user_prompt,
        _coerce_confidence,
        _default_invalid_response,
        _extract_changes_from_detection,
        _extract_uncertain_from_detection,
        _compute_overall,
    )
    from AssessmentAgent.rag import search_gov_rules, search_history_cases
    from AssessmentAgent import config as cfg
except ImportError:
    from assessment_agent import (
        AssessmentAgent,
        SYSTEM_PROMPT,
        _safe_str,
        _extract_json_object,
        _normalize_level,
        _is_uncertain_change,
        _rule_based_override,
        _allowed_output_levels,
        _level_rank,
        _build_query_text,
        _build_user_prompt,
        _coerce_confidence,
        _default_invalid_response,
        _extract_changes_from_detection,
        _extract_uncertain_from_detection,
        _compute_overall,
    )
    from rag import search_gov_rules, search_history_cases
    import config as cfg


class AssessmentAgentRAGWS:
    """
    details RAG-WS details AssessmentAgent。
    
    details AssessmentAgent details，details RAG-WS details RAG details。
    """
    
    def __init__(
        self,
        model_path: str = cfg.LOCAL_LLM_MODEL_PATH,
        max_new_tokens: int = cfg.MAX_NEW_TOKENS_PER_CHANGE,
        temperature: float = cfg.TEMPERATURE,
        top_p: float = cfg.TOP_P,
        enable_web_search: bool = True,
    ):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.enable_web_search = enable_web_search
        
        # details AssessmentAgent details LLM details
        self._agent = AssessmentAgent(
            model_path=model_path,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        
        # RAG-WS details（details）
        self._ragws_gov: Optional[RAGWS] = None
        self._ragws_history: Optional[RAGWS] = None
    
    def _get_ragws_gov(self) -> RAGWS:
        """details RAG-WS details"""
        if self._ragws_gov is None:
            def rag_func(query: str, top_k: int) -> RAGResult:
                try:
                    hits = search_gov_rules(query, top_k=top_k) or []
                except Exception:
                    hits = []
                
                chunks = []
                scores = []
                for hit in hits:
                    if isinstance(hit, dict):
                        chunks.append({
                            "text": hit.get("text", str(hit)),
                            "source": hit.get("source", "gov_rules"),
                            "meta": hit.get("meta", {}),
                        })
                        scores.append(hit.get("score", 0.5))
                    else:
                        chunks.append({"text": str(hit), "source": "gov_rules"})
                        scores.append(0.5)
                
                return RAGResult(chunks=chunks, scores=scores, query=query, source="internal")
            
            self._ragws_gov = create_assessment_ragws(
                rag_func=rag_func,
                llm_func=None,  # details
                enable_web_search=self.enable_web_search,
            )
        
        return self._ragws_gov
    
    def _get_ragws_history(self) -> RAGWS:
        """details RAG-WS details"""
        if self._ragws_history is None:
            def rag_func(query: str, top_k: int) -> RAGResult:
                try:
                    hits = search_history_cases(query, top_k=top_k) or []
                except Exception:
                    hits = []
                
                chunks = []
                scores = []
                for hit in hits:
                    if isinstance(hit, dict):
                        chunks.append({
                            "text": hit.get("text", str(hit)),
                            "source": hit.get("source", "history_cases"),
                            "meta": hit.get("meta", {}),
                        })
                        scores.append(hit.get("score", 0.5))
                    else:
                        chunks.append({"text": str(hit), "source": "history_cases"})
                        scores.append(0.5)
                
                return RAGResult(chunks=chunks, scores=scores, query=query, source="internal")
            
            self._ragws_history = create_assessment_ragws(
                rag_func=rag_func,
                llm_func=None,
                enable_web_search=self.enable_web_search,
            )
        
        return self._ragws_history
    
    def assess_one_change(
        self,
        change: Dict[str, Any],
        *,
        hazard_type: str = "",
        gov_top_k: int = cfg.DEFAULT_GOV_TOP_K,
        history_top_k: int = cfg.DEFAULT_HISTORY_TOP_K,
        detection_tier: str = "confirmed_disaster_damage",
    ) -> Dict[str, Any]:
        """
        details RAG-WS details。
        """
        # details
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
            out["_ragws_meta"] = {"source": "rule_override", "web_search_triggered": False}
            return out
        
        query_text = _build_query_text(change, hazard_type=hazard_type)
        context = {"hazard_type": hazard_type}
        
        # details RAG-WS details
        ragws_gov = self._get_ragws_gov()
        gov_result = ragws_gov.retrieve(
            query=query_text,
            top_k=gov_top_k,
            context=context,
        )
        gov_rules = gov_result.chunks
        
        # details RAG-WS details
        ragws_history = self._get_ragws_history()
        history_result = ragws_history.retrieve(
            query=query_text,
            top_k=history_top_k,
            context=context,
        )
        history_cases = history_result.chunks
        
        # details prompt details LLM
        user_prompt = _build_user_prompt(
            change=change,
            gov_rules=gov_rules,
            history_cases=history_cases,
            detection_tier=_safe_str(detection_tier) or "confirmed_disaster_damage",
        )
        raw = self._agent.generate_json_assessment(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
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
        
        out["_rag"] = {
            "gov_rules_count": len(gov_rules),
            "history_cases_count": len(history_cases),
        }
        out["_query_text"] = query_text
        out["_raw_model_output"] = raw
        
        # details RAG-WS details
        out["_ragws_meta"] = {
            "gov_source": gov_result.source,
            "gov_web_search_triggered": gov_result.web_search_triggered,
            "gov_evaluation_score": gov_result.evaluation.score,
            "history_source": history_result.source,
            "history_web_search_triggered": history_result.web_search_triggered,
            "history_evaluation_score": history_result.evaluation.score,
        }
        
        return out


def run_assessment_ragws(
    detection_result: Dict[str, Any],
    *,
    agent: Optional[AssessmentAgentRAGWS] = None,
    hazard_type: Optional[str] = None,
    gov_top_k: int = cfg.DEFAULT_GOV_TOP_K,
    history_top_k: int = cfg.DEFAULT_HISTORY_TOP_K,
    enable_web_search: bool = True,
) -> Dict[str, Any]:
    """
    details RAG-WS details Assessment。
    
    details run_assessment details RAG-WS details。
    """
    hz = _safe_str(hazard_type) or _safe_str(detection_result.get("hazard_type")) or "hurricane"
    
    # details confirmed details uncertain
    raw_confirmed = _extract_changes_from_detection(detection_result)
    raw_uncertain = _extract_uncertain_from_detection(detection_result)
    confirmed_changes: List[Dict[str, Any]] = []
    uncertain_changes: List[Dict[str, Any]] = list(raw_uncertain)
    for ch in raw_confirmed:
        if isinstance(ch, dict) and _is_uncertain_change(ch):
            uncertain_changes.append(ch)
        else:
            confirmed_changes.append(ch)
    
    # details agent
    if agent is None:
        agent = AssessmentAgentRAGWS(enable_web_search=enable_web_search)
    
    per_change_assessments: List[Dict[str, Any]] = []
    for ch in confirmed_changes:
        assessment = agent.assess_one_change(
            ch,
            hazard_type=hz,
            gov_top_k=int(gov_top_k),
            history_top_k=int(history_top_k),
            detection_tier="confirmed_disaster_damage",
        )
        per_change_assessments.append({
            "tier": "confirmed_disaster_damage",
            "change": ch,
            "assessment": assessment,
        })
    
    # uncertain details
    uncertain_change_assessments: List[Dict[str, Any]] = []
    include_uncertain = bool(getattr(cfg, "INCLUDE_UNCERTAIN", True))
    uncertain_weight = float(getattr(cfg, "UNCERTAIN_WEIGHT", 0.3))
    if include_uncertain:
        for ch in uncertain_changes:
            assessment = agent.assess_one_change(
                ch,
                hazard_type=hz,
                gov_top_k=int(gov_top_k),
                history_top_k=int(history_top_k),
                detection_tier="uncertain",
            )
            assessment["_evidence_weight"] = float(uncertain_weight)
            uncertain_change_assessments.append({
                "tier": "uncertain",
                "change": ch,
                "assessment": assessment,
            })
    
    # details overall
    flattened = [
        x["assessment"]
        for x in per_change_assessments
        if isinstance(x, dict) and isinstance(x.get("assessment"), dict)
    ]
    overall_level, overall_conf, overall_reasoning = _compute_overall(flattened)
    
    # uncertain details（details）
    if include_uncertain and uncertain_change_assessments:
        unc_flat = [
            x["assessment"]
            for x in uncertain_change_assessments
            if isinstance(x, dict) and isinstance(x.get("assessment"), dict)
        ]
        unc_level, unc_conf, unc_reason = _compute_overall(unc_flat) if unc_flat else ("minor", 0.0, "")
        
        if len(confirmed_changes) == 0 and len(uncertain_changes) > 0:
            overall_level = "minor"
            overall_conf = min(float(overall_conf), 0.25)
            overall_reasoning = (overall_reasoning + "\nMost evidence is UNCERTAIN; overall grade is conservative.").strip()
        else:
            if overall_level == "minor" and unc_level == "moderate":
                overall_level = "moderate"
                overall_conf = min(float(overall_conf), 0.35)
                overall_reasoning = (overall_reasoning + "\nModerate is suggested only by UNCERTAIN evidence; confidence is lowered.").strip()
            
            if overall_level == "severe" and _compute_overall(flattened)[0] != "severe":
                overall_level = _compute_overall(flattened)[0]
                overall_conf = min(float(overall_conf), 0.6)
                overall_reasoning = (overall_reasoning + "\nUNCERTAIN evidence cannot alone justify severe.").strip()
            
            if unc_reason:
                overall_reasoning = (overall_reasoning + "\n\n[uncertain_support]\n" + unc_reason).strip()
    
    # details RAG-WS details
    ragws_stats = {
        "total_assessments": len(per_change_assessments) + len(uncertain_change_assessments),
        "gov_web_search_count": sum(
            1 for x in per_change_assessments + uncertain_change_assessments
            if x.get("assessment", {}).get("_ragws_meta", {}).get("gov_web_search_triggered", False)
        ),
        "history_web_search_count": sum(
            1 for x in per_change_assessments + uncertain_change_assessments
            if x.get("assessment", {}).get("_ragws_meta", {}).get("history_web_search_triggered", False)
        ),
    }
    
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
            "ragws_enabled": True,
            "web_search_enabled": enable_web_search,
        },
        "_ragws_stats": ragws_stats,
    }
    
    # details
    out["overall_damage_level"] = out["overall_area_damage_level"]
    out["overall_confidence"] = out["overall_area_confidence"]
    out["overall_reasoning"] = out["overall_area_reasoning"]
    
    return out


# ----------------------------------------------------------------------
# details
# ----------------------------------------------------------------------

_GLOBAL_AGENT_RAGWS: Optional[AssessmentAgentRAGWS] = None


def get_assessment_agent_ragws(enable_web_search: bool = True) -> AssessmentAgentRAGWS:
    """details AssessmentAgentRAGWS details"""
    global _GLOBAL_AGENT_RAGWS
    if _GLOBAL_AGENT_RAGWS is None:
        _GLOBAL_AGENT_RAGWS = AssessmentAgentRAGWS(
            model_path=cfg.LOCAL_LLM_MODEL_PATH,
            enable_web_search=enable_web_search,
        )
    return _GLOBAL_AGENT_RAGWS

