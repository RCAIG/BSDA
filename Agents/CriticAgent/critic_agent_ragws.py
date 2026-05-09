#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/critic_agent_ragws.py

details RAG-WS (RAG with Web Search) details CriticAgent details。

RAG-WS details：
1. Internal RAG: details
2. External Web Search: details（details Fallback）

details：
1. details Critic details
2. RAG-WS details
3. details QC details
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# details tools details
_AGENTS_ROOT = Path(__file__).resolve().parent.parent
_CRITIC_DIR = Path(__file__).resolve().parent

if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
if str(_CRITIC_DIR) not in sys.path:
    sys.path.insert(0, str(_CRITIC_DIR))

from tools.rag_ws import (
    RAGWS,
    RAGResult,
    RAGWSResult,
    create_critic_ragws,
)

# details CriticAgent details
try:
    from CriticAgent.critic_agent import (
        CriticAgent as OriginalCriticAgent,
        _safe_str,
        _now_iso,
        _extract_json_object,
        _normalize_confidence_label,
        _clean_critic_output,
        _extract_overall_grade_and_confidence,
        _collect_valid_change_ids,
        _post_validate_against_inputs,
        _generate_llm,
    )
    from CriticAgent import config as critic_cfg
    from CriticAgent import rag as critic_rag
except ImportError:
    from critic_agent import (
        CriticAgent as OriginalCriticAgent,
        _safe_str,
        _now_iso,
        _extract_json_object,
        _normalize_confidence_label,
        _clean_critic_output,
        _extract_overall_grade_and_confidence,
        _collect_valid_change_ids,
        _post_validate_against_inputs,
        _generate_llm,
    )
    import config as critic_cfg
    import rag as critic_rag


class CriticAgentRAGWS:
    """
    details RAG-WS details CriticAgent。
    
    details CriticAgent details，details RAG-WS details RAG details。
    """
    
    critic_version: str = critic_cfg.CRITIC_VERSION + "-ragws"
    
    def __init__(
        self,
        *,
        use_rag: bool = critic_cfg.DEFAULT_USE_RAG,
        rag_top_k: int = critic_cfg.DEFAULT_RAG_TOP_K,
        use_llm: Optional[bool] = None,
        enable_web_search: bool = True,
    ) -> None:
        """
        Args:
            use_rag: details RAG
            rag_top_k: RAG details top-k
            use_llm: details LLM
            enable_web_search: details
        """
        self.use_rag = bool(use_rag)
        self.rag_top_k = int(rag_top_k)
        self.use_llm = bool(critic_cfg.ENABLE_LLM_CRITIC) if use_llm is None else bool(use_llm)
        self.enable_web_search = enable_web_search
        
        # RAG-WS details（details）
        self._ragws: Optional[RAGWS] = None
    
    def _get_ragws(self) -> RAGWS:
        """details RAG-WS details"""
        if self._ragws is None:
            def rag_func(query: str, top_k: int) -> RAGResult:
                try:
                    hits = critic_rag.search_gov_rules_for_critic(query, top_k=top_k) or []
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
            
            self._ragws = create_critic_ragws(
                rag_func=rag_func,
                llm_func=None,  # details
                enable_web_search=self.enable_web_search,
            )
        
        return self._ragws
    
    def run(
        self,
        *,
        perception_output: Dict[str, Any],
        detection_output: Dict[str, Any],
        assessment_output: Dict[str, Any],
        rules_summary: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        details Critic details。
        
        details RAG-WS details，details QC details。
        """
        # RAG-WS details
        used_rules_snippets: List[Dict[str, Any]] = []
        ragws_meta: Dict[str, Any] = {}
        
        if self.use_rag and rules_summary is None:
            hazard = _safe_str(assessment_output.get("_meta", {}).get("hazard_type")) or _safe_str(
                detection_output.get("hazard_type")
            )
            
            # details
            q = critic_rag.build_critic_rag_query(
                hazard_type=hazard or "hurricane",
                overall_grade=_safe_str(
                    assessment_output.get("overall_area_damage_level")
                    or assessment_output.get("overall_damage_level")
                ),
                overall_confidence_label=_safe_str(
                    assessment_output.get("overall_area_confidence")
                    or assessment_output.get("overall_confidence")
                ),
                post_scene_summary=_safe_str(
                    perception_output.get("post_scene", {}).get("summary")
                ) if isinstance(perception_output.get("post_scene"), dict) else "",
                overall_reasoning=_safe_str(
                    assessment_output.get("overall_area_reasoning")
                    or assessment_output.get("overall_reasoning")
                ),
            )
            
            # details RAG-WS details
            ragws = self._get_ragws()
            context = {"hazard_type": hazard or "hurricane"}
            ragws_result = ragws.retrieve(
                query=q,
                top_k=int(self.rag_top_k),
                context=context,
            )
            
            used_rules_snippets = ragws_result.chunks
            ragws_meta = {
                "source": ragws_result.source,
                "web_search_triggered": ragws_result.web_search_triggered,
                "evaluation_score": ragws_result.evaluation.score,
                "evaluation_reason": ragws_result.evaluation.reason,
                "attempts": ragws_result.attempts,
            }
        
        if not self.use_llm:
            # LLM details fallback
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
                "critic_recommendation": {
                    "proposed_grade": None,
                    "proposed_confidence": None,
                    "rationale": "LLM disabled.",
                },
                "_meta": {
                    "critic_version": self.critic_version,
                    "timestamp": _now_iso(),
                    "used_rules": list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [],
                    "used_rules_snippets": used_rules_snippets,
                    "ragws_meta": ragws_meta,
                },
            }
        
        # details system prompt
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
        
        # details
        current_grade, current_conf = _extract_overall_grade_and_confidence(
            assessment_output if isinstance(assessment_output, dict) else {}
        )
        facts = f"FACTS (must treat as ground truth): current_overall_grade={current_grade or 'unknown'}; current_overall_confidence={current_conf if current_conf is not None else 'unknown'}."
        
        # details payload
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
                "critic_recommendation": {
                    "proposed_grade": None,
                    "proposed_confidence": None,
                    "rationale": "LLM output invalid.",
                },
                "_meta": {
                    "critic_version": self.critic_version,
                    "timestamp": _now_iso(),
                    "used_rules": list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [],
                    "used_rules_snippets": used_rules_snippets,
                    "ragws_meta": ragws_meta,
                    "llm_parse_failed": True,
                    "llm_raw_output": raw,
                },
            }
        
        # details
        cleaned = _clean_critic_output(obj)
        
        meta = cleaned.get("_meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("critic_version", self.critic_version)
        meta.setdefault("timestamp", _now_iso())
        meta.setdefault("used_rules", list((rules_summary or {}).keys()) if isinstance(rules_summary, dict) else [])
        meta.setdefault("used_rules_snippets", used_rules_snippets)
        meta.setdefault("ragws_meta", ragws_meta)
        meta.setdefault("llm_raw_output", _safe_str(raw))
        cleaned["_meta"] = meta
        
        return _post_validate_against_inputs(
            cleaned,
            detection_output=detection_output if isinstance(detection_output, dict) else {},
            assessment_output=assessment_output if isinstance(assessment_output, dict) else {},
        )


# ----------------------------------------------------------------------
# details
# ----------------------------------------------------------------------

def get_critic_agent_ragws(
    *,
    use_rag: bool = True,
    rag_top_k: int = critic_cfg.DEFAULT_RAG_TOP_K,
    use_llm: bool = True,
    enable_web_search: bool = True,
) -> CriticAgentRAGWS:
    """
    details CriticAgentRAGWS details。
    
    Args:
        use_rag: details RAG
        rag_top_k: RAG details top-k
        use_llm: details LLM
        enable_web_search: details
    
    Returns:
        CriticAgentRAGWS details
    """
    return CriticAgentRAGWS(
        use_rag=use_rag,
        rag_top_k=rag_top_k,
        use_llm=use_llm,
        enable_web_search=enable_web_search,
    )

