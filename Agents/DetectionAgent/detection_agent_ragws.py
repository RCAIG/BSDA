#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/detection_agent_ragws.py

details RAG-WS (RAG with Web Search) details DetectionAgent details。

RAG-WS details：
1. Internal RAG: details
2. External Web Search: details（details Fallback）

details：
1. details (Step 1)
2. RAG-WS details (Step 2) - details Evidence Evaluator + Query Rewriter + Web Search Fallback
3. details (Step 3)

details：details，details detection_agent.py details。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# details tools details
_AGENTS_ROOT = Path(__file__).resolve().parent.parent
_DETECTION_DIR = Path(__file__).resolve().parent

# details
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
if str(_DETECTION_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTION_DIR))

from tools.rag_ws import (
    RAGWS,
    RAGResult,
    RAGWSResult,
    create_detection_ragws,
)

# details DetectionAgent details RAG（details）
try:
    from DetectionAgent.rag import DamageFeatureRAG
except ImportError:
    # details DetectionAgent details，details
    from rag import DamageFeatureRAG

# details LLM
from shared_llm import get_shared_llm


# ----------------------------------------------------------------------
# details（details detection_agent.py details）
# ----------------------------------------------------------------------
def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _repair_truncated_json(text: str) -> str:
    if not text or not text.strip():
        return text
    
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    open_quotes = text.count('"') - text.count('\\"')
    
    repaired = text
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    
    if open_quotes % 2 != 0:
        last_quote_idx = repaired.rfind('"')
        if last_quote_idx >= 0:
            if last_quote_idx > 0 and repaired[last_quote_idx - 1] != "\\":
                repaired = repaired[:last_quote_idx + 1] + '"'
    
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)
    
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)
    
    return repaired


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    t = _safe_str(text)
    if not t:
        return None

    m = re.search(r"```json\s*([\s\S]*:)\s*```", t, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                repaired = _repair_truncated_json(candidate)
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    l = t.find("{")
    r = t.rfind("}")
    if l >= 0 and r > l:
        candidate = t[l : r + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            try:
                repaired = _repair_truncated_json(candidate)
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    
    if l >= 0 and r <= l:
        candidate = t[l:].strip()
        try:
            repaired = _repair_truncated_json(candidate)
            obj = json.loads(repaired)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    
    return None


def _normalize_step3_buckets(obj: Dict[str, Any]) -> Dict[str, Any]:
    if "confirmed_disaster_damage" not in obj and "verified_hurricane_damages" in obj:
        obj["confirmed_disaster_damage"] = obj.get("verified_hurricane_damages")
    if "likely_pseudo_change" not in obj and "pseudo_changes" in obj:
        obj["likely_pseudo_change"] = obj.get("pseudo_changes")
    if "uncertain" not in obj:
        obj["uncertain"] = obj.get("uncertain_changes", []) if isinstance(obj.get("uncertain_changes"), list) else []

    confirmed = obj.get("confirmed_disaster_damage", [])
    pseudo = obj.get("likely_pseudo_change", [])
    uncertain = obj.get("uncertain", [])
    if not isinstance(confirmed, list):
        confirmed = []
    if not isinstance(pseudo, list):
        pseudo = []
    if not isinstance(uncertain, list):
        uncertain = []

    def _dr(x: Any) -> str:
        return _safe_str(x).lower()

    confirmed_out: List[Dict[str, Any]] = []
    pseudo_out: List[Dict[str, Any]] = []
    uncertain_out: List[Dict[str, Any]] = []

    pool: List[Tuple[str, Dict[str, Any]]] = []
    for it in confirmed:
        if isinstance(it, dict):
            pool.append(("confirmed", it))
    for it in pseudo:
        if isinstance(it, dict):
            pool.append(("pseudo", it))
    for it in uncertain:
        if isinstance(it, dict):
            pool.append(("uncertain", it))

    for bucket, it in pool:
        dr = _dr(it.get("disaster_related", ""))
        if dr in {"likely", "possible"}:
            confirmed_out.append(it)
            continue
        if dr in {"unlikely"}:
            pseudo_out.append(it)
            continue
        if bucket == "pseudo":
            uncertain_out.append(it)
        elif bucket == "confirmed":
            uncertain_out.append(it)
        else:
            uncertain_out.append(it)

    obj["confirmed_disaster_damage"] = confirmed_out
    obj["likely_pseudo_change"] = pseudo_out
    obj["uncertain"] = uncertain_out
    obj["verified_hurricane_damages"] = confirmed_out
    obj["pseudo_changes"] = pseudo_out
    return obj


# ----------------------------------------------------------------------
# DetectionAgentRAGWS
# ----------------------------------------------------------------------
@dataclass
class DetectionAgentRAGWS:
    """
    details RAG-WS details DetectionAgent。
    
    details，details detection_agent.py。
    """
    
    model_path: str
    rag: DamageFeatureRAG
    max_new_tokens: int = 800
    max_new_tokens_step1: Optional[int] = None
    max_new_tokens_step3: Optional[int] = None
    rag_top_k: int = 5
    hazard_type: str = "hurricane"
    enable_web_search: bool = True
    ragws_max_retries: int = 2

    def __post_init__(self) -> None:
        tokenizer, model, torch = get_shared_llm(self.model_path)
        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model
        self._ragws: Optional[RAGWS] = None

    def _get_ragws(self) -> RAGWS:
        """details RAG-WS details"""
        if self._ragws is None:
            def rag_func(query: str, top_k: int) -> RAGResult:
                hits = self.rag.search_for_detection_change(
                    hazard_type=self.hazard_type,
                    component="",
                    change_desc=query,
                    pre_desc="",
                    post_desc="",
                    extra_context="",
                    language="en",
                    top_k=top_k,
                )
                
                chunks = []
                scores = []
                for hit in hits:
                    chunks.append({
                        "text": hit.get("text", ""),
                        "source": hit.get("source", ""),
                        "meta": hit.get("meta", {}),
                    })
                    scores.append(hit.get("score", 0.0))
                
                return RAGResult(
                    chunks=chunks,
                    scores=scores,
                    query=query,
                    source="internal",
                )
            
            def llm_func(system_prompt: str, user_prompt: str) -> str:
                return self._generate_llm(
                    system_prompt,
                    user_prompt,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.0,
                )
            
            self._ragws = create_detection_ragws(
                rag_func=rag_func,
                llm_func=llm_func,
                enable_web_search=self.enable_web_search,
            )
        
        return self._ragws

    def _generate_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_new_tokens: Optional[int] = None,
        do_sample: Optional[bool] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> str:
        system_prompt = _safe_str(system_prompt)
        user_prompt = _safe_str(user_prompt)
        if not user_prompt:
            return ""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text_input = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text_input = system_prompt + "\n\n" + user_prompt

        model_inputs = self._tokenizer([text_input], return_tensors="pt").to(self._model.device)
        mnt = self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        _do_sample = True if do_sample is None else bool(do_sample)
        _temperature = 0.25 if temperature is None else float(temperature)
        _top_p = 0.9 if top_p is None else float(top_p)
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=int(mnt),
                temperature=_temperature,
                do_sample=_do_sample,
                top_p=_top_p,
                repetition_penalty=1.05,
            )

        gen_only = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        out = self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
        return _safe_str(out)

    def _build_change_extraction_prompt(
        self,
        pre_text: str,
        post_text: str,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        pre_text = _safe_str(pre_text)
        post_text = _safe_str(post_text)

        return (
            "Below are pre-disaster and post-disaster text descriptions (structured JSON) of the same location. "
            "Your task is to extract changes only, without judging whether they are disaster-related.\n\n"
            "Output requirement: Output only a single JSON object (no markdown code blocks).\n\n"
            "JSON schem<LOCAL_PATH>"
            "{\n"
            '  "changes": [\n'
            "    {\n"
            '      "module": "module name",\n'
            '      "change_description": "Summarize the difference in 1-2 sentences",\n'
            '      "pre_evidence": "Evidence from pre-disaster description",\n'
            '      "post_evidence": "Evidence from post-disaster description"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            f"[Pre-disaster description]\n{pre_text}\n\n"
            f"[Post-disaster description]\n{post_text}\n"
        )

    def extract_changes(
        self,
        pre_text: str,
        post_text: str,
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        system_prompt = (
            "You are a rigorous change extraction assistant. "
            "Extract changes from pre/post-disaster descriptions. Output JSON only."
        )
        user_prompt = self._build_change_extraction_prompt(pre_text, post_text, pair_meta)
        raw = self._generate_llm(
            system_prompt,
            user_prompt,
            max_new_tokens=self.max_new_tokens_step1 or 1500,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return []

        changes = obj.get("changes", [])
        if not isinstance(changes, list):
            return []

        cleaned: List[Dict[str, Any]] = []
        for i, ch in enumerate(changes, start=1):
            if not isinstance(ch, dict):
                continue
            module = _safe_str(ch.get("module", "")) or "unknown"
            change_description = _safe_str(ch.get("change_description", ""))
            pre_evidence = _safe_str(ch.get("pre_evidence", ""))
            post_evidence = _safe_str(ch.get("post_evidence", ""))

            if not (change_description or pre_evidence or post_evidence):
                continue

            component_hint = module
            if pre_evidence:
                component_hint = pre_evidence[:24]
            elif post_evidence:
                component_hint = post_evidence[:24]

            ch_out = {
                "id": f"chg_{i:03d}",
                "module": module,
                "change_description": change_description,
                "pre_evidence": pre_evidence,
                "post_evidence": post_evidence,
                "component": component_hint,
            }
            cleaned.append(ch_out)
        return cleaned

    def enrich_changes_with_rag(
        self,
        changes: List[Dict[str, Any]],
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """details RAG-WS details"""
        if not changes:
            return []
        
        ragws = self._get_ragws()
        enriched: List[Dict[str, Any]] = []
        
        for ch in changes:
            module = _safe_str(ch.get("module", ""))
            component = _safe_str(ch.get("component", ""))
            change_desc = _safe_str(ch.get("change_description", ""))
            
            query_parts = []
            if self.hazard_type:
                query_parts.append(self.hazard_type)
            if component:
                query_parts.append(component)
            if change_desc:
                query_parts.append(change_desc)
            query = " ".join(query_parts)
            
            context = {
                "hazard_type": self.hazard_type,
                "component": component,
                "module": module,
            }
            
            ragws_result = ragws.retrieve(
                query=query,
                top_k=self.rag_top_k,
                context=context,
            )
            
            hits = []
            for chunk in ragws_result.chunks:
                hits.append({
                    "text": chunk.get("text", ""),
                    "source": chunk.get("source", ""),
                    "meta": chunk.get("meta", {}),
                })
            
            ch2 = dict(ch)
            ch2["rag_hits"] = hits
            ch2["_ragws_meta"] = {
                "source": ragws_result.source,
                "is_sufficient": ragws_result.is_sufficient,
                "attempts": ragws_result.attempts,
                "web_search_triggered": ragws_result.web_search_triggered,
                "evaluation_score": ragws_result.evaluation.score,
                "evaluation_reason": ragws_result.evaluation.reason,
            }
            enriched.append(ch2)
        
        return enriched

    def _build_classification_prompt(
        self,
        enriched_changes: List[Dict[str, Any]],
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        changes_json = json.dumps(enriched_changes, ensure_ascii=False, indent=2)

        return (
            f"You are a disaster analysis Detection Agent. Disaster type: {self.hazard_type}.\n\n"
            "For each change, classify int<LOCAL_PATH>"
            "1) confirmed_disaster_damage: High confidence disaster-related\n"
            "2) likely_pseudo_change: High confidence non-disaster causes\n"
            "3) uncertain: Insufficient evidence\n\n"
            "[Change entries and their RAG hits]\n"
            f"{changes_json}\n\n"
            "Output a JSON object with these key<LOCAL_PATH>"
            "- confirmed_disaster_damage: []\n"
            "- likely_pseudo_change: []\n"
            "- uncertain: []\n"
        )

    def classify_with_rag(
        self,
        enriched_changes: List[Dict[str, Any]],
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if not enriched_changes:
            return {
                "verified_hurricane_damages": [],
                "pseudo_changes": [],
                "_note": "no_changes_extracted",
            }

        system_prompt = (
            "You are a disaster analysis Detection Agent. Output JSON only."
        )
        user_prompt = self._build_classification_prompt(enriched_changes, pair_meta=pair_meta)
        raw = self._generate_llm(
            system_prompt,
            user_prompt,
            max_new_tokens=self.max_new_tokens_step3 or 1500,
        )
        obj = _extract_json_object(raw)

        if not isinstance(obj, dict):
            return {
                "verified_hurricane_damages": [],
                "pseudo_changes": [],
                "_parse_failed": True,
                "_raw_output": raw,
            }

        if not isinstance(obj.get("verified_hurricane_damages", None), list):
            obj["verified_hurricane_damages"] = []
        if not isinstance(obj.get("pseudo_changes", None), list):
            obj["pseudo_changes"] = []

        obj = _normalize_step3_buckets(obj)
        return obj

    def run(
        self,
        pre_text: str,
        post_text: str,
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
        max_changes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """details"""
        pre_text = _safe_str(pre_text)
        post_text = _safe_str(post_text)

        # Step 1: details
        changes = self.extract_changes(pre_text, post_text, pair_meta=pair_meta)

        if max_changes is not None:
            try:
                n = int(max_changes)
            except Exception:
                n = 0
            if n > 0:
                changes = changes[:n]

        # Step 2: details RAG-WS details
        enriched_changes = self.enrich_changes_with_rag(changes, pair_meta=pair_meta)

        # Step 3: details
        result = self.classify_with_rag(enriched_changes, pair_meta=pair_meta)

        result["_intermediate"] = {
            "changes": changes,
            "enriched_changes": enriched_changes,
        }

        # RAG-WS details
        ragws_stats = {
            "total_changes": len(enriched_changes),
            "web_search_triggered_count": sum(
                1 for ch in enriched_changes
                if ch.get("_ragws_meta", {}).get("web_search_triggered", False)
            ),
            "avg_evaluation_score": (
                sum(ch.get("_ragws_meta", {}).get("evaluation_score", 0) for ch in enriched_changes)
                / len(enriched_changes)
                if enriched_changes else 0
            ),
        }
        result["_ragws_stats"] = ragws_stats

        return result


# ----------------------------------------------------------------------
# details
# ----------------------------------------------------------------------

def get_detection_agent_ragws(
    model_path: str,
    rag: DamageFeatureRAG,
    *,
    enable_web_search: bool = True,
    hazard_type: str = "hurricane",
    rag_top_k: int = 5,
    max_new_tokens: int = 800,
) -> DetectionAgentRAGWS:
    """details RAG-WS details DetectionAgent details"""
    return DetectionAgentRAGWS(
        model_path=model_path,
        rag=rag,
        enable_web_search=enable_web_search,
        hazard_type=hazard_type,
        rag_top_k=rag_top_k,
        max_new_tokens=max_new_tokens,
    )
