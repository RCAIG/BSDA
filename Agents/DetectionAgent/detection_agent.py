#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/detection_agent.py

details A details DetectionAgent：

Step 1: details
  - details：details + details
  - details：details“details changes”，details：
      - id
      - component
      - location
      - pre_state
      - post_state
      - change_description

Step 2: details RAG
  - details DamageFeatureRAG.search_for_detection_change(...)
  - details hazard_type（details hurricane，details）
  - details hits

Step 3: details
  - details「details + details RAG hits」details JSON details LOCAL_MODEL
  - details LOCAL_MODEL details：
      - verified_hurricane_damages
      - pseudo_changes
    details disaster_related / reason / confidence details

details：details LLM details，details“details”details“details”details。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from rag import DamageFeatureRAG
from shared_llm import get_shared_llm


# ----------------------------------------------------------------------
# details
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
    """
    details JSON：
    - details
    - details
    - details/details
    """
    if not text or not text.strip():
        return text
    
    # details/details
    open_braces = text.count("{")
    close_braces = text.count("}")
    open_brackets = text.count("[")
    close_brackets = text.count("]")
    open_quotes = text.count('"') - text.count('\\"')
    
    repaired = text
    
    # details，details
    repaired = repaired.rstrip()
    if repaired.endswith(","):
        repaired = repaired[:-1]
    
    # details（details），details
    if open_quotes % 2 != 0:
        # details
        last_quote_idx = repaired.rfind('"')
        if last_quote_idx >= 0:
            # details
            if last_quote_idx > 0 and repaired[last_quote_idx - 1] != "\\":
                # details
                repaired = repaired[:last_quote_idx + 1] + '"'
    
    # details
    if open_brackets > close_brackets:
        repaired += "]" * (open_brackets - close_brackets)
    
    # details
    if open_braces > close_braces:
        repaired += "}" * (open_braces - close_braces)
    
    return repaired


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """
    details JSON details（details）：
    - details ```json ... ``` details
    - details '{' details '}' details json.loads
    - details，details JSON details
    """
    t = _safe_str(text)
    if not t:
        return None

    # fenced block
    m = re.search(r"```json\s*([\s\S]*:)\s*```", t, flags=re.IGNORECASE)
    if m:
        candidate = m.group(1).strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # details
            try:
                repaired = _repair_truncated_json(candidate)
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass

    # first {...last}
    l = t.find("{")
    r = t.rfind("}")
    if l >= 0 and r > l:
        candidate = t[l : r + 1].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            # details
            try:
                repaired = _repair_truncated_json(candidate)
                obj = json.loads(repaired)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                pass
    
    # details '}'，details '{' details，details
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
    """
    details Step3 details“details”：
    - details：confirmed_disaster_damage / likely_pseudo_change / uncertain
    - details verified_hurricane_damages / pseudo_changes，details。
    - details disaster_related details（details likely/possible details pseudo），details disaster_related details。
    """
    # -------- details schema --------
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

    # details，details disaster_related details
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
        # details：details confirmed
        if dr in {"likely", "possible"}:
            confirmed_out.append(it)
            continue
        # details：details pseudo
        if dr in {"unlikely"}:
            pseudo_out.append(it)
            continue
        # details/details：details，details uncertain
        if bucket == "pseudo":
            # details；details dr details unknown，details uncertain
            uncertain_out.append(it)
        elif bucket == "confirmed":
            uncertain_out.append(it)
        else:
            uncertain_out.append(it)

    obj["confirmed_disaster_damage"] = confirmed_out
    obj["likely_pseudo_change"] = pseudo_out
    obj["uncertain"] = uncertain_out

    # details（details），details“details”
    obj["verified_hurricane_damages"] = confirmed_out
    obj["pseudo_changes"] = pseudo_out
    return obj


# ----------------------------------------------------------------------
# Heuristic fallback (when LLM fails due to truncation/OOM etc.)
# ----------------------------------------------------------------------
def _try_parse_json_text(s: str) -> Optional[Dict[str, Any]]:
    try:
        obj = json.loads(_safe_str(s))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _heuristic_fallback_from_pair_meta(pair_meta: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(pair_meta, dict):
        return []
    pre_raw = _safe_str(pair_meta.get("pre_text_raw", ""))
    post_raw = _safe_str(pair_meta.get("post_text_raw", ""))
    pre_obj = _try_parse_json_text(pre_raw)
    post_obj = _try_parse_json_text(post_raw)
    if isinstance(pre_obj, dict) and isinstance(post_obj, dict):
        return _heuristic_changes_from_perception_json(pre_obj=pre_obj, post_obj=post_obj)
    return []


def _heuristic_changes_from_perception_json(
    *,
    pre_obj: Dict[str, Any],
    post_obj: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    details LLM details/details，details diff details。
    details：details Step2/Step3 details，details no_changes_extracted。
    """

    def g(obj: Dict[str, Any], *keys: str) -> str:
        cur: Any = obj
        for k in keys:
            if not isinstance(cur, dict):
                return ""
            cur = cur.get(k)
        return _safe_str(cur)

    candidates: List[Tuple[str, str, str, str]] = []

    # overall_scene
    for k in [("overall_scene", "weather"), ("overall_scene", "lighting"), ("overall_scene", "time_of_day")]:
        pre_v = g(pre_obj, *k)
        post_v = g(post_obj, *k)
        if (pre_v or post_v) and pre_v != post_v:
            candidates.append(("overall_scene", f"{k[-1]} details：{pre_v} -> {post_v}", pre_v, post_v))

    pre_free = g(pre_obj, "overall_scene", "free_text")
    post_free = g(post_obj, "overall_scene", "free_text")
    if (pre_free or post_free) and pre_free != post_free:
        candidates.append(("overall_scene", "details（details）", pre_free[:80], post_free[:80]))

    # road
    pre_surface = g(pre_obj, "road_and_traffic", "surface_condition")
    post_surface = g(post_obj, "road_and_traffic", "surface_condition")
    if (pre_surface or post_surface) and pre_surface != post_surface:
        candidates.append(("road_and_traffic", "details", pre_surface[:100], post_surface[:100]))

    pre_obs = g(pre_obj, "road_and_traffic", "obstacles_on_road")
    post_obs = g(post_obj, "road_and_traffic", "obstacles_on_road")
    if (pre_obs or post_obs) and pre_obs != post_obs:
        candidates.append(("road_and_traffic", "details", pre_obs[:100], post_obs[:100]))

    # buildings
    pre_dmg = g(pre_obj, "buildings_and_structures", "damage_level_qualitative")
    post_dmg = g(post_obj, "buildings_and_structures", "damage_level_qualitative")
    if (pre_dmg or post_dmg) and pre_dmg != post_dmg:
        candidates.append(("buildings_and_structures", "details", pre_dmg[:120], post_dmg[:120]))

    pre_fac = g(pre_obj, "buildings_and_structures", "facades_and_roofs_damage")
    post_fac = g(post_obj, "buildings_and_structures", "facades_and_roofs_damage")
    if (pre_fac or post_fac) and pre_fac != post_fac:
        candidates.append(("buildings_and_structures", "details/details", pre_fac[:120], post_fac[:120]))

    pre_deb = g(pre_obj, "buildings_and_structures", "debris_near_buildings")
    post_deb = g(post_obj, "buildings_and_structures", "debris_near_buildings")
    if (pre_deb or post_deb) and pre_deb != post_deb:
        candidates.append(("buildings_and_structures", "details/details", pre_deb[:120], post_deb[:120]))

    # vegetation / debris / water traces
    pre_trash = g(pre_obj, "vegetation_ground_and_debris", "debris_and_objects") or g(
        pre_obj, "vegetation_ground_and_debris", "loose_debris_and_trash"
    )
    post_trash = g(post_obj, "vegetation_ground_and_debris", "debris_and_objects") or g(
        post_obj, "vegetation_ground_and_debris", "loose_debris_and_trash"
    )
    if (pre_trash or post_trash) and pre_trash != post_trash:
        candidates.append(("vegetation_ground_and_debris", "details/details", pre_trash[:120], post_trash[:120]))

    pre_water = g(pre_obj, "vegetation_ground_and_debris", "water_related_ground_features")
    post_water = g(post_obj, "vegetation_ground_and_debris", "water_related_ground_features")
    if (pre_water or post_water) and pre_water != post_water:
        candidates.append(("vegetation_ground_and_debris", "details", pre_water[:120], post_water[:120]))

    out: List[Dict[str, Any]] = []
    for i, (module, desc, pre_ev, post_ev) in enumerate(candidates[:8], start=1):
        out.append(
            {
                "id": f"chg_{i:03d}",
                "module": module,
                "change_description": _safe_str(desc),
                "pre_evidence": _safe_str(pre_ev),
                "post_evidence": _safe_str(post_ev),
                "component": _safe_str(module),
                "_heuristic": True,
            }
        )
    return out


# ----------------------------------------------------------------------
# DetectionAgent
# ----------------------------------------------------------------------
@dataclass
class DetectionAgent:
    model_path: str
    rag: DamageFeatureRAG
    max_new_tokens: int = 800
    max_new_tokens_step1: Optional[int] = None
    max_new_tokens_step3: Optional[int] = None
    rag_top_k: int = 5
    hazard_type: str = "hurricane"  # details "typhoon" / "wind" details

    def __post_init__(self) -> None:
        # details LOCAL_MODEL details（details）
        tokenizer, model, torch = get_shared_llm(self.model_path)
        self._torch = torch  # type: ignore[assignment]
        self._tokenizer = tokenizer  # type: ignore[assignment]
        self._model = model  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # details LLM details
    # ------------------------------------------------------------------
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
            # details：details + details
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

    # ------------------------------------------------------------------
    # Step 1：details
    # ------------------------------------------------------------------
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
            "Working method: Compare module by module, identify points where 'pre and post are clearly inconsistent'. Do not fabricate.\n"
            "Suggested module orde<LOCAL_PATH>"
            "- overall_scene\n"
            "- road_and_traffic\n"
            "- buildings_and_structures\n"
            "- infrastructure_and_street_furniture\n"
            "- vegetation_ground_and_debris\n"
            "- people_vehicles_and_activities\n"
            "- viewpoint_and_layout / uncertainty_and_occlusions (only when it actually affects comparability)\n\n"
            "Output requirement: Output only a single JSON object (no markdown code blocks).\n\n"
            "JSON schema (field names must be strictly followed):\n"
            "{\n"
            '  "changes": [\n'
            "    {\n"
            '      "module": "module name",\n'
            '      "change_description": "Summarize the difference in 1-2 sentences (no attribution; prefer summary over verbatim repetition)",\n'
            '      "pre_evidence": "Evidence snippet extracted/close to original sentence from pre-disaster description (keep short, 10-50 characters)",\n'
            '      "post_evidence": "Evidence snippet extracted/close to original sentence from post-disaster description (keep short, 10-50 characters)"\n'
            "    }\n"
            "  ],\n"
            '  "no_change_notes": [\n'
            "    {\n"
            '      "module": "module name (must be in the module list)",\n'
            '      "component": "key module/object with no obvious change (optional; can omit)",\n'
            '      "note": "one sentence explaining why no change is considered (optional)",\n'
            '      "pre_evidence": "optional",\n'
            '      "post_evidence": "optional"\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Important rule<LOCAL_PATH>"
            "1) Only write points that are 'truly inconsistent'; when both sides say not seen/complete/none, do not treat as change.\n"
            "2) Evidence must come from input text, do not write external knowledge.\n"
            "3) When the same module has multiple independent differences, split into multiple entries.\n"
            "4) Please check module by module: especially do not miss overall_scene's weather/lighting,"
            "people_vehicles_and_activities's vehicle/person count and types,"
            "infrastructure_and_street_furniture's elements list changes, etc.\n"
            "5) **Mandatory coverage of all modules**: Each module in the module list above must appear at least onc<LOCAL_PATH>"
            "   - Either have at least 1 change entry for that module in changes;\n"
            "   - Or have at least 1 'no obvious change' note for that module in no_change_notes (and try to provide pre_evidence/post_evidence).\n"
            "   - No module is allowed to be absent.\n\n"
            "6) **Avoid redundancy**: no_change_notes can only be used for modules with 'no change';\n"
            "   - If a module already appears in changes (has changes), do not write it into no_change_notes.\n"
            "   - no_change_notes has at most 1 entry per module.\n\n"
            "Input as follow<LOCAL_PATH>"
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
            "You are a rigorous change extraction assistant, responsible only for extracting 'what changes occurred' from pre/post-disaster descriptions, "
            "without making disaster attribution judgments. Your output must be JSON."
        )
        user_prompt = self._build_change_extraction_prompt(pre_text, post_text, pair_meta)
        raw = self._generate_llm(
            system_prompt,
            user_prompt,
            max_new_tokens=self.max_new_tokens_step1,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
        )
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            # Step1 details，details JSON details。
            # details“details pass”：details/details JSON（details schema）。
            repair_system = "details JSON details。details JSON，details。"
            repair_user = (
                "details，details。\n"
                "details JSON details，details schema：\n"
                "{\n"
                '  "changes":[{"module":"","change_description":"","pre_evidence":"","post_evidence":""}],\n'
                '  "no_change_notes":[{"module":"","component":"","note":"","pre_evidence":"","post_evidence":""}]\n'
                "}\n"
                "details：\n"
                "1) details JSON details；\n"
                "2) details；\n"
                "3) details module details changes details，details no_change_notes；\n"
                "4) no_change_notes details module details 1 details。\n\n"
                f"[details]\n{_safe_str(raw)}\n"
            )
            repaired = self._generate_llm(
                repair_system,
                repair_user,
                max_new_tokens=900,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            obj = _extract_json_object(repaired)
            if not isinstance(obj, dict):
                return _heuristic_fallback_from_pair_meta(pair_meta)

        changes = obj.get("changes", [])
        if not isinstance(changes, list):
            return _heuristic_fallback_from_pair_meta(pair_meta)

        # details change：details schema + details id
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

            # details：details Step2(RAG) details component details（details）
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
        if cleaned:
            return cleaned

        # Fallback: if we have raw structured perception JSON, do a conservative diff-based extraction.
        fallback = _heuristic_fallback_from_pair_meta(pair_meta)
        if fallback:
            return fallback

        return cleaned

    # ------------------------------------------------------------------
    # Step 2：details RAG details
    # ------------------------------------------------------------------
    def enrich_changes_with_rag(
        self,
        changes: List[Dict[str, Any]],
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        details DamageFeatureRAG.search_for_detection_change，
        details“details/details”。
        """
        if not changes:
            return []

        pre_desc_global = None
        post_desc_global = None
        if isinstance(pair_meta, dict):
            pre_desc_global = _safe_str(pair_meta.get("pre_text_raw", ""))
            post_desc_global = _safe_str(pair_meta.get("post_text_raw", ""))

        enriched: List[Dict[str, Any]] = []
        for ch in changes:
            module = _safe_str(ch.get("module", ""))
            component = _safe_str(ch.get("component", ""))
            change_desc = _safe_str(ch.get("change_description", ""))
            # details pre_state/post_state，details
            pre_desc = pre_desc_global
            post_desc = post_desc_global

            # Step2 details/details（details），details module details
            extra_context_parts: List[str] = []
            if module:
                extra_context_parts.append(f"module={module}")
            extra_ctx = "; ".join(extra_context_parts) if extra_context_parts else ""

            hits = self.rag.search_for_detection_change(
                hazard_type=self.hazard_type,
                component=component,
                change_desc=change_desc,
                pre_desc=pre_desc,
                post_desc=post_desc,
                extra_context=extra_ctx,
                language="en",  # details（details），details "auto"
                top_k=int(self.rag_top_k),
            )

            ch2 = dict(ch)
            ch2["rag_hits"] = hits
            enriched.append(ch2)

        return enriched

    # ------------------------------------------------------------------
    # Step 3：details（details + RAG）
    # ------------------------------------------------------------------
    def _build_classification_prompt(
        self,
        enriched_changes: List[Dict[str, Any]],
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        meta_block = ""
        if isinstance(pair_meta, dict) and pair_meta:
            pre_date = _safe_str(pair_meta.get("pre_date", ""))
            post_date = _safe_str(pair_meta.get("post_date", ""))
            lat = _safe_str(pair_meta.get("lat", ""))
            lon = _safe_str(pair_meta.get("lon", ""))
            dist_m = _safe_str(pair_meta.get("dist_m", ""))
            pair_id = _safe_str(pair_meta.get("pair_id", ""))

            lines: List[str] = []
            if pair_id:
                lines.append(f"pair_id: {pair_id}")
            if pre_date:
                lines.append(f"Pre-disaster shooting date (pre_date): {pre_date}")
            if post_date:
                lines.append(f"Post-disaster shooting date (post_date): {post_date}")
            if lat and lon:
                lines.append(f"Latitude and longitude (lat, lon): ({lat}, {lon})")
            if dist_m:
                lines.append(f"Distance between shooting points (dist_m): {dist_m} m")

            if lines:
                meta_block = "[details]\n" + "\n".join(lines) + "\n\n"

        # details，details enriched_changes details JSON details。
        # details JSON details。
        changes_json = json.dumps(enriched_changes, ensure_ascii=False, indent=2)

        return (
            "You are a rigorous building and disaster analysis Detection Agent. The current disaster type of concern is: "
            f"{self.hazard_type} (hurricanes typically include combined effects of strong winds + heavy rainfall + inland flooding + storm surge + erosion, etc.).\n\n"
            "The previous step has extracted several 'change entries' from pre/post-disaster descriptions, "
            "and retrieved several disaster damage-related rules/case texts (RAG hits) for each change.\n\n"
            "Your task now i<LOCAL_PATH>"
            "Perform 'hierarchical labeling' for each change, output three categories instead of hard deletio<LOCAL_PATH>"
            "1) confirmed_disaster_damage: High confidence disaster-related, highly matching typical patterns/rules of the disaster type;\n"
            "2) likely_pseudo_change: High confidence non-disaster causes (must be able to point out clear non-disaster cause evidence, e.g., construction/season/viewpoint/vehicles, etc.);\n"
            "3) uncertain: There is indeed a change, but insufficient evidence to confirm whether it was caused by disaster.\n\n"
            "Decision strategy (please strictly follow):\n"
            "- If there is insufficient evidence, label as uncertain, do not directly classify as likely_pseudo_change;\n"
            "- Only when you can point out clear non-disaster causes from RAG hits or input evidence, can you label as likely_pseudo_change;\n"
            "- For changes highly consistent with typical patterns of the target disaster type, even if somewhat ambiguous, should be prioritized as confirmed_disaster_damage, and explain the basis.\n\n"
            "You also need to provide disaster_related for each change: likely | possible | unlikely | unknown, "
            "to express the strength of disaster correlation (and as a basis for post-processing correction).\n\n"
            "Special attentio<LOCAL_PATH>"
            "  - Comprehensively consider shooting date differences (seasonal changes, long-term construction, etc.) and rules provided by RAG, do not misjudge normal changes as disaster damage;\n"
            "  - **Rainfall/flooding/storm surge related signs are usually part of hurricane impact**: e.g., mud coverage, standing water/puddles, water stain boundaries, silt lines, scouring/erosion, ground subsidence, floating/accumulated debris, etc. Do not default these changes to pseudo;\n"
            "    If these water-related changes are consistent with flood/erosion/storm surge rules in RAG hits, should tend to judge as likely/possible;\n"
            "  - If RAG hits clearly state that a certain type of change is common in non-disaster causes, should tend to judge as pseudo_changes;\n"
            "  - If there is insufficient evidence, you can also give \"unknown\" or low confidence.\n\n"
            "Below are the input change entries and corresponding RAG hits (JSON array), field meaning<LOCAL_PATH>"
            "  - id: change entry identifier\n"
            "  - module: module to which the change belongs\n"
            "  - pre_evidence / post_evidence: evidence snippets extracted/close to original sentences from input descriptions (can be quoted)\n"
            "  - change_description: 1-2 sentence difference summary (no disaster attribution)\n"
            "  - rag_hits: several rules/cases retrieved from knowledge base for this change, containing text / source / meta\n\n"
            f"{meta_block}"
            "[Change entries and their RAG hits]\n"
            f"{changes_json}\n\n"
            "Please output a JSON object with the following structure (top-level keys must be strictly followed):\n"
            "{\n"
            '  "confirmed_disaster_damage": [\n'
            "    {\n"
            "      \"id\": 1,  // corresponding change entry id\n"
            "      \"component\": \"affected component or area\",\n"
            "      \"location\": \"location description\",\n"
            "      \"damage_type\": \"damage type, e.g.: roof blown off, exterior wall damage, glass broken, trees snapped, etc.\",\n"
            "      \"description\": \"brief text description of this damage, quoting key information from change_description and pre/post_state\",\n"
            "      \"evidence\": \"key basis supporting your judgment as disaster-related, can quote one or two key points from rag_hits\",\n"
            "      \"disaster_related\": \"one of: likely | possible | unlikely | unknown\",\n"
            "      \"confidence\": \"your confidence in this judgment, e.g.: high/medium/low, or 0-1 numeric value\",\n"
            "      \"source_change_id\": 1  // points to original change entry id\n"
            "    }\n"
            "  ],\n"
            '  "likely_pseudo_change": [\n'
            "    {\n"
            "      \"id\": 1,\n"
            "      \"affected_feature\": \"element considered as pseudo change/non-disaster factor\",\n"
            "      \"location\": \"location or area description where this pseudo change appears\",\n"
            "      \"category\": \"pseudo change type, e.g.: construction/renovation, seasonal change, viewpoint change, lighting/shadow change, vehicle/pedestrian change, etc.\",\n"
            "      \"reason\": \"brief reason explanation why considered as pseudo change or not caused by current disaster type\",\n"
            "      \"disaster_related\": \"one of: likely | possible | unlikely | unknown (usually unlikely or unknown)\",\n"
            "      \"confidence\": \"your confidence in this judgment\",\n"
            "      \"source_change_id\": 2  // points to original change entry id\n"
            "    }\n"
            "  ],\n"
            '  "uncertain": [\n'
            "    {\n"
            "      \"id\": 1,\n"
            "      \"component\": \"component or area where change occurred (short)\",\n"
            "      \"location\": \"location description\",\n"
            "      \"description\": \"change description (no disaster attribution)\",\n"
            "      \"why_uncertain\": \"why cannot confirm whether caused by disaster (insufficient evidence/ambiguous/possibly multiple causes)\",\n"
            "      \"disaster_related\": \"one of: likely | possible | unlikely | unknown (usually unknown or possible)\",\n"
            "      \"confidence\": \"your confidence in this judgment\",\n"
            "      \"source_change_id\": 3\n"
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Requirement<LOCAL_PATH>"
            "1. Output only a single JSON object, do not include any explanatory natural language;\n"
            "2. Field names must strictly follow the above structure, you can omit some fields, but do not add new top-level keys;\n"
            "3. If a certain category is completely empty, output an empty list.\n"
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
            "You are a rigorous building and disaster analysis Detection Agent, specialized in judging whether a single change is caused by a given disaster. "
            "Your output must be JSON."
        )
        user_prompt = self._build_classification_prompt(enriched_changes, pair_meta=pair_meta)
        raw = self._generate_llm(
            system_prompt,
            user_prompt,
            max_new_tokens=self.max_new_tokens_step3,
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

        # details（details likely/possible details pseudo_changes）
        obj = _normalize_step3_buckets(obj)

        vh_len = len(obj.get("confirmed_disaster_damage", []) or [])
        pc_len = len(obj.get("likely_pseudo_change", []) or [])
        uc_len = len(obj.get("uncertain", []) or [])
        obj["_stats"] = {
            "confirmed_disaster_damage_count": vh_len,
            "likely_pseudo_change_count": pc_len,
            "uncertain_count": uc_len,
            "change_count_input": len(enriched_changes),
        }
        if vh_len + pc_len <= 2:
            obj.setdefault(
                "_note",
                "details，details，details prompt details。",
            )
        return obj

    # ------------------------------------------------------------------
    # details：run
    # ------------------------------------------------------------------
    def run(
        self,
        pre_text: str,
        post_text: str,
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
        max_changes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        details：details
          1) details
          2) details RAG
          3) details+RAG details
        """
        pre_text = _safe_str(pre_text)
        post_text = _safe_str(post_text)

        # Step 1: details
        changes = self.extract_changes(pre_text, post_text, pair_meta=pair_meta)

        # details/details：details N details RAG + details
        if max_changes is not None:
            try:
                n = int(max_changes)
            except Exception:
                n = 0
            if n > 0:
                changes = changes[:n]

        # Step 2: details RAG details
        enriched_changes = self.enrich_changes_with_rag(changes, pair_meta=pair_meta)

        # Step 3: details enriched_changes details
        result = self.classify_with_rag(enriched_changes, pair_meta=pair_meta)

        # details，details（details/details）
        result["_intermediate"] = {
            "changes": changes,
            "enriched_changes": enriched_changes,
        }
        return result