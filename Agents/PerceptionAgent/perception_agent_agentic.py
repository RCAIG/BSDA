#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PerceptionAgent/perception_agent_agentic.py

Perception details agent loop（details）：
- Deliberation: details/details
- Tool Calling: details quality_check / align / describe
- Observation: details
- Memory Update: details memory trace

details：
- details“details agent loop”，details；
- details Detection details ReAct details，details Perception details。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

try:
    from tools.image_alignment import match_and_crop_pre_to_post as _align_tool_fn
except Exception:
    _align_tool_fn = None
try:
    from tools.reverse_geocode import reverse_geocode_location as _reverse_geocode_fn
except Exception:
    _reverse_geocode_fn = None

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def _parse_dt(dt_text: Any) -> Optional[datetime]:
    s = _safe_str(dt_text)
    if not s:
        return None
    s = s.replace("Z", "+00:00")
    for cand in [s]:
        try:
            return datetime.fromisoformat(cand)
        except Exception:
            pass
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%Y/%m/%d %H:%M:%S",
        "%Y/%m/%d %H:%M",
        "%Y/%m/%d",
    ]
    for fmt in fmts:
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            pass
    return None


def _infer_season(dt_obj: Optional[datetime], lat: Optional[float]) -> str:
    if dt_obj is None:
        return "unknown"
    m = int(dt_obj.month)
    north = lat is None or lat >= 0
    if north:
        if m in (12, 1, 2):
            return "winter"
        if m in (3, 4, 5):
            return "spring"
        if m in (6, 7, 8):
            return "summer"
        return "autumn"
    # Southern hemisphere reversed seasons.
    if m in (12, 1, 2):
        return "summer"
    if m in (3, 4, 5):
        return "autumn"
    if m in (6, 7, 8):
        return "winter"
    return "spring"


def _infer_day_part(dt_obj: Optional[datetime]) -> str:
    if dt_obj is None:
        return "unknown"
    h = int(dt_obj.hour)
    if 5 <= h < 8:
        return "dawn_dusk"
    if 8 <= h < 17:
        return "day"
    if 17 <= h < 20:
        return "dawn_dusk"
    return "night"
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


@dataclass
class PerceptionMemoryStep:
    iteration: int
    deliberation: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: Dict[str, Any] = field(default_factory=dict)
    memory_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerceptionMemory:
    goal: str = "Generate reliable pre/post scene descriptions for downstream agents"
    need_alignment: bool = False
    quality_ok: bool = True
    stop_reason: str = ""
    steps: list[PerceptionMemoryStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "need_alignment": self.need_alignment,
            "quality_ok": self.quality_ok,
            "stop_reason": self.stop_reason,
            "steps": [asdict(s) for s in self.steps],
        }


@dataclass
class PerceptionAgentAgentic:
    """
    details Perception agent：
    - details；
    - details，details。
    """

    # details（details，details）
    describe_fn: Callable[[Path], Dict[str, Any]]
    quality_check_fn: Optional[Callable[[Path, Path], Dict[str, Any]]] = None
    align_fn: Optional[Callable[[Path, Path], Dict[str, Any]]] = None

    def memory_schema(self) -> Dict[str, Any]:
        return PerceptionMemory().to_dict()

    def run(
        self,
        *,
        pre_image: Path,
        post_image: Path,
        force_alignment: bool = False,
        pair_meta: Optional[Dict[str, Any]] = None,
        pre_time: Optional[str] = None,
        post_time: Optional[str] = None,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        geocode_language: str = "en",
    ) -> Dict[str, Any]:
        """
        details perception agent loop。
        """
        memory = PerceptionMemory()
        memory.need_alignment = bool(force_alignment)
        meta = pair_meta or {}
        pre_time_val = _safe_str(pre_time) or _safe_str(meta.get("pre_time") or meta.get("pre_date"))
        post_time_val = _safe_str(post_time) or _safe_str(meta.get("post_time") or meta.get("post_date"))
        lat_val = _safe_float(lat if lat is not None else (meta.get("lat") or meta.get("latitude")))
        lon_val = _safe_float(lon if lon is not None else (meta.get("lon") or meta.get("longitude")))

        # Step 1: Deliberation + quality check
        quality_obs: Dict[str, Any] = {"quality_ok": True, "reason": "quality_check_skipped"}
        if self.quality_check_fn is not None:
            quality_obs = self.quality_check_fn(pre_image, post_image) or {}
            memory.quality_ok = bool(quality_obs.get("quality_ok", True))
            # details，details
            if bool(quality_obs.get("suggest_alignment", False)):
                memory.need_alignment = True
        memory.steps.append(
            PerceptionMemoryStep(
                iteration=1,
                deliberation="Assess pair quality and determine whether alignment is needed.",
                action="quality_check" if self.quality_check_fn is not None else "skip_quality_check",
                action_input={"pre_image": str(pre_image), "post_image": str(post_image)},
                observation=quality_obs,
                memory_update={
                    "quality_ok": memory.quality_ok,
                    "need_alignment": memory.need_alignment,
                },
            )
        )

        # Step 2: Optional alignment
        align_obs: Dict[str, Any] = {"applied": False}
        aligned_pre = pre_image
        aligned_post = post_image
        align_callable: Optional[Callable[[Path, Path], Dict[str, Any]]] = self.align_fn or _align_tool_fn
        if memory.need_alignment and align_callable is not None:
            align_obs = align_callable(pre_image, post_image) or {}
            aligned_pre = Path(_safe_str(align_obs.get("aligned_pre_path")) or str(pre_image))
            aligned_post = Path(_safe_str(align_obs.get("aligned_post_path")) or str(post_image))
        memory.steps.append(
            PerceptionMemoryStep(
                iteration=2,
                deliberation="Apply alignment if needed to improve cross-time comparability.",
                action="align" if (memory.need_alignment and align_callable is not None) else "skip_align",
                action_input={"pre_image": str(pre_image), "post_image": str(post_image)},
                observation=align_obs,
                memory_update={
                    "aligned_pre": str(aligned_pre),
                    "aligned_post": str(aligned_post),
                },
            )
        )
        alignment_used = bool(align_obs.get("applied", False))
        alignment_summary = {
            "used": alignment_used,
            "method": _safe_str(align_obs.get("method")) if alignment_used else "",
            "reason": _safe_str(align_obs.get("reason")) if not alignment_used else "",
            "failed": bool(align_obs.get("alignment_failed", not alignment_used)),
            "quality": _safe_str(align_obs.get("alignment_quality")),
            "confidence": align_obs.get("alignment_confidence"),
            "fallback_reason": _safe_str(align_obs.get("fallback_reason")),
        }

        pre_dt = _parse_dt(pre_time_val)
        post_dt = _parse_dt(post_time_val)
        time_context = {
            "pre_time_input": pre_time_val,
            "post_time_input": post_time_val,
            "pre_season": _infer_season(pre_dt, lat_val),
            "post_season": _infer_season(post_dt, lat_val),
            "pre_day_part": _infer_day_part(pre_dt),
            "post_day_part": _infer_day_part(post_dt),
        }
        location_context: Dict[str, Any] = {
            "lat": lat_val,
            "lon": lon_val,
            "display_name": "",
            "address": {},
            "source": "",
            "geocode_success": False,
        }
        if lat_val is not None and lon_val is not None and _reverse_geocode_fn is not None:
            geo_out = _reverse_geocode_fn(lat_val, lon_val, language=geocode_language)
            location_context.update(
                {
                    "display_name": _safe_str(geo_out.get("display_name")),
                    "address": geo_out.get("address", {}) if isinstance(geo_out.get("address"), dict) else {},
                    "source": _safe_str(geo_out.get("source")),
                    "geocode_success": bool(geo_out.get("success", False)),
                    "geocode_error": _safe_str(geo_out.get("error")),
                }
            )

        # Step 3: Describe pre/post images
        pre_desc = self.describe_fn(aligned_pre) or {}
        post_desc = self.describe_fn(aligned_post) or {}
        if isinstance(pre_desc, dict):
            pre_desc["capture_context"] = {
                "capture_time": pre_time_val,
                "season": time_context["pre_season"],
                "day_part": time_context["pre_day_part"],
                "location": location_context.get("display_name", ""),
            }
        if isinstance(post_desc, dict):
            post_desc["capture_context"] = {
                "capture_time": post_time_val,
                "season": time_context["post_season"],
                "day_part": time_context["post_day_part"],
                "location": location_context.get("display_name", ""),
            }
        memory.steps.append(
            PerceptionMemoryStep(
                iteration=3,
                deliberation="Generate structured descriptions for pre and post images.",
                action="describe",
                action_input={"pre_image": str(aligned_pre), "post_image": str(aligned_post)},
                observation={
                    "pre_desc_keys": list(pre_desc.keys()) if isinstance(pre_desc, dict) else [],
                    "post_desc_keys": list(post_desc.keys()) if isinstance(post_desc, dict) else [],
                },
                memory_update={
                    "description_ready": True,
                },
            )
        )

        memory.stop_reason = "descriptions_generated"

        return {
            "pre_description": pre_desc,
            "post_description": post_desc,
            "_agentic_meta": {
                "loop_pattern": ["deliberation", "tool_call", "observation", "memory_update"],
                "alignment_used": alignment_used,
                "alignment_summary": alignment_summary,
                "time_context": time_context,
                "location_context": location_context,
                "memory_schema": self.memory_schema(),
                "memory_trace": memory.to_dict(),
            },
        }

