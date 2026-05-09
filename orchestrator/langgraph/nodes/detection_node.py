from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path


def detection_node(state: PipelineState) -> PipelineState:
    """
    Detection node:
    - Runs DetectionAgent if rerun_detection=True (first pass or revision).
    - Uses perception_output summaries/raw JSON as inputs.
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    if not bool(state.get("rerun_detection")):
        return state

    detection_agent = state.get("detection_agent")
    if detection_agent is None:
        raise RuntimeError("detection_agent is not initialized")

    perception_output: Dict[str, Any] = state.get("perception_output", {})  # type: ignore[assignment]
    pre_raw = perception_output.get("pre_scene", {}).get("raw", {})
    post_raw = perception_output.get("post_scene", {}).get("raw", {})

    pre_text = json.dumps(pre_raw, ensure_ascii=False, indent=2) if pre_raw else json.dumps(perception_output.get("pre_scene", {}), ensure_ascii=False)
    post_text = json.dumps(post_raw, ensure_ascii=False, indent=2) if post_raw else json.dumps(perception_output.get("post_scene", {}), ensure_ascii=False)

    detection_output: Dict[str, Any] = detection_agent.run(
        pre_text,
        post_text,
        pair_meta=None,
        max_changes=None,
    )
    detection_output.setdefault("_meta", {})["revision_round"] = int(state.get("revision_count", 0))
    state["detection_output"] = detection_output

    state["rerun_detection"] = False
    return state


