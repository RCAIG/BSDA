from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path


def end_node(state: PipelineState) -> PipelineState:
    """
    End node: build a compact final summary under state["final_report"].
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    final_report: Dict[str, Any] = {
        "pair_id": state.get("pair_id", ""),
        "final_status": state.get("final_status", "in_progress"),
        "revision_count": int(state.get("revision_count", 0)),
        "perception_output": state.get("perception_output"),
        "detection_output": state.get("detection_output"),
        "assessment_output": state.get("assessment_output"),
        "critic_output": state.get("critic_output"),
        "critic_history": state.get("critic_history", []),
    }
    state["final_report"] = final_report  # type: ignore[typeddict-item]
    return state


