from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path


def critic_node(state: PipelineState) -> PipelineState:
    """
    Critic node:
    - Runs CriticAgent and appends to critic_history
    - Records round snapshot into revision_history for debugging/audit
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    critic_agent = state.get("critic_agent")
    if critic_agent is None:
        raise RuntimeError("critic_agent is not initialized")

    perception_output: Dict[str, Any] = state.get("perception_output", {})  # type: ignore[assignment]
    detection_output: Dict[str, Any] = state.get("detection_output", {})  # type: ignore[assignment]
    assessment_output: Dict[str, Any] = state.get("assessment_output", {})  # type: ignore[assignment]

    critic_output: Dict[str, Any] = critic_agent.run(
        perception_output=perception_output,
        detection_output=detection_output,
        assessment_output=assessment_output,
        rules_summary=None,
    )
    state["critic_output"] = critic_output

    critic_hist = state.setdefault("critic_history", [])
    if isinstance(critic_hist, list):
        critic_hist.append(critic_output)

    rev_hist = state.setdefault("revision_history", [])
    if isinstance(rev_hist, list):
        rev_hist.append(
            {
                "revision_round": int(state.get("revision_count", 0)),
                "perception_output": perception_output,
                "detection_output": detection_output,
                "assessment_output": assessment_output,
                "critic_output": critic_output,
            }
        )

    return state


