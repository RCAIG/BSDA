from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from schemas.state import PipelineState
from orchestrator.langgraph.utils import (
    apply_critic_feedback_to_assessment,
    apply_critic_feedback_to_detection,
    ensure_repo_root_on_path,
)


def assessment_node(state: PipelineState) -> PipelineState:
    """
    Assessment node:
    - Runs Assessment (run_assessment) if rerun_assessment=True.
    - If revision_count>0 and critic_output exists, inject critic feedback into detection/assessment _meta.
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    if not bool(state.get("rerun_assessment")):
        return state

    from Agents.AssessmentAgent.assessment_agent import run_assessment

    detection_output: Dict[str, Any] = state.get("detection_output", {})  # type: ignore[assignment]
    critic_output: Dict[str, Any] = state.get("critic_output", {}) if isinstance(state.get("critic_output"), dict) else {}

    # If this is a revision round, attach critic feedback so Assessment can react.
    if int(state.get("revision_count", 0)) > 0 and critic_output:
        detection_output = apply_critic_feedback_to_detection(detection_output, critic_output)
        prev_assessment = state.get("assessment_output")
        if isinstance(prev_assessment, dict):
            prev_assessment = apply_critic_feedback_to_assessment(prev_assessment, critic_output)
        else:
            prev_assessment = None
        assessment_output: Dict[str, Any] = run_assessment(
            detection_output,
            agent=state.get("assessment_agent"),
            assessment_input=prev_assessment,
        )
    else:
        assessment_output = run_assessment(
            detection_output,
            agent=state.get("assessment_agent"),
        )

    assessment_output.setdefault("_meta", {})["revision_round"] = int(state.get("revision_count", 0))
    state["assessment_output"] = assessment_output
    state["detection_output"] = detection_output

    state["rerun_assessment"] = False
    return state


