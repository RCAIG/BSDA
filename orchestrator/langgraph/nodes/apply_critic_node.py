from __future__ import annotations

from pathlib import Path

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path, issues_related_agents


def apply_critic_node(state: PipelineState) -> PipelineState:
    """
    Apply critic suggestions:
    - Determine which agent(s) to rerun next by reading critic issues.
    - If schema does not provide agent targets, fall back to rerun Detection + Assessment on revise.
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    critic_output = state.get("critic_output") if isinstance(state.get("critic_output"), dict) else {}
    related = [x.lower() for x in issues_related_agents(critic_output or {})]

    needs_perception = any("perception" in x.lower() for x in related)
    needs_detection = any("detection" in x.lower() for x in related)
    needs_assessment = any("assessment" in x.lower() for x in related)

    # If Critic output doesn't indicate targets, default to rerun detection+assessment (most safe).
    if not (needs_perception or needs_detection or needs_assessment):
        needs_detection = True
        needs_assessment = True

    if needs_perception:
        state["rerun_perception"] = True
        state["rerun_detection"] = True
        state["rerun_assessment"] = True
    elif needs_detection:
        state["rerun_perception"] = False
        state["rerun_detection"] = True
        state["rerun_assessment"] = True
    elif needs_assessment:
        state["rerun_perception"] = False
        state["rerun_detection"] = False
        state["rerun_assessment"] = True

    return state


