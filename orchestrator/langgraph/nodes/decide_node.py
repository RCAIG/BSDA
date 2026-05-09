from __future__ import annotations

from pathlib import Path

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path, extract_judgement


def decide_node(state: PipelineState) -> PipelineState:
    """
    Decide next step based on Critic judgement and revision_count/max_revisions.
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    critic_output = state.get("critic_output") if isinstance(state.get("critic_output"), dict) else {}
    judgement = extract_judgement(critic_output or {})

    max_revisions = int(state.get("max_revisions", 1))
    revision_count = int(state.get("revision_count", 0))

    if judgement == "accept":
        state["final_status"] = "accepted"
        state["next_route"] = "accept"
        return state

    if judgement == "flag_for_human":
        state["final_status"] = "flagged"
        # annotate assessment meta for downstream consumers
        ao = state.get("assessment_output")
        if isinstance(ao, dict):
            ao.setdefault("_meta", {})["flagged_for_human"] = True
            ao.setdefault("_meta", {})["flag_reason"] = (critic_output or {}).get("summary", {}).get("revise_reason") or "Critic flagged for human review"
        state["next_route"] = "flag"
        return state

    if judgement == "revise":
        if revision_count >= max_revisions:
            state["final_status"] = "exhausted"
            ao = state.get("assessment_output")
            if isinstance(ao, dict):
                ao.setdefault("_meta", {})["max_revisions_reached"] = True
                ao.setdefault("_meta", {})["max_revisions_limit"] = max_revisions
            state["next_route"] = "exhausted"
            return state

        # start next revision round
        state["revision_count"] = revision_count + 1
        state["next_route"] = "revise"
        return state

    # Unknown judgement -> treat as accept
    state["final_status"] = "accepted"
    state["next_route"] = "accept"
    return state

