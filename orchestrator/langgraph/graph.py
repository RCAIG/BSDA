from __future__ import annotations

from typing import Callable

from langgraph.graph import END, START, StateGraph

from schemas.state import PipelineState
from orchestrator.langgraph.nodes import (
    apply_critic_node,
    assessment_node,
    critic_node,
    decide_node,
    end_node,
    detection_node,
    init_node,
    perception_node,
)


def _route_after_apply(state: PipelineState) -> str:
    if bool(state.get("rerun_perception")):
        return "perception"
    if bool(state.get("rerun_detection")):
        return "detection"
    if bool(state.get("rerun_assessment")):
        return "assessment"
    return "critic"


def build_app() -> Callable[[PipelineState], PipelineState]:
    """
    Build and compile the LangGraph app.
    """
    g: StateGraph = StateGraph(PipelineState)  # type: ignore[arg-type]

    g.add_node("init", init_node)
    g.add_node("perception", perception_node)
    g.add_node("detection", detection_node)
    g.add_node("assessment", assessment_node)
    g.add_node("critic", critic_node)
    g.add_node("decide", decide_node)
    g.add_node("apply_critic", apply_critic_node)
    g.add_node("end", end_node)

    g.add_edge(START, "init")
    g.add_edge("init", "perception")
    g.add_edge("perception", "detection")
    g.add_edge("detection", "assessment")
    g.add_edge("assessment", "critic")
    g.add_edge("critic", "decide")

    g.add_conditional_edges(
        "decide",
        lambda s: s.get("next_route", "accept"),
        {
            "accept": "end",
            "flag": "end",
            "exhausted": "end",
            "revise": "apply_critic",
        },
    )

    g.add_conditional_edges(
        "apply_critic",
        _route_after_apply,
        {
            "perception": "perception",
            "detection": "detection",
            "assessment": "assessment",
            "critic": "critic",
        },
    )

    g.add_edge("end", END)

    return g.compile()


