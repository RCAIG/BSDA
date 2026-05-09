from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, TypedDict


FinalStatus = Literal["in_progress", "accepted", "flagged", "exhausted"]
Judgement = Literal["accept", "revise", "flag_for_human"]
DecideRoute = Literal["accept", "revise", "flag", "exhausted"]


class PipelineState(TypedDict, total=False):
    """
    Global LangGraph runtime state.

    A small number of runtime objects may be stored here because the current
    flow uses in-memory invocation. For persistence or distributed execution,
    move runtime objects to an external dependency injection layer.
    """

    # ---- inputs ----
    pair_id: str
    pre_path: str
    post_path: str
    pre_image_path: Optional[str]
    post_image_path: Optional[str]
    use_rag: int
    use_llm: int
    max_revisions: int

    # ---- runtime (non-serializable is ok for now) ----
    detection_agent: Any
    assessment_agent: Any
    critic_agent: Any
    vlm_client: Any

    # ---- agent outputs ----
    perception_output: Dict[str, Any]
    detection_output: Dict[str, Any]
    assessment_output: Dict[str, Any]
    critic_output: Dict[str, Any]

    # ---- revision control ----
    revision_count: int
    final_status: FinalStatus
    critic_history: List[Dict[str, Any]]
    revision_history: List[Dict[str, Any]]

    # flags to control reruns in next cycle
    rerun_perception: bool
    rerun_detection: bool
    rerun_assessment: bool

    # final aggregated output (for external callers / CLI)
    final_report: Dict[str, Any]

    # internal routing label (set by DecideNode)
    next_route: DecideRoute


