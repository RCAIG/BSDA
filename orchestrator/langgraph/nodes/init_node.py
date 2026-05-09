from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from schemas.state import PipelineState
from orchestrator.langgraph.utils import ensure_repo_root_on_path, load_text_or_json


def init_node(state: PipelineState) -> PipelineState:
    """
    Initialize state:
    - ensure repo root on sys.path
    - load perception outputs from pre/post JSON files
    - init agent instances if missing
    """
    # Ensure repo root on path so "Agents.*" imports work consistently
    ensure_repo_root_on_path(Path(__file__).resolve())

    pre_path = Path(str(state["pre_path"]))
    post_path = Path(str(state["post_path"]))
    pre_text, pre_obj = load_text_or_json(pre_path)
    post_text, post_obj = load_text_or_json(post_path)

    perception_output: Dict[str, Any] = {
        "_meta": {"source": "perception_files", "pre_path": str(pre_path), "post_path": str(post_path)},
        "pre_scene": {"raw": pre_obj, "summary": pre_text[:1000]},
        "post_scene": {"raw": post_obj, "summary": post_text[:1000]},
    }
    state["perception_output"] = perception_output

    # Initialize history and control flags
    state.setdefault("revision_count", 0)
    state.setdefault("critic_history", [])
    state.setdefault("revision_history", [])
    state.setdefault("final_status", "in_progress")

    # First pass always runs Detection+Assessment; Perception is loaded from files.
    state["rerun_perception"] = False
    state["rerun_detection"] = True
    state["rerun_assessment"] = True

    # Initialize agents (if caller did not inject them)
    if state.get("detection_agent") is None:
        from Agents.DetectionAgent.detection_agent import DetectionAgent
        from Agents.DetectionAgent import config as det_cfg
        from Agents.DetectionAgent.rag import DamageFeatureRAG

        corpus = str(getattr(det_cfg, "RAG_CORPUS", "gov")).lower()
        if corpus == "street":
            embed_model = str(getattr(det_cfg, "RAG_EMBED_MODEL_STREET", "sentence-transformers/all-MiniLM-L6-v2"))
        else:
            corpus = "gov"
            embed_model = str(getattr(det_cfg, "RAG_EMBED_MODEL_GOV", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"))

        rag = DamageFeatureRAG(
            artifacts_dir=det_cfg.RAG_ARTIFACTS_DIR,
            corpus=corpus,
            embed_model=embed_model,
        )
        state["detection_agent"] = DetectionAgent(
            model_path=det_cfg.LOCAL_LLM_MODEL_PATH,
            rag=rag,
            max_new_tokens=int(getattr(det_cfg, "MAX_NEW_TOKENS", 800)),
            max_new_tokens_step1=int(getattr(det_cfg, "MAX_NEW_TOKENS_STEP1", getattr(det_cfg, "MAX_NEW_TOKENS", 800))),
            max_new_tokens_step3=int(getattr(det_cfg, "MAX_NEW_TOKENS_STEP3", getattr(det_cfg, "MAX_NEW_TOKENS", 800))),
            rag_top_k=int(getattr(det_cfg, "RAG_TOP_K", 5)),
        )

    if state.get("assessment_agent") is None:
        from Agents.AssessmentAgent.assessment_agent import get_assessment_agent

        state["assessment_agent"] = get_assessment_agent()

    if state.get("critic_agent") is None:
        from Agents.CriticAgent.critic_agent import CriticAgent
        import Agents.CriticAgent.config as critic_cfg

        state["critic_agent"] = CriticAgent(
            use_rag=bool(int(state.get("use_rag", 0))),
            rag_top_k=int(getattr(critic_cfg, "DEFAULT_RAG_TOP_K", 5)),
            use_llm=bool(int(state.get("use_llm", 1))),
        )

    # Optional VLM client is expensive; only use if injected by caller (e.g., main_revision did).
    # state["vlm_client"] may be None.
    return state


