#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/revision_orchestrator.py

Revision orchestrator that implements automatic multi-round revision based on Critic feedback.

Features:
- Supports max_revisions parameter (0/1/2)
- Automatically re-runs Perception/Detection/Assessment based on Critic's fix_by_agent suggestions
- Tracks revision history in _meta fields
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import agents (using direct imports to avoid subprocess overhead)
import sys

# Add repo root to path
_here = Path(__file__).resolve().parent
_repo_root = None
for p in [_here] + list(_here.parents):
    if (p / "models").exists() and (p / "RAG").exists():
        _repo_root = p
        break
if _repo_root and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Import agents
try:
    from Agents.DetectionAgent.detection_agent import DetectionAgent
    from Agents.DetectionAgent import config as det_cfg
    from Agents.DetectionAgent.rag import DamageFeatureRAG
except ImportError:
    DetectionAgent = None  # type: ignore[assignment]
    det_cfg = None
    DamageFeatureRAG = None

try:
    from Agents.AssessmentAgent.assessment_agent import get_assessment_agent, run_assessment
except ImportError:
    get_assessment_agent = None  # type: ignore[assignment]
    run_assessment = None  # type: ignore[assignment]

try:
    from Agents.CriticAgent.critic_agent import CriticAgent
    import Agents.CriticAgent.config as critic_cfg
except ImportError:
    CriticAgent = None  # type: ignore[assignment]
    critic_cfg = None

# Import PerceptionAgent components for revision
try:
    from Agents.PerceptionAgent.main import VLMClient, build_single_image_prompt, try_parse_json_loose, load_image
    from PIL import Image
except ImportError:
    VLMClient = None  # type: ignore[assignment]
    build_single_image_prompt = None  # type: ignore[assignment]
    try_parse_json_loose = None  # type: ignore[assignment]
    load_image = None  # type: ignore[assignment]
    Image = None


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _apply_critic_suggestions_to_assessment(
    assessment_output: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inject Critic feedback into Assessment output's _meta for next revision round.
    This allows Assessment to read and respond to Critic's suggestions.
    """
    revised = dict(assessment_output)  # Shallow copy
    
    meta = revised.setdefault("_meta", {})
    feedback_history = meta.setdefault("critic_feedback_history", [])
    
    # Extract key information from Critic
    critic_summary = critic_output.get("summary", {})
    critic_rec = critic_output.get("critic_recommendation", {})
    detected_issues = critic_output.get("detected_issues", [])
    
    # Record this round's feedback
    feedback_entry = {
        "summary": critic_summary,
        "recommendation": critic_rec,
        "issues": detected_issues,
    }
    feedback_history.append(feedback_entry)
    
    # Add convenient top-level hints for Assessment to read
    meta["critic_last_recommended_grade"] = critic_rec.get("proposed_grade")
    meta["critic_last_recommended_confidence"] = critic_rec.get("proposed_confidence")
    meta["critic_last_revise_reason"] = critic_summary.get("revise_reason")
    meta["critic_last_overall_judgement"] = critic_summary.get("overall_judgement")
    
    # Extract issues that specifically target Assessment
    assessment_issues = [
        it for it in detected_issues
        if isinstance(it, dict) and it.get("fix_by_agent") == "Assessment"
    ]
    meta["critic_assessment_issues"] = assessment_issues
    
    return revised


def _apply_critic_suggestions_to_detection(
    detection_output: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inject Critic feedback into Detection output's _meta for next revision round.
    """
    revised = dict(detection_output)
    
    meta = revised.setdefault("_meta", {})
    feedback_history = meta.setdefault("critic_feedback_history", [])
    
    detected_issues = critic_output.get("detected_issues", [])
    feedback_entry = {
        "issues": detected_issues,
        "summary": critic_output.get("summary", {}),
    }
    feedback_history.append(feedback_entry)
    
    # Extract issues that specifically target Detection
    detection_issues = [
        it for it in detected_issues
        if isinstance(it, dict) and it.get("fix_by_agent") == "Detection"
    ]
    meta["critic_detection_issues"] = detection_issues
    
    return revised


def _apply_critic_suggestions_to_perception(
    perception_output: Dict[str, Any],
    critic_output: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Inject Critic feedback into Perception output's _meta for next revision round.
    Note: Perception revision requires re-running the VLM on images, which is handled separately.
    This function just marks the perception output as needing revision.
    """
    revised = dict(perception_output)
    
    meta = revised.setdefault("_meta", {})
    feedback_history = meta.setdefault("critic_feedback_history", [])
    
    detected_issues = critic_output.get("detected_issues", [])
    feedback_entry = {
        "issues": detected_issues,
        "summary": critic_output.get("summary", {}),
    }
    feedback_history.append(feedback_entry)
    
    # Extract issues that specifically target Perception
    perception_issues = [
        it for it in detected_issues
        if isinstance(it, dict) and it.get("fix_by_agent") == "Perception"
    ]
    meta["critic_perception_issues"] = perception_issues
    meta["needs_perception_revision"] = len(perception_issues) > 0
    
    return revised


def run_pipeline_with_revision(
    *,
    pre_path: Path,
    post_path: Path,
    pair_id: str = "",
    max_revisions: int = 1,
    use_rag: int = 0,
    use_llm: int = 1,
    # Image paths for Perception revision (optional)
    pre_image_path: Optional[Path] = None,
    post_image_path: Optional[Path] = None,
    # Agent instances (optional, will be created if None)
    detection_agent: Optional[DetectionAgent] = None,
    assessment_agent: Optional[Any] = None,
    critic_agent: Optional[CriticAgent] = None,
    vlm_client: Optional[Any] = None,  # Optional pre-initialized VLMClient for Perception revision
) -> Dict[str, Any]:
    """
    Run the full pipeline (Perception -> Detection -> Assessment -> Critic) with automatic revision.
    
    Args:
        pre_path: Path to pre-disaster Perception JSON description
        post_path: Path to post-disaster Perception JSON description
        pair_id: Optional pair ID for output file naming
        max_revisions: Maximum number of revision rounds (0 = no revision, 1 = one revision, 2 = two revisions)
        use_rag: Whether to enable RAG for Critic (0/1)
        use_llm: Whether to enable LLM for Critic (0/1)
        detection_agent: Optional pre-initialized DetectionAgent (for batch processing efficiency)
        assessment_agent: Optional pre-initialized AssessmentAgent (for batch processing efficiency)
        critic_agent: Optional pre-initialized CriticAgent (for batch processing efficiency)
    
    Returns:
        Dict with keys:
        - perception_output: Final Perception output (may be revised)
        - detection_output: Final Detection output (may be revised)
        - assessment_output: Final Assessment output (may be revised)
        - critic_output: Final Critic output
        - revision_count: Number of revision rounds performed
        - revision_history: List of all round outputs (optional, for debugging)
    """
    revision_count = 0
    revision_history: List[Dict[str, Any]] = []
    
    # Load initial Perception outputs
    try:
        pre_text = pre_path.read_text(encoding="utf-8", errors="replace")
        post_text = post_path.read_text(encoding="utf-8", errors="replace")
        try:
            pre_obj = json.loads(pre_text)
            post_obj = json.loads(post_text)
        except Exception:
            pre_obj = {"raw_text": pre_text}
            post_obj = {"raw_text": post_text}
    except Exception as e:
        raise RuntimeError(f"Failed to load Perception outputs from {pre_path} / {post_path}: {e}")
    
    perception_output: Dict[str, Any] = {
        "_meta": {"source": "perception_files", "pre_path": str(pre_path), "post_path": str(post_path)},
        "pre_scene": {"raw": pre_obj, "summary": pre_text[:1000]},
        "post_scene": {"raw": post_obj, "summary": post_text[:1000]},
    }
    
    # Store pre_text and post_text as variables for use in Detection
    pre_text_original = pre_text
    post_text_original = post_text
    
    # Initialize agents if not provided
    if detection_agent is None and DetectionAgent is not None and det_cfg is not None:
        if DamageFeatureRAG is not None:
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
            detection_agent = DetectionAgent(
                model_path=det_cfg.LOCAL_LLM_MODEL_PATH,
                rag=rag,
                max_new_tokens=int(getattr(det_cfg, "MAX_NEW_TOKENS", 800)),
                max_new_tokens_step1=int(getattr(det_cfg, "MAX_NEW_TOKENS_STEP1", getattr(det_cfg, "MAX_NEW_TOKENS", 800))),
                max_new_tokens_step3=int(getattr(det_cfg, "MAX_NEW_TOKENS_STEP3", getattr(det_cfg, "MAX_NEW_TOKENS", 800))),
                rag_top_k=int(getattr(det_cfg, "RAG_TOP_K", 5)),
            )
        else:
            raise RuntimeError("DamageFeatureRAG not available")
    
    if assessment_agent is None and get_assessment_agent is not None:
        assessment_agent = get_assessment_agent()
    
    if critic_agent is None and CriticAgent is not None:
        critic_agent = CriticAgent(
            use_rag=bool(use_rag),
            rag_top_k=int(getattr(critic_cfg, "DEFAULT_RAG_TOP_K", 5)) if critic_cfg else 5,
            use_llm=bool(use_llm),
        )
    
    # Main revision loop
    while True:
        # 1) Run Detection (if needed or first round)
        if revision_count == 0 or (revision_count > 0 and detection_agent is not None):
            if detection_agent is None:
                raise RuntimeError("DetectionAgent not available")
            
            # Use current pre_text and post_text (may have been updated by Perception revision)
            # Check if Perception was revised in a previous iteration
            current_pre_text = pre_text_original
            current_post_text = post_text_original
            
            # If Perception was revised, use the revised JSON text
            if perception_output.get("_meta", {}).get("revision_round"):
                pre_scene_raw = perception_output.get("pre_scene", {}).get("raw", {})
                post_scene_raw = perception_output.get("post_scene", {}).get("raw", {})
                if pre_scene_raw and post_scene_raw:
                    current_pre_text = json.dumps(pre_scene_raw, ensure_ascii=False, indent=2)
                    current_post_text = json.dumps(post_scene_raw, ensure_ascii=False, indent=2)
            
            detection_output: Dict[str, Any] = detection_agent.run(
                current_pre_text,
                current_post_text,
                pair_meta=None,
                max_changes=None,
            )
            
            # Mark revision round in detection output
            detection_output.setdefault("_meta", {})["revision_round"] = revision_count
        
        # 2) Run Assessment (if needed or first round)
        if revision_count == 0 or (revision_count > 0 and assessment_agent is not None and run_assessment is not None):
            if run_assessment is None:
                raise RuntimeError("run_assessment not available")
            
            # Apply revision hints if this is a revision round
            if revision_count > 0:
                detection_output = _apply_critic_suggestions_to_detection(detection_output, critic_output)
            
            assessment_output: Dict[str, Any] = run_assessment(
                detection_output,
                agent=assessment_agent,
            )
            
            # Mark revision round in assessment output
            assessment_output.setdefault("_meta", {})["revision_round"] = revision_count
        
        # 3) Run Critic
        if critic_agent is None:
            raise RuntimeError("CriticAgent not available")
        
        critic_output: Dict[str, Any] = critic_agent.run(
            perception_output=perception_output,
            detection_output=detection_output,
            assessment_output=assessment_output,
            rules_summary=None,
        )
        
        # Record this round
        round_data = {
            "revision_round": revision_count,
            "perception_output": perception_output,
            "detection_output": detection_output,
            "assessment_output": assessment_output,
            "critic_output": critic_output,
        }
        revision_history.append(round_data)
        
        # Check Critic's judgement
        judgement = _safe_str(critic_output.get("summary", {}).get("overall_judgement") or critic_output.get("overall_judgement")).lower()
        
        # 1. Accept: done
        if judgement == "accept":
            break
        
        # 2. Flag for human: stop revision, mark and exit
        if judgement == "human_review":
            assessment_output.setdefault("_meta", {})["flagged_for_human"] = True
            assessment_output.setdefault("_meta", {})["flag_reason"] = critic_output.get("summary", {}).get("revise_reason") or "Critic flagged for human review"
            break
        
        # 3. Revise: check if we can do another round
        if judgement == "revise":
            if revision_count >= max_revisions:
                # Max revisions reached, stop
                assessment_output.setdefault("_meta", {})["max_revisions_reached"] = True
                assessment_output.setdefault("_meta", {})["max_revisions_limit"] = max_revisions
                break
            
            # Determine which agents need revision based on fix_by_agent
            detected_issues = critic_output.get("detected_issues", [])
            needs_perception_revision = any(
                isinstance(it, dict) and it.get("fix_by_agent") == "Perception"
                for it in detected_issues
            )
            needs_detection_revision = any(
                isinstance(it, dict) and it.get("fix_by_agent") == "Detection"
                for it in detected_issues
            )
            needs_assessment_revision = any(
                isinstance(it, dict) and it.get("fix_by_agent") == "Assessment"
                for it in detected_issues
            )
            
            # If Perception needs revision, re-run Perception on images
            if needs_perception_revision:
                if not pre_image_path or not post_image_path:
                    print("[WARN] Perception revision requested but image paths not provided. Skipping Perception revision.")
                    # Continue with Detection/Assessment revision only
                elif VLMClient is None or vlm_client is None:
                    print("[WARN] VLMClient not available for Perception revision. Skipping Perception revision.")
                    # Continue with Detection/Assessment revision only
                else:
                    # Re-run Perception on both images with revised prompts
                    print(f"[INFO] Re-running Perception with Critic feedback (revision round {revision_count + 1})")
                    
                    try:
                        # Re-run pre-disaster Perception (usually doesn't need revision, but we do it for consistency)
                        pre_revised = _run_perception_revision(
                            pre_image_path,
                            critic_output,
                            is_post_disaster=False,
                            vlm_client=vlm_client,
                        )
                        
                        # Re-run post-disaster Perception (this is the critical one)
                        post_revised = _run_perception_revision(
                            post_image_path,
                            critic_output,
                            is_post_disaster=True,
                            vlm_client=vlm_client,
                        )
                        
                        # Update perception_output with revised descriptions
                        pre_text_revised = json.dumps(pre_revised, ensure_ascii=False, indent=2)
                        post_text_revised = json.dumps(post_revised, ensure_ascii=False, indent=2)
                        
                        perception_output = {
                            "_meta": {
                                "source": "perception_files_revised",
                                "pre_path": str(pre_path),
                                "post_path": str(post_path),
                                "revision_round": revision_count + 1,
                            },
                            "pre_scene": {"raw": pre_revised, "summary": pre_text_revised[:1000]},
                            "post_scene": {"raw": post_revised, "summary": post_text_revised[:1000]},
                        }
                        
                        # Update pre_text_original and post_text_original for Detection
                        pre_text_original = pre_text_revised
                        post_text_original = post_text_revised
                        
                        print("[INFO] Perception revision completed")
                    except Exception as e:
                        print(f"[ERROR] Perception revision failed: {e}")
                        print("[WARN] Continuing with original Perception output")
                        # Continue with original perception_output
            
            # If only Assessment needs revision, we can do a lightweight revision
            if needs_assessment_revision and not needs_detection_revision and not needs_perception_revision:
                # Apply feedback to Assessment input
                assessment_output = _apply_critic_suggestions_to_assessment(assessment_output, critic_output)
                # Re-run Assessment only (pass the revised assessment_output as assessment_input so it can read critic_feedback)
                if run_assessment is not None and assessment_agent is not None:
                    assessment_output = run_assessment(
                        detection_output,  # Use same detection output
                        agent=assessment_agent,
                        assessment_input=assessment_output,  # Pass revised output so it can read critic_feedback from _meta
                    )
                    assessment_output.setdefault("_meta", {})["revision_round"] = revision_count + 1
                    # Re-run Critic on revised Assessment
                    critic_output = critic_agent.run(
                        perception_output=perception_output,
                        detection_output=detection_output,
                        assessment_output=assessment_output,
                        rules_summary=None,
                    )
                    revision_count += 1
                    # Continue loop to check new judgement
                    continue
            
            # Otherwise, do a full revision (Detection + Assessment)
            revision_count += 1
            # Continue loop to re-run Detection and Assessment
            continue
        
        # Unknown judgement: treat as accept
        break
    
    return {
        "perception_output": perception_output,
        "detection_output": detection_output,
        "assessment_output": assessment_output,
        "critic_output": critic_output,
        "revision_count": revision_count,
        "revision_history": revision_history,
    }
