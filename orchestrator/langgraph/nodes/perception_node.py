from __future__ import annotations

import json
from pathlib import Path

from schemas.state import PipelineState
from orchestrator.langgraph.utils import (
    build_revision_prompt_suffix,
    ensure_repo_root_on_path,
    safe_str,
)


def _build_revision_prompt(*, base_prompt: str, critic_output: dict, is_post_disaster: bool) -> str:
    # Best-effort: reuse existing prompt by flipping "pre-disaster" wording if needed.
    p = base_prompt
    if is_post_disaster:
        p = p.replace("pre-disaster", "post-disaster").replace("pre-disaster street-view", "post-disaster street-view")
    suffix = build_revision_prompt_suffix(critic_output)
    return (p + "\n\n" + suffix).strip()


def _run_perception_revision(
    *,
    image_path: Path,
    critic_output: dict,
    vlm_client: object,
    is_post_disaster: bool,
) -> dict:
    """
    Minimal but functional Perception revision:
    - Use existing `Agents/PerceptionAgent/main.py` VLMClient.infer(prompt, img)
    - Parse output as JSON (loose) and return dict; fallback to raw_text wrapper.
    """
    from Agents.PerceptionAgent.main import build_single_image_prompt, load_image, try_parse_json_loose

    base_prompt = build_single_image_prompt()
    prompt = _build_revision_prompt(base_prompt=base_prompt, critic_output=critic_output, is_post_disaster=is_post_disaster)
    img = load_image(image_path)

    # VLMClient.infer returns a string
    raw = ""
    try:
        raw = getattr(vlm_client, "infer")(prompt, img)
    except Exception:
        # If injected client differs, try calling as function
        raw = str(vlm_client(prompt, img))  # type: ignore[misc]

    parsed = try_parse_json_loose(raw)
    if isinstance(parsed, dict):
        parsed.setdefault("_meta", {})
        parsed["_meta"]["revision_source"] = "vlm_revision"
        return parsed
    return {"raw_text": raw, "_meta": {"revision_source": "vlm_revision_unparsed"}}


def perception_node(state: PipelineState) -> PipelineState:
    """
    Perception node:
    - First pass: perception_output is loaded from files in InitNode -> no-op.
    - Revision pass: if rerun_perception=True and VLM+image paths are available, rerun perception and update perception_output.
    """
    ensure_repo_root_on_path(Path(__file__).resolve())

    if not bool(state.get("rerun_perception")):
        return state

    critic_output = state.get("critic_output") or {}
    if not isinstance(critic_output, dict):
        return state

    pre_img = state.get("pre_image_path")
    post_img = state.get("post_image_path")
    vlm_client = state.get("vlm_client")
    if not pre_img or not post_img or vlm_client is None:
        # Cannot revise; keep existing perception but annotate.
        meta = state.get("perception_output", {}).setdefault("_meta", {})
        meta["needs_perception_revision"] = True
        meta["perception_revision_skipped_reason"] = "missing pre/post images or vlm_client"
        state["rerun_perception"] = False
        return state

    pre_path = Path(str(pre_img))
    post_path = Path(str(post_img))
    if not pre_path.exists() or not post_path.exists():
        meta = state.get("perception_output", {}).setdefault("_meta", {})
        meta["needs_perception_revision"] = True
        meta["perception_revision_skipped_reason"] = "pre/post image path not found"
        state["rerun_perception"] = False
        return state

    # Run VLM revision on both images
    pre_revised = _run_perception_revision(
        image_path=pre_path,
        critic_output=critic_output,
        vlm_client=vlm_client,
        is_post_disaster=False,
    )
    post_revised = _run_perception_revision(
        image_path=post_path,
        critic_output=critic_output,
        vlm_client=vlm_client,
        is_post_disaster=True,
    )

    pre_text = json.dumps(pre_revised, ensure_ascii=False, indent=2)
    post_text = json.dumps(post_revised, ensure_ascii=False, indent=2)

    state["perception_output"] = {
        "_meta": {
            "source": "perception_files_revised",
            "pre_path": safe_str(state.get("pre_path")),
            "post_path": safe_str(state.get("post_path")),
            "revision_round": int(state.get("revision_count", 0)),
        },
        "pre_scene": {"raw": pre_revised, "summary": pre_text[:1000]},
        "post_scene": {"raw": post_revised, "summary": post_text[:1000]},
    }

    # Perception has been rerun for this cycle; clear flag.
    state["rerun_perception"] = False
    return state


