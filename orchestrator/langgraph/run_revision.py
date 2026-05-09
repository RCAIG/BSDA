#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LangGraph details revision details（details 5）。

details：
- Agents/CriticAgent/main_revision.py
- Agents/CriticAgent/revision_orchestrator.py（while/if）
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from orchestrator.langgraph.graph import build_app


def _ensure_utf8_console() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    return cur


def main() -> None:
    _ensure_utf8_console()
    repo_root = _find_repo_root(Path(__file__).resolve().parent)

    p = argparse.ArgumentParser(description="Run LangGraph pipeline with Critic-driven revisions")
    p.add_argument("--pre_path", type=str, required=True, help="Pre-disaster Perception JSON description path")
    p.add_argument("--post_path", type=str, required=True, help="Post-disaster Perception JSON description path")
    p.add_argument("--pair_id", type=str, default="", help="Optional pair id (used for output naming)")
    p.add_argument("--max_revisions", type=int, default=1, help="Max revision rounds (0/1/2...)")
    p.add_argument("--use_rag", type=int, default=0, help="Enable RAG for Critic (0/1)")
    p.add_argument("--use_llm", type=int, default=1, help="Enable LLM for Critic (0/1)")
    p.add_argument("--pre_image_path", type=str, default="", help="Optional pre image path (for Perception revision)")
    p.add_argument("--post_image_path", type=str, default="", help="Optional post image path (for Perception revision)")
    p.add_argument("--output_dir", type=str, default="", help="Output directory root (default: <repo>/Agents)")
    args = p.parse_args()

    pre_path = Path(args.pre_path)
    post_path = Path(args.post_path)
    if not pre_path.exists():
        raise FileNotFoundError(f"pre_path not found: {pre_path}")
    if not post_path.exists():
        raise FileNotFoundError(f"post_path not found: {post_path}")

    output_root = Path(args.output_dir) if args.output_dir else (repo_root / "Agents")
    det_out_dir = output_root / "DetectionAgent" / "output"
    asr_out_dir = output_root / "AssessmentAgent" / "output"
    crt_out_dir = output_root / "CriticAgent" / "output"
    for d in [det_out_dir, asr_out_dir, crt_out_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Optional: init VLMClient if images are provided (keeps behavior consistent with main_revision.py)
    vlm_client: Optional[object] = None
    pre_img = Path(args.pre_image_path) if args.pre_image_path else None
    post_img = Path(args.post_image_path) if args.post_image_path else None
    if pre_img and post_img and pre_img.exists() and post_img.exists():
        try:
            from Agents.PerceptionAgent.main import VLMClient as VLMClientClass
            import os

            model_name = os.getenv("LOCAL_VLM_MODEL_PATH", "./models/LOCAL_MODEL")
            vlm_client = VLMClientClass(
                model_name=model_name,
                device="cuda",
                torch_dtype="bfloat16",
                max_new_tokens=2048,
                temperature=0.0,
            )
            print("[INFO] VLMClient initialized for Perception revision")
        except Exception as e:
            print(f"[WARN] Failed to initialize VLMClient: {e}")
            vlm_client = None

    app = build_app()
    pair_id = (args.pair_id or pre_path.stem.replace("_2023", "").replace("_2024", "")).strip()

    init_state = {
        "pair_id": pair_id,
        "pre_path": str(pre_path),
        "post_path": str(post_path),
        "pre_image_path": str(pre_img) if pre_img else None,
        "post_image_path": str(post_img) if post_img else None,
        "use_rag": int(args.use_rag),
        "use_llm": int(args.use_llm),
        "max_revisions": int(args.max_revisions),
        "vlm_client": vlm_client,
    }

    result = app.invoke(init_state)  # type: ignore[arg-type]

    stem = pre_path.stem
    if not stem.endswith("_2023") and not stem.endswith("_2024"):
        stem = f"{pair_id}_2023"

    detection_path = det_out_dir / f"{stem}_detection.json"
    assessment_path = asr_out_dir / f"{stem}_assessment.json"
    critic_path = crt_out_dir / f"{stem}_critic.json"

    detection_path.write_text(json.dumps(result.get("detection_output"), ensure_ascii=False, indent=2), encoding="utf-8")
    assessment_path.write_text(json.dumps(result.get("assessment_output"), ensure_ascii=False, indent=2), encoding="utf-8")
    critic_path.write_text(json.dumps(result.get("critic_output"), ensure_ascii=False, indent=2), encoding="utf-8")

    final_judgement = (
        (result.get("critic_output") or {}).get("summary", {}).get("overall_judgement")
        or (result.get("critic_output") or {}).get("overall_judgement")
    )
    print(f"\n[DONE] LangGraph pipeline completed. revision_count={result.get('revision_count', 0)} judgement={final_judgement}")
    print("[INFO] Outputs:")
    print(f"  Detection:  {detection_path}")
    print(f"  Assessment: {assessment_path}")
    print(f"  Critic:     {critic_path}")


if __name__ == "__main__":
    main()


