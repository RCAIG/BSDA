#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CriticAgent/main_revision.py

CLI entry point for running pipeline with automatic revision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add repo root to path
_here = Path(__file__).resolve().parent
_repo_root = None
for p in [_here] + list(_here.parents):
    if (p / "models").exists() and (p / "RAG").exists():
        _repo_root = p
        break
if _repo_root and str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from Agents.CriticAgent.revision_orchestrator import run_pipeline_with_revision


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-agent pipeline with automatic revision based on Critic feedback"
    )
    parser.add_argument("--pre_path", type=str, required=True, help="Pre-disaster Perception JSON description path")
    parser.add_argument("--post_path", type=str, required=True, help="Post-disaster Perception JSON description path")
    parser.add_argument("--pair_id", type=str, default="", help="Optional pair ID for output file naming")
    parser.add_argument("--max_revisions", type=int, default=1, help="Maximum revision rounds (0=no revision, 1=one round, 2=two rounds)")
    parser.add_argument("--use_rag", type=int, default=0, help="Enable RAG for Critic (0/1)")
    parser.add_argument("--use_llm", type=int, default=1, help="Enable LLM for Critic (0/1)")
    parser.add_argument("--pre_image_path", type=str, default="", help="Pre-disaster image path (required for Perception revision)")
    parser.add_argument("--post_image_path", type=str, default="", help="Post-disaster image path (required for Perception revision)")
    parser.add_argument("--output_dir", type=str, default="", help="Output directory (default: Agents/*/output)")
    args = parser.parse_args()
    
    pre_path = Path(args.pre_path)
    post_path = Path(args.post_path)
    
    if not pre_path.exists():
        print(f"[ERROR] Pre-disaster file not found: {pre_path}")
        sys.exit(1)
    if not post_path.exists():
        print(f"[ERROR] Post-disaster file not found: {post_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if _repo_root:
            output_dir = _repo_root / "Agents"
        else:
            output_dir = Path(__file__).parent.parent
    
    # Parse image paths (optional, for Perception revision)
    pre_image_path = Path(args.pre_image_path) if args.pre_image_path else None
    post_image_path = Path(args.post_image_path) if args.post_image_path else None
    
    if pre_image_path and not pre_image_path.exists():
        print(f"[WARN] Pre-disaster image not found: {pre_image_path}")
        pre_image_path = None
    if post_image_path and not post_image_path.exists():
        print(f"[WARN] Post-disaster image not found: {post_image_path}")
        post_image_path = None
    
    # Initialize VLMClient if image paths are provided (for Perception revision)
    vlm_client = None
    if pre_image_path and post_image_path:
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
            print("[WARN] Perception revision will be skipped if needed")
            vlm_client = None
    
    # Run pipeline with revision
    print(f"[INFO] Running pipeline with max_revisions={args.max_revisions}")
    print(f"[INFO] Pre: {pre_path}")
    print(f"[INFO] Post: {post_path}")
    if pre_image_path:
        print(f"[INFO] Pre image: {pre_image_path}")
    if post_image_path:
        print(f"[INFO] Post image: {post_image_path}")
    
    result = run_pipeline_with_revision(
        pre_path=pre_path,
        post_path=post_path,
        pair_id=args.pair_id,
        max_revisions=args.max_revisions,
        use_rag=args.use_rag,
        use_llm=args.use_llm,
        pre_image_path=pre_image_path,
        post_image_path=post_image_path,
        vlm_client=vlm_client,
    )
    
    # Save outputs
    pair_id = args.pair_id or pre_path.stem.replace("_2023", "").replace("_2024", "")
    
    det_out_dir = output_dir / "DetectionAgent" / "output"
    asr_out_dir = output_dir / "AssessmentAgent" / "output"
    crt_out_dir = output_dir / "CriticAgent" / "output"
    
    for d in [det_out_dir, asr_out_dir, crt_out_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Determine filename stem
    stem = pre_path.stem  # e.g., "5_2023"
    if not stem.endswith("_2023") and not stem.endswith("_2024"):
        stem = f"{pair_id}_2023"
    
    detection_path = det_out_dir / f"{stem}_detection.json"
    assessment_path = asr_out_dir / f"{stem}_assessment.json"
    critic_path = crt_out_dir / f"{stem}_critic.json"
    
    detection_path.write_text(
        json.dumps(result["detection_output"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    assessment_path.write_text(
        json.dumps(result["assessment_output"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    critic_path.write_text(
        json.dumps(result["critic_output"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    
    print(f"\n[DONE] Pipeline completed with {result['revision_count']} revision rounds")
    print(f"[INFO] Outputs:")
    print(f"  Detection: {detection_path}")
    print(f"  Assessment: {assessment_path}")
    print(f"  Critic: {critic_path}")
    
    # Print final judgement
    final_judgement = result["critic_output"].get("summary", {}).get("overall_judgement") or result["critic_output"].get("overall_judgement")
    print(f"[INFO] Final judgement: {final_judgement}")


if __name__ == "__main__":
    main()
