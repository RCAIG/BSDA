#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/pipeline.py

details multi-agent orchestrator（details）：
- DetectionAgent: details/details detection JSON
- AssessmentAgent: details detection JSON details assessment JSON
- CriticAgent: details detection+assessment（details perception）details critic JSON

details：
- PerceptionAgent（details）details，details/detailsGPU；
  details“Perception details”，details Perception details（pre/post details JSON）details。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional


def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    return cur


def _run(cmd: list[str]) -> None:
    # details
    print("\n[RUN] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def _latest_match(dir_path: Path, pattern: str) -> Optional[Path]:
    candidates = list(dir_path.glob(pattern))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _resolve_detection_output(detection_out_dir: Path, pair_id: str) -> Optional[Path]:
    pair_id = (pair_id or "").strip()
    if pair_id:
        # details：5_2023_detection.json
        hit = _latest_match(detection_out_dir, f"{pair_id}_*_detection.json")
        if hit:
            return hit
    return _latest_match(detection_out_dir, "*_detection.json")


def _default_assessment_output_path(assessment_out_dir: Path, detection_path: Path) -> Path:
    stem = detection_path.name.replace("_detection.json", "")
    return assessment_out_dir / f"{stem}_assessment.json"


def _default_critic_output_path(critic_out_dir: Path, assessment_path: Path) -> Path:
    stem = assessment_path.name.replace("_assessment.json", "")
    return critic_out_dir / f"{stem}_critic.json"


def _ensure_utf8_console() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def main() -> None:
    _ensure_utf8_console()

    repo_root = _find_repo_root(Path(__file__).resolve().parent)
    agents_root = repo_root / "Agents"
    det_main = agents_root / "DetectionAgent" / "main.py"
    asr_main = agents_root / "AssessmentAgent" / "main.py"
    crt_main = agents_root / "CriticAgent" / "main.py"

    det_out_dir = agents_root / "DetectionAgent" / "output"
    asr_out_dir = agents_root / "AssessmentAgent" / "output"
    crt_out_dir = agents_root / "CriticAgent" / "output"
    for d in [det_out_dir, asr_out_dir, crt_out_dir]:
        os.makedirs(str(d), exist_ok=True)

    p = argparse.ArgumentParser(description="Multi-agent pipeline orchestrator (Detection -> Assessment -> Critic)")
    p.add_argument("--pair_id", type=str, default="", help="details pair_id（details 5）；details detection details")
    p.add_argument("--pre_path", type=str, default="", help="Perception details：details（.json/.txt）")
    p.add_argument("--post_path", type=str, default="", help="Perception details：details（.json/.txt）")
    p.add_argument("--run_detection", type=int, default=1, help="details DetectionAgent：1/0（details1，details agent details detection）")
    p.add_argument("--max_changes", type=int, default=0, help="details Detection details --max_changes（details）")
    p.add_argument("--input_detection", type=str, default="", help="details detection JSON（details Detection）")
    p.add_argument("--output_assessment", type=str, default="", help="details assessment details（details）")
    p.add_argument("--output_critic", type=str, default="", help="details critic details（details）")
    p.add_argument("--use_rag", type=int, default=0, help="Critic details RAG：1/0（details0，details）")
    p.add_argument("--use_llm", type=int, default=1, help="Critic details LLM：1/0")
    args = p.parse_args()

    # 1) Detection
    detection_path: Optional[Path] = Path(args.input_detection) if args.input_detection else None
    if args.run_detection:
        # details Perception details pre/post details，details ONE_SHOT details（details multi-agent details）
        pre_path = (args.pre_path or "").strip()
        post_path = (args.post_path or "").strip()
        if pre_path and post_path:
            out_detection = det_out_dir / (Path(pre_path).stem + "_detection.json")
            cmd = [
                sys.executable,
                str(det_main),
                "--pre_path",
                pre_path,
                "--post_path",
                post_path,
                "--output_detection",
                str(out_detection),
            ]
            if args.pair_id:
                cmd += ["--only_pair_id", str(args.pair_id)]
            if int(args.max_changes) > 0:
                cmd += ["--max_changes", str(int(args.max_changes))]
            _run(cmd)
            detection_path = out_detection
        else:
            # details：details DetectionAgent details（details，details）
            cmd = [sys.executable, str(det_main)]
            if args.pair_id:
                cmd += ["--only_pair_id", str(args.pair_id)]
            if int(args.max_changes) > 0:
                cmd += ["--max_changes", str(int(args.max_changes))]
            _run(cmd)
            detection_path = _resolve_detection_output(det_out_dir, str(args.pair_id))

    if not detection_path or not detection_path.exists():
        # details detection，details output details pair_id details
        detection_path = _resolve_detection_output(det_out_dir, str(args.pair_id))
    if not detection_path or not detection_path.exists():
        raise FileNotFoundError(f"details detection details。details Detection details --input_detection。details：{det_out_dir}")

    # details detection details error，details（details“details”details）
    try:
        det_obj = json.loads(detection_path.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        det_obj = {}
    if isinstance(det_obj, dict) and det_obj.get("error"):
        raise RuntimeError(f"DetectionAgent details：{det_obj.get('error')}")

    # 2) Assessment
    assessment_path = Path(args.output_assessment) if args.output_assessment else _default_assessment_output_path(asr_out_dir, detection_path)
    _run(
        [
            sys.executable,
            str(asr_main),
            "--input_detection",
            str(detection_path),
            "--output_assessment",
            str(assessment_path),
        ]
    )

    # 3) Critic
    critic_path = Path(args.output_critic) if args.output_critic else _default_critic_output_path(crt_out_dir, assessment_path)
    critic_cmd = [
        sys.executable,
        str(crt_main),
        "--input_assessment",
        str(assessment_path),
        "--input_detection",
        str(detection_path),
        "--output_critic",
        str(critic_path),
        "--use_rag",
        str(int(args.use_rag)),
        "--use_llm",
        str(int(args.use_llm)),
    ]
    # If we started the pipeline from Perception JSON files, also pass them to Critic
    # so that CriticAgent can see the original scene descriptions.
    if args.pre_path:
        critic_cmd += ["--pre_path", args.pre_path]
    if args.post_path:
        critic_cmd += ["--post_path", args.post_path]
    _run(critic_cmd)

    summary = {
        "detection": str(detection_path),
        "assessment": str(assessment_path),
        "critic": str(critic_path),
    }
    print("\n[DONE] pipeline outputs:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

