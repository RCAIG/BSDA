#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/main.py
details：
- details/details
- details
- details pair_map（details/details）
- details DetectionAgent（details：details → details RAG → LLM details）
- details OUTPUT_DIR
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import config
from detection_agent import DetectionAgent
import detection_agent as detmod
from rag import DamageFeatureRAG


# ===================== details =====================

def _ensure_utf8_console() -> None:
    """Windows details：details UTF-8，details。"""
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _read_text_maybe_json(path: Path) -> str:
    """
    details .json：
    - details JSON details text/description/content details，details；
    - details JSON。
    """
    raw = path.read_text(encoding="utf-8", errors="replace")

    if path.suffix.lower() != ".json":
        return raw

    try:
        obj = json.loads(raw)
    except Exception:
        return raw

    def _get(d: Dict[str, Any], *keys: str) -> str:
        cur: Any = d
        for k in keys:
            if not isinstance(cur, dict):
                return ""
            cur = cur.get(k)
        return cur.strip() if isinstance(cur, str) else ""

    if isinstance(obj, dict):
        for k in ["text", "description", "content", "scene_description", "caption"]:
            v = obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()

        # PerceptionAgent details：details；details，details JSON details Detection LLM details OOM。
        if "overall_scene" in obj or "road_and_traffic" in obj or "buildings_and_structures" in obj:
            parts: List[str] = []
            overall_free = _get(obj, "overall_scene", "free_text")
            if overall_free:
                parts.append(f"[overall_scene] {overall_free}")
            road_surface = _get(obj, "road_and_traffic", "surface_condition")
            if road_surface:
                parts.append(f"[road] {road_surface}")
            road_obstacles = _get(obj, "road_and_traffic", "obstacles_on_road")
            if road_obstacles:
                parts.append(f"[road_obstacles] {road_obstacles}")
            bld_dmg = _get(obj, "buildings_and_structures", "damage_level_qualitative")
            if bld_dmg:
                parts.append(f"[buildings_damage] {bld_dmg}")
            bld_facade = _get(obj, "buildings_and_structures", "facades_and_roofs_damage")
            if bld_facade:
                parts.append(f"[facade_roof] {bld_facade}")
            debris = _get(obj, "buildings_and_structures", "debris_near_buildings")
            if debris:
                parts.append(f"[building_debris] {debris}")
            veg = _get(obj, "vegetation_ground_and_debris", "loose_debris_and_trash")
            if veg:
                parts.append(f"[debris_trash] {veg}")
            water = _get(obj, "vegetation_ground_and_debris", "water_related_ground_features")
            if water:
                parts.append(f"[water_traces] {water}")
            view = _get(obj, "viewpoint_and_layout", "main_layout_summary")
            if view:
                parts.append(f"[layout] {view}")

            if parts:
                return "\n".join(parts).strip()

        # details JSON details，details
        return json.dumps(obj, ensure_ascii=False)

    if isinstance(obj, list):
        parts = []
        for it in obj:
            if isinstance(it, str):
                parts.append(it.strip())
            elif isinstance(it, dict):
                parts.append(json.dumps(it, ensure_ascii=False))
        return "\n".join([p for p in parts if p])

    return raw


def _extract_pair_key(filename: str) -> Optional[str]:
    """
    details/details key：
    - details：1_2023.json / 1_2024.json  -> key="1"
    - details：1_pre.json / 1_post.json  -> key="1"
    - details，details stem details key
    """
    name = Path(filename).name
    stem = Path(name).stem

    # details _pre details _post details（details 1_pre.json / 1_post.json）
    m = re.match(r"^(:P<base>.+)_(pre|post)$", stem, re.IGNORECASE)
    if m:
        return m.group("base")
    
    # details _YYYY details（details 1_2023.json / 1_2024.json）
    m = re.match(r"^(:P<base>.+)_(:P<year>\d{4})$", stem)
    if m:
        return m.group("base")

    return stem or None


def _index_files_for_pairing(dir_path: Path) -> Dict[str, Path]:
    """
    details，details：pair_key -> details details。
    details .txt / .json。
    """
    out: Dict[str, Path] = {}

    if not dir_path.exists():
        return out

    for p in dir_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".txt", ".json"]:
            continue

        key = _extract_pair_key(p.name)
        if not key:
            continue

        # details key details，details
        out.setdefault(key, p)

    return out


def _load_pair_map(csv_path: str) -> Dict[str, Dict[str, Any]]:
    """
    details pair_map CSV，details：pair_id(str) -> row(dict)
    details：pre_date / post_date / lat / lon / dist_m（details）
    """
    out: Dict[str, Dict[str, Any]] = {}

    if not csv_path:
        return out

    p = Path(csv_path)
    if not p.exists():
        return out

    with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not isinstance(row, dict):
                continue
            pid = (row.get("pair_id") or "").strip()
            if not pid:
                continue
            out[pid] = row

    return out


def _extract_pair_meta(pair_map: Dict[str, Dict[str, Any]], pair_id: str) -> Dict[str, Any]:
    """
    details pair_id details pair_map details：
    - pair_id / pre_date / post_date / lat / lon / dist_m
    details DetectionAgent details prompt，details extra_context details RAG。
    """
    row = pair_map.get(str(pair_id), {})
    if not row:
        return {}

    return {
        "pair_id": str(pair_id),
        "pre_date": (row.get("pre_date") or "").strip(),
        "post_date": (row.get("post_date") or "").strip(),
        "lat": (row.get("lat") or "").strip(),
        "lon": (row.get("lon") or "").strip(),
        "dist_m": (row.get("dist_m") or "").strip(),
        # details，details
    }


# ===================== details =====================

def main() -> None:
    _ensure_utf8_console()
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    parser = argparse.ArgumentParser(description="DetectionAgent details（details pair / details）")
    # details：details“details/details”（details Perception details）
    parser.add_argument("--pre_path", type=str, default="", help="details：details（.json/.txt）")
    parser.add_argument("--post_path", type=str, default="", help="details：details（.json/.txt）")
    parser.add_argument("--output_detection", type=str, default="", help="details：details detection details JSON details（details）")
    parser.add_argument("--only_pair_id", type=str, default="", help="details pair_id（details 1）")
    parser.add_argument("--max_changes", type=int, default=0, help="details N details RAG+details（details）")
    parser.add_argument("--step1_only", action="store_true", help="details Step1（details），details RAG/details；details step1_raw/parsed/cleaned")
    parser.add_argument(
        "--debug_full",
        action="store_true",
        help="details Step1/2/3，details（step1/step2/step3 details prompt/raw/parsed/enriched）",
    )
    args, _ = parser.parse_known_args()

    out_dir = Path(config.OUTPUT_DIR)

    # ---------- details RAG ----------
    corpus = str(getattr(config, "RAG_CORPUS", "gov")).lower()
    if corpus == "street":
        embed_model = str(
            getattr(config, "RAG_EMBED_MODEL_STREET",
                    "sentence-transformers/all-MiniLM-L6-v2")
        )
    else:
        # details gov details
        corpus = "gov"
        embed_model = str(
            getattr(config, "RAG_EMBED_MODEL_GOV",
                    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        )

    rag = DamageFeatureRAG(
        artifacts_dir=config.RAG_ARTIFACTS_DIR,
        corpus=corpus,
        embed_model=embed_model,
    )

    # ---------- details DetectionAgent ----------
    agent = DetectionAgent(
        model_path=config.LOCAL_LLM_MODEL_PATH,
        rag=rag,
        max_new_tokens=int(getattr(config, "MAX_NEW_TOKENS", 800)),
        max_new_tokens_step1=int(getattr(config, "MAX_NEW_TOKENS_STEP1", getattr(config, "MAX_NEW_TOKENS", 800))),
        max_new_tokens_step3=int(getattr(config, "MAX_NEW_TOKENS_STEP3", getattr(config, "MAX_NEW_TOKENS", 800))),
        rag_top_k=int(getattr(config, "RAG_TOP_K", 5)),
    )

    # ---------- details（details pre/post details） ----------
    pre_path_arg = (args.pre_path or "").strip()
    post_path_arg = (args.post_path or "").strip()
    if pre_path_arg and post_path_arg:
        pre_path = Path(pre_path_arg)
        post_path = Path(post_path_arg)
        if not pre_path.exists():
            raise FileNotFoundError(f"--pre_path not found: {pre_path}")
        if not post_path.exists():
            raise FileNotFoundError(f"--post_path not found: {post_path}")

        # pair_id：details --only_pair_id；details
        key = (args.only_pair_id or "").strip()
        if not key:
            key = _extract_pair_key(pre_path.name) or _extract_pair_key(post_path.name) or ""

        print(f"\n[ONE_SHOT] details：{pre_path.name}  <->  {post_path.name} (pair_id={key or 'unknown'})")
        # details perception JSON details Detection（details/details）
        pre_text = pre_path.read_text(encoding="utf-8", errors="replace")
        post_text = post_path.read_text(encoding="utf-8", errors="replace")

        # details pair_map（details；details key details）
        pair_meta: Dict[str, Any] = {}
        pair_map_csv = str(getattr(config, "PAIR_MAP_CSV", "") or "").strip()
        pair_map = _load_pair_map(pair_map_csv) if key else {}
        if pair_map and key:
            pair_meta = _extract_pair_meta(pair_map, key)
        # details“details JSON details”，details/details（details）
        try:
            pair_meta["pre_text_raw"] = pre_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pair_meta["pre_text_raw"] = pre_text
        try:
            pair_meta["post_text_raw"] = post_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            pair_meta["post_text_raw"] = post_text
        # details slim details（details）

        try:
            max_changes = int(args.max_changes) if int(args.max_changes) > 0 else None
            result = agent.run(pre_text, post_text, pair_meta=pair_meta, max_changes=max_changes)
        except Exception as e:
            print(f"[ERROR] ONE_SHOT pair={key} details：{e}")
            result = {"error": str(e), "pair_id": key, "pair_meta": pair_meta}

        if args.output_detection:
            out_path = Path(args.output_detection)
        else:
            # details（details）
            out_path = out_dir / f"{pre_path.stem}_detection.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"[ONE_SHOT] details：{out_path}")
        return

    # ---------- details（details） ----------
    pre_dir = Path(config.PRE_DESC_DIR)
    post_dir = Path(config.POST_DESC_DIR)

    pre_files = _index_files_for_pairing(pre_dir)
    post_files = _index_files_for_pairing(post_dir)

    common_keys = sorted(set(pre_files.keys()) & set(post_files.keys()))
    if not common_keys:
        print("details（details <id>_YYYY.* details id details）。details id details。")
        print(f"details: {pre_dir}")
        print(f"details: {post_dir}")
        return

    only_pair_id = (args.only_pair_id or "").strip()
    if only_pair_id:
        common_keys = [k for k in common_keys if k == only_pair_id]
        if not common_keys:
            print(f"details only_pair_id={only_pair_id} details。")
            return

    # ---------- details pair_map ----------
    pair_map_csv = str(getattr(config, "PAIR_MAP_CSV", "") or "").strip()
    pair_map = _load_pair_map(pair_map_csv)
    if pair_map:
        print(f"details pair_map：{pair_map_csv}（rows={len(pair_map)}）")
    else:
        print("details：details pair_map（details/details）。")

    # ---------- details pair，details DetectionAgent ----------
    for key in common_keys:
        pre_path = pre_files[key]
        post_path = post_files[key]

        print(f"\ndetails：{pre_path.name}  <->  {post_path.name}")

        pre_text = _read_text_maybe_json(pre_path)
        post_text = _read_text_maybe_json(post_path)
        pair_meta = _extract_pair_meta(pair_map, key)
        # details pair_meta（details prompt details，details RAG/details）
        if isinstance(pair_meta, dict):
            pair_meta["pre_text_raw"] = pre_text
            pair_meta["post_text_raw"] = post_text

        # ---------- Step1-only details：details ----------
        if bool(args.step1_only):
            debug_dir = out_dir / f"debug_step1_pair_{key}"
            debug_dir.mkdir(parents=True, exist_ok=True)
            # details DetectionAgent details Step1 prompt details
            step1_system = (
                "details，details/details“details”，"
                "details。details JSON。"
            )
            step1_user = agent._build_change_extraction_prompt(pre_text, post_text, pair_meta=pair_meta)
            step1_raw = agent._generate_llm(
                step1_system,
                step1_user,
                max_new_tokens=agent.max_new_tokens_step1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            step1_obj = detmod._extract_json_object(step1_raw)
            # details Step1 details schema：state + evidence + kind（details id；details id）
            def _s(x: Any) -> str:
                if x is None:
                    return ""
                if isinstance(x, str):
                    return x.strip()
                return str(x).strip()

            allowed_kinds = {
                "unknown",
            }

            changes_cleaned: List[Dict[str, Any]] = []
            if isinstance(step1_obj, dict):
                raw_changes = step1_obj.get("changes", [])
                if isinstance(raw_changes, list):
                    for i, ch in enumerate(raw_changes, start=1):
                        if not isinstance(ch, dict):
                            continue
                        module = _s(ch.get("module", "")) or "unknown"
                        change_description = _s(ch.get("change_description", ""))

                        pre_evidence = _s(ch.get("pre_evidence", ""))
                        post_evidence = _s(ch.get("post_evidence", ""))

                        if not (change_description or pre_evidence or post_evidence):
                            continue

                        component_hint = module
                        if pre_evidence:
                            component_hint = pre_evidence[:24]
                        elif post_evidence:
                            component_hint = post_evidence[:24]

                        changes_cleaned.append(
                            {
                                "id": f"chg_{i:03d}",
                                "module": module,
                                "change_description": change_description,
                                "pre_evidence": pre_evidence,
                                "post_evidence": post_evidence,
                                # details hint（details）
                                "component": component_hint,
                            }
                        )

            (debug_dir / "step1_system.txt").write_text(step1_system, encoding="utf-8")
            (debug_dir / "step1_user_prompt.txt").write_text(step1_user, encoding="utf-8")
            (debug_dir / "step1_raw.txt").write_text(step1_raw, encoding="utf-8")
            (debug_dir / "step1_parsed.json").write_text(
                json.dumps(step1_obj, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (debug_dir / "changes_cleaned.json").write_text(
                json.dumps(changes_cleaned, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[STEP1_ONLY] details：{debug_dir}")
            # step1_only details pair details
            return

        # ---------- Debug Full：details 1/2/3 details ----------
        if bool(getattr(args, "debug_full", False)):
            debug_dir = out_dir / f"debug_pair_{key}_full"
            debug_dir.mkdir(parents=True, exist_ok=True)

            def _dump_text(p: Path, s: str) -> None:
                p.write_text(s or "", encoding="utf-8")

            def _dump_json(p: Path, obj: Any) -> None:
                p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

            def _compact_enriched_for_step3(enriched: Any, *, max_hit_chars: int = 700) -> Any:
                """
                Step3 details prompt details OOM details：enriched_changes details rag_hits，
                details text details。details，details Step3 details：
                - rag_hits: details source/meta details + text details
                - change: details Step3 details（id/module/change_description/pre/post evidence）
                """
                if not isinstance(enriched, list):
                    return enriched
                out: list = []
                for ch in enriched:
                    if not isinstance(ch, dict):
                        continue
                    rag_hits = ch.get("rag_hits", [])
                    compact_hits = []
                    if isinstance(rag_hits, list):
                        for h in rag_hits:
                            if not isinstance(h, dict):
                                continue
                            txt = str(h.get("text", "") or "")
                            if len(txt) > max_hit_chars:
                                txt = txt[:max_hit_chars] + " ..."
                            meta = h.get("meta", {})
                            if not isinstance(meta, dict):
                                meta = {}
                            compact_hits.append(
                                {
                                    "text": txt,
                                    "source": h.get("source", ""),
                                    "meta": {
                                        "doc_id": meta.get("doc_id", ""),
                                        "chunk_id": meta.get("chunk_id", ""),
                                        "title": meta.get("title", ""),
                                        "section": meta.get("section", ""),
                                        "page": meta.get("page", ""),
                                        "score_hint": meta.get("score_hint", ""),
                                    },
                                }
                            )
                    out.append(
                        {
                            "id": ch.get("id"),
                            "module": ch.get("module", ""),
                            "change_description": ch.get("change_description", ""),
                            "pre_evidence": ch.get("pre_evidence", ""),
                            "post_evidence": ch.get("post_evidence", ""),
                            "rag_hits": compact_hits,
                        }
                    )
                return out

            if isinstance(pair_meta, dict):
                _dump_json(debug_dir / "pair_meta.json", pair_meta)
            _dump_text(debug_dir / "pre_text.txt", pre_text)
            _dump_text(debug_dir / "post_text.txt", post_text)

            # ===== Step 1 =====
            step1_system = (
                "details，details/details“details”，"
                "details。details JSON。"
            )
            step1_user = agent._build_change_extraction_prompt(pre_text, post_text, pair_meta=pair_meta)
            _dump_text(debug_dir / "step1_system.txt", step1_system)
            _dump_text(debug_dir / "step1_user_prompt.txt", step1_user)

            step1_raw = agent._generate_llm(
                step1_system,
                step1_user,
                max_new_tokens=agent.max_new_tokens_step1,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
            _dump_text(debug_dir / "step1_raw.txt", step1_raw)
            step1_obj = detmod._extract_json_object(step1_raw)
            _dump_json(debug_dir / "step1_parsed.json", step1_obj)

            changes_all = agent.extract_changes(pre_text, post_text, pair_meta=pair_meta)
            _dump_json(debug_dir / "changes_cleaned_all.json", changes_all)
            max_changes = int(args.max_changes) if int(args.max_changes) > 0 else None
            changes = changes_all[:max_changes] if (max_changes is not None and max_changes > 0) else changes_all
            _dump_json(debug_dir / "changes_cleaned.json", changes)

            # ===== Step 2 =====
            enriched_changes = agent.enrich_changes_with_rag(changes, pair_meta=pair_meta)
            _dump_json(debug_dir / "enriched_changes.json", enriched_changes)

            # ===== Step 3 =====
            # OOM details：enriched_changes details（rag_hits text details）。
            # details：details Step3 details + details，details。
            step3_system = (
                "details Detection Agent，details。"
                "details JSON。"
            )
            _dump_text(debug_dir / "step3_system.txt", step3_system)

            compact_all = _compact_enriched_for_step3(enriched_changes, max_hit_chars=700)
            _dump_json(debug_dir / "enriched_changes_compact_for_step3.json", compact_all)

            batch_size = 2  # details 2 details，details；details
            confirmed_all: list = []
            pseudo_all: list = []
            uncertain_all: list = []
            step3_batches_dir = debug_dir / "step3_batches"
            step3_batches_dir.mkdir(parents=True, exist_ok=True)

            if not isinstance(compact_all, list):
                compact_all = []
            for bi in range(0, len(compact_all), batch_size):
                batch = compact_all[bi : bi + batch_size]
                step3_user = agent._build_classification_prompt(batch, pair_meta=pair_meta)
                _dump_text(step3_batches_dir / f"step3_user_prompt_batch_{bi//batch_size+1}.txt", step3_user)

                try:
                    step3_raw = agent._generate_llm(
                        step3_system,
                        step3_user,
                        max_new_tokens=agent.max_new_tokens_step3,
                    )
                except RuntimeError as e:
                    # details OOM：details batch
                    if "not enough memory" in str(e).lower() and len(batch) > 1:
                        for si, single in enumerate(batch, start=1):
                            single_user = agent._build_classification_prompt([single], pair_meta=pair_meta)
                            _dump_text(
                                step3_batches_dir / f"step3_user_prompt_batch_{bi//batch_size+1}_single_{si}.txt",
                                single_user,
                            )
                            single_raw = agent._generate_llm(
                                step3_system,
                                single_user,
                                max_new_tokens=agent.max_new_tokens_step3,
                            )
                            _dump_text(
                                step3_batches_dir / f"step3_raw_batch_{bi//batch_size+1}_single_{si}.txt",
                                single_raw,
                            )
                            single_obj = detmod._extract_json_object(single_raw)
                            _dump_json(
                                step3_batches_dir / f"step3_parsed_batch_{bi//batch_size+1}_single_{si}.json",
                                single_obj,
                            )
                            if isinstance(single_obj, dict):
                                single_obj = detmod._normalize_step3_buckets(single_obj)  # type: ignore[attr-defined]
                                confirmed_all.extend(single_obj.get("confirmed_disaster_damage", []) or [])
                                pseudo_all.extend(single_obj.get("likely_pseudo_change", []) or [])
                                uncertain_all.extend(single_obj.get("uncertain", []) or [])
                        continue
                    raise

                _dump_text(step3_batches_dir / f"step3_raw_batch_{bi//batch_size+1}.txt", step3_raw)
                step3_obj = detmod._extract_json_object(step3_raw)
                _dump_json(step3_batches_dir / f"step3_parsed_batch_{bi//batch_size+1}.json", step3_obj)

                if isinstance(step3_obj, dict):
                    step3_obj = detmod._normalize_step3_buckets(step3_obj)  # type: ignore[attr-defined]
                    confirmed_all.extend(step3_obj.get("confirmed_disaster_damage", []) or [])
                    pseudo_all.extend(step3_obj.get("likely_pseudo_change", []) or [])
                    uncertain_all.extend(step3_obj.get("uncertain", []) or [])

            final = {
                "confirmed_disaster_damage": confirmed_all,
                "likely_pseudo_change": pseudo_all,
                "uncertain": uncertain_all,
                # details（details）
                "verified_hurricane_damages": confirmed_all,
                "pseudo_changes": pseudo_all,
                "_stats": {
                    "confirmed_disaster_damage_count": len(confirmed_all),
                    "likely_pseudo_change_count": len(pseudo_all),
                    "uncertain_count": len(uncertain_all),
                    "change_count_input": len(enriched_changes),
                    "step3_batch_size": batch_size,
                },
                "_intermediate": {
                    "changes": changes,
                    "enriched_changes": enriched_changes,
                },
            }
            # details，details
            try:
                final = detmod._normalize_step3_buckets(final)  # type: ignore[attr-defined]
            except Exception:
                pass
            _dump_json(debug_dir / "final_with_intermediate.json", final)

            out_path = out_dir / f"{pre_path.stem}_detection.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(final, f, ensure_ascii=False, indent=2)

            print(f"[DEBUG_FULL] details：{debug_dir}")
            print(f"details：{out_path}")
            return

        # details：details
        # 1）extract_changes
        # 2）enrich_changes_with_rag（details RAG details）
        # 3）classify_with_rag（LLM details RAG hits details）
        try:
            max_changes = int(args.max_changes) if int(args.max_changes) > 0 else None
            result = agent.run(pre_text, post_text, pair_meta=pair_meta, max_changes=max_changes)
        except Exception as e:
            # details，details
            print(f"[ERROR] details pair={key} details：{e}")
            result = {
                "error": str(e),
                "pair_id": key,
                "pair_meta": pair_meta,
            }

        # details（details）
        out_path = out_dir / f"{pre_path.stem}_detection.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"details：{out_path}")


if __name__ == "__main__":
    main()