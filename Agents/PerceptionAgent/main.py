#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Single-image description script - OPTIMIZED VERSION
- Flash Attention 2 enabled
- Parallel image preprocessing
- Optimized batch inference
- Multi-GPU support with better memory management
"""

from __future__ import annotations

import argparse
import os
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
import threading

import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig

# Prefer the new API (transformers >= 5.0). If unavailable, fall back to the old API.
try:
    from transformers import AutoModelForImageTextToText as AutoModelForVision2Seq
except ImportError:
    from transformers import AutoModelForVision2Seq


LOGGER = logging.getLogger("single_image_perception")

Image.MAX_IMAGE_PIXELS = None

# ---- Defaults ----
os.environ.setdefault("HF_ENDPOINT", "http<LOCAL_PATH>")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


def _patch_flash_attn_symbol() -> None:
    """Compat for repos importing flash_attn_varlen_func from transformers internals."""
    try:
        import transformers.modeling_flash_attention_utils as flash_utils  # type: ignore
    except Exception:
        return
    if hasattr(flash_utils, "flash_attn_varlen_func"):
        return
    try:
        from flash_attn import flash_attn_varlen_func  # type: ignore

        setattr(flash_utils, "flash_attn_varlen_func", flash_attn_varlen_func)
    except Exception:
        # Keep silent here; caller may still load with non-flash attention path.
        pass


def _patch_layer_type_validation_symbol() -> None:
    """Compat for repos importing layer_type_validation from transformers internals."""
    try:
        import transformers.configuration_utils as cfg_utils  # type: ignore
    except Exception:
        return
    if hasattr(cfg_utils, "layer_type_validation"):
        return

    def _layer_type_validation(layer_types: object) -> None:
        if layer_types is None:
            return
        if not isinstance(layer_types, (list, tuple)):
            raise ValueError("layer_types must be a list or tuple.")
        allowed = {"full_attention", "sliding_attention"}
        invalid = [t for t in layer_types if t not in allowed]
        if invalid:
            raise ValueError(
                f"Invalid layer_types={invalid}. Allowed values: {sorted(allowed)}"
            )

    try:
        setattr(cfg_utils, "layer_type_validation", _layer_type_validation)
    except Exception:
        pass

def _find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists():
            return p
    return cur.parents[1] if len(cur.parents) > 1 else cur


_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_HERE)

MODEL_NAME = str(_REPO_ROOT / "models" / "LOCAL_MODEL")
DEVICE = "cuda"
DTYPE = "bfloat16"
MAX_NEW_TOKENS = 2048
TEMPERATURE = 0.0

# GitHub details；details。
DEFAULT_INPUT = str(_REPO_ROOT / "data" / "images")
DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "data" / "descriptions")


def build_single_image_prompt() -> str:
    """Same prompt as before - omitted for brevity"""
    return r"""You are a professional street-view annotator. Now you need to provide a as-comprehensive-as-possible, objective description of a single **pre-disaster street-view image**, and output it in a JSON structure. This will be used later for comparison with post-hurricane images.

Important principles:

1. Only describe what is **visibly present in the image**. Do not guess or infer anything outside the frame or that is not clearly visible.

2. Even though this is a pre-disaster image, you must accurately record any **pre-existing aging, minor damage, surface wetness or damp marks, piles of miscellaneous items**, etc. These details are very important for subsequent comparison.

3. Use **neutral and objective** language. Do not make subjective value judgments (e.g., "beautiful", "dilapidated", "dangerous", etc.).

4. Do not infer specific disaster types, geographic locations, city names, identities of people, ages, or economic status.

5. When describing objects, try to use **positional phrases** (such as "left side of the image", "right-center area", "foreground", "background", "near the bottom edge", "near the top edge") to help locate each object roughly.

6. For known fixed capture devices (for example, vehicle roof and mounting structure that consistently appear at the bottom of the frame), you can ignore their appearance details and do not need to repeatedly describe them.

Please output **only one JSON object**, and do not output any extra text or explanation. The required JSON structure and fields are as follows:

{
  "image_id": "string to identify the image; if no suitable ID is available, use null or an empty string",
  "overall_scene": {
    "scene_type": "string, briefly describe the scene type, e.g.: 'urban arterial road', 'residential side street', 'rural road', 'commercial street', 'industrial area road', etc.",
    "time_of_day": "string, choose one from: 'day', 'night', 'dawn_dusk', 'unknown'",
    "weather": "string, choose one from: 'clear', 'cloudy', 'rain', 'fog', 'snow', 'unknown'",
    "lighting": "string, choose one from: 'bright', 'normal', 'dim', 'unknown'",
    "free_text": "string, use 2–4 sentences to summarize the scene: for example, the location and direction of the road, whether both sides are mainly buildings or open space/vegetation, etc. You may include positional information."
  },
  "road_and_traffic": {
    "road_layout": "string, describe the position and direction of the road in the frame, e.g.: 'The road starts from the central foreground and extends toward the background, with one lane on each side' or 'The road is mainly on the right side of the frame, extending from the lower right foreground toward the background'. Positional phrases are required.",
    "road_type": "string, e.g.: 'two-way two-lane road', 'one-way multi-lane urban road', 'narrow alley', etc.",
    "lane_count_approx": "integer or null; approximate total number of lanes in both directions combined. Use null if hard to determine.",
    "surface_material": "string, e.g.: 'asphalt', 'concrete', 'stone pavers', 'unpaved dirt road', 'unknown'",
    "surface_condition": "string, describe the road surface condition in detail, including whether it is flat, presence of cracks, potholes, repair patches, etc., and include positional phrases (e.g., 'In the right foreground near the curb there is a small patch of repaired pavement').",
    "water_and_wetness": {
      "has_visible_water": "boolean, whether obvious water surfaces or standing water can be seen",
      "water_locations": "string, describe the approximate locations and extent of water/standing water, e.g.: 'There is a small shallow puddle on the left shoulder in the foreground', 'The grass area on the right side looks extensively damp with darker color'. If no water is visible, write 'No obvious standing water or water surface is observed.'",
      "water_type": "string, choose one or a combination from: 'none', 'puddle', 'large_water_area', 'wet_surface', 'unknown'",
      "drainage_visible": "string, describe whether drainage ditches, storm drains, etc. are visible, and whether they are covered by debris. If not visible, you can write 'No obvious drainage facilities observed' or 'unclear'."
    },
    "markings_and_separators": "string, describe lane markings, center lines, crosswalks, directional arrows, curbs, central medians, guardrails, etc., their approximate locations, and whether they are clear or faded."
  },
  "buildings_and_structures": {
    "distribution": "string, describe the distribution and general positions of buildings in the frame, e.g.: 'In the middle distance on the left side there is a row of low-rise buildings, and in the middle distance on the right side there is an elongated building; a few small houses are sparsely visible in the far background.'",
    "types_and_heights": "string, describe the main building types (residential, shops, factories, warehouses, walls, bridges, etc.) and approximate heights (low-rise / mid-rise / high-rise), optionally with positional phrases.",
    "facades_and_materials": "string, describe the façade materials and colors (brick, concrete, glass, metal panels, etc.), possibly segmented by left/right, foreground/background, etc.",
    "existing_minor_issues": "string, if there are pre-existing aging or minor damage, describe in detail with positional phrases, e.g.: 'On the left middle part of the image, the lower two floors of one building have noticeable stains and partial surface peeling.' If such conditions are basically not observed, you can state: 'No obvious aging or minor damage observed, only small amounts of normal surface staining.'",
    "special_elements": "string, describe noticeable elements such as signs, billboards, canopies, balconies, exterior staircases, fences, etc., including their positions and apparent condition (intact, worn, slightly deformed, etc.), and use positional phrases."
  },
  "infrastructure_and_street_furniture": {
    "elements": "string, list and describe visible infrastructure and street furniture, such as utility poles, street lights, traffic signals, traffic signs, guardrails, mailboxes, fire hydrants, bus stops, phone booths, storm drains, etc., and indicate their approximate positions.",
    "condition": "string, describe whether these facilities are generally intact, and whether there is minor leaning, rust, surface peeling, discoloration, graffiti, or obstruction by other objects.",
    "power_and_communication_lines": "string, if there are overhead power or communication lines in the frame, describe their approximate directions (e.g., 'Several lines run across the sky from the upper right toward the left side of the image') and whether there are obvious abnormalities (sagging, entanglement, etc.)."
  },
  "vegetation_ground_and_debris": {
    "vegetation_distribution": "string, describe the positions and approximate density of trees, shrubs, lawns, flowerbeds, etc., e.g.: 'In the middle distance on the left side there are several tall palm trees, and in the right background there is a denser cluster of trees.' Also mention if branches extend toward the road or block the view.",
    "non_road_ground_condition": "string, describe the ground materials and condition of non-road areas such as sidewalks, open ground, grass, bare soil, etc., including any damage, depressions, potholes, or obvious unevenness.",
    "debris_and_objects": "string, describe whether there are fallen leaves, household garbage, small piles of construction materials, scattered objects, etc., and their approximate locations and quantity level. If the environment is generally clean, you can state: 'No obvious garbage or miscellaneous debris observed, only small amounts of natural fallen leaves.'",
    "water_related_ground_features": "string, specifically describe ground features related to water, such as mud traces, damp zones, possible water flow marks, etc., and their approximate positions."
  },
  "people_and_activities": {
    "has_people": "boolean, whether any people are visible in the frame",
    "approx_count_level": "string, choose one from: 'none', 'few', 'several', 'many', 'unknown'",
    "locations": "string, describe the approximate distribution of people, such as 'There are a few pedestrians on the sidewalk on the right side of the image.' If no people are visible, write 'No pedestrians observed.'",
    "activities": "string, only describe ongoing activities (walking, waiting, talking, cycling, etc.). Do not describe age, occupation, emotions, or health status. If there are no people, write 'none'."
  },
  "viewpoint_and_layout": {
    "viewpoint_type": "string, e.g.: 'vehicle roof perspective', 'roadside perspective near human eye level', 'high vantage point bird's-eye view', 'unknown'",
    "camera_position_relative_to_road": "string, describe the approximate position of the camera relative to the road, such as 'On the vehicle roof positioned slightly to the right of the road center', 'On the sidewalk near the road edge', etc.",
    "main_layout_summary": "string, use 1–3 sentences to summarize the spatial layout of the main elements, such as the position of the road in the frame, whether both sides are buildings or open areas/vegetation, and whether there are any prominent water areas."
  },
  "uncertainty_and_occlusions": {
    "blurry_or_far_areas": "string, specify which regions have unclear details due to long distance, poor lighting, or low resolution, and provide approximate positions.",
    "occlusions": "string, specify which regions are blocked by other objects (such as trees, buildings, or capture equipment) and their approximate positions.",
    "other_notes": "string, if you have any uncertain judgments (such as suspected cracks or standing water), explain the reasons for uncertainty here, for example: 'Due to strong reflections, it is unclear whether the bright patch on the road surface is standing water or a color difference.'"
  }
}

Notes:

- You must return a valid JSON object, and the keys must exactly match the names above.

- All string fields must be enclosed in double quotes.

- Do not add any comment symbols (such as // or #) or extra text inside or outside the JSON.

- If certain information cannot be determined from the image, you may write a reasonable indication in the corresponding field, such as "unknown", "cannot determine", or null (for fields that allow null).
""".strip()


def list_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    images = [p for p in folder.iterdir() if p.suffix.lower() in exts and p.is_file()]

    def sort_key(p: Path) -> int:
        first_token = p.stem.split("_")[0]
        digits = "".join(ch for ch in first_token if ch.isdigit())
        try:
            return int(digits) if digits else int(first_token)
        except ValueError:
            return 999999

    return sorted(images, key=sort_key)


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class VLMClient:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 2048,
        temperature: float = 0.0,
        token: Optional[str] = None,
        num_workers: int = 4,  # for parallel image preprocessing
        gpu_ids: Optional[str] = None,
        max_memory_per_gpu: Optional[str] = None,
        max_shard_gpus: int = 4,
        max_memory_utilization: float = 0.90,
        reserve_gib: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.temperature = float(temperature)
        self.max_new_tokens = int(max_new_tokens)
        self.token = token
        self.num_workers = num_workers
        self.device = device

        assert torch.cuda.is_available(), "CUDA is not available!"
        _patch_flash_attn_symbol()
        _patch_layer_type_validation_symbol()

        # Determine dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.dtype = dtype

        num_gpus = torch.cuda.device_count()
        LOGGER.info("Detected %d GPU(s). Loading model with optimizations...", num_gpus)

        def _parse_gpu_ids(s: Optional[str]) -> Optional[List[int]]:
            if not s:
                return None
            parts = [p for p in re.split(r"[,\s]+", str(s).strip()) if p]
            ids: List[int] = []
            for p in parts:
                try:
                    ids.append(int(p))
                except ValueError:
                    raise ValueError(f"Invalid --gpu_ids value: {s!r}") from None
            return ids

        selected_gpu_ids = _parse_gpu_ids(gpu_ids)
        if selected_gpu_ids is None:
            selected_gpu_ids = list(range(num_gpus))
        else:
            bad = [i for i in selected_gpu_ids if i < 0 or i >= num_gpus]
            if bad:
                raise ValueError(
                    f"--gpu_ids contains invalid id(s) {bad}. Available range: [0, {num_gpus-1}]"
                )

        if int(max_shard_gpus) > 0:
            selected_gpu_ids = selected_gpu_ids[: int(max_shard_gpus)]

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_name, trust_remote_code=True, token=self.token
        )
        
        # Ensure pad_token is set
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            self.processor.tokenizer.pad_token_id = self.processor.tokenizer.eos_token_id

        # Preload and sanitize config for remote-code model compatibility.
        cfg = AutoConfig.from_pretrained(
            model_name, trust_remote_code=True, token=self.token
        )
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processor.tokenizer.eos_token_id
        for cfg_obj in (cfg, getattr(cfg, "text_config", None)):
            if cfg_obj is not None and getattr(cfg_obj, "pad_token_id", None) is None:
                try:
                    cfg_obj.pad_token_id = int(pad_token_id) if pad_token_id is not None else None
                except Exception:
                    pass

        # ========== KEY OPTIMIZATION: Flash Attention 2 ==========
        load_kwargs = {
            "config": cfg,
            "dtype": dtype,
            "device_map": "auto",
            "trust_remote_code": True,
            "token": self.token,
            "low_cpu_mem_usage": True,
            # Enable Flash Attention 2 - this is the biggest speedup!
            "attn_implementation": "flash_attention_2",
        }

        # Multi-GPU memory allocation (avoid hard-coding "20GiB" per GPU).
        # If you have a busy GPU, pass --gpu_ids to exclude it (e.g., --gpu_ids 0,2,3).
        if num_gpus >= 1 and selected_gpu_ids:
            max_memory: Dict[int, str] = {}

            # If user provided an explicit per-GPU cap, use it; otherwise infer from free VRAM.
            if max_memory_per_gpu:
                for i in selected_gpu_ids:
                    max_memory[int(i)] = str(max_memory_per_gpu)
                LOGGER.info(
                    "Using explicit max_memory_per_gpu=%s across GPUs=%s",
                    str(max_memory_per_gpu),
                    selected_gpu_ids,
                )
            else:
                util = float(max_memory_utilization)
                reserve_bytes = int(float(reserve_gib) * (1024**3))
                for i in selected_gpu_ids:
                    try:
                        free_b, total_b = torch.cuda.mem_get_info(int(i))
                        usable = int(max(0, free_b - reserve_bytes) * util)
                        usable_mib = max(256, usable // (1024**2))  # keep a small floor
                        max_memory[int(i)] = f"{usable_mib}MiB"
                        LOGGER.info(
                            "GPU %d mem: free=%.2fGiB total=%.2fGiB -> max_memory=%s (util=%.2f reserve=%.2fGiB)",
                            int(i),
                            free_b / (1024**3),
                            total_b / (1024**3),
                            max_memory[int(i)],
                            util,
                            float(reserve_gib),
                        )
                    except Exception as e:
                        LOGGER.warning("Failed to query mem on GPU %d (%s); skipping max_memory cap for it", int(i), e)

            # Setting max_memory also restricts which GPUs can be used for device_map="auto"
            # (useful when you want to exclude a busy GPU).
            if max_memory:
                load_kwargs["max_memory"] = max_memory
                LOGGER.info("Model sharding GPUs=%s with max_memory=%s", selected_gpu_ids, max_memory)

        self.model = None
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
            LOGGER.info("Model loaded with Vision2Seq + Flash Attention 2")
        except Exception as e:
            LOGGER.warning("Vision2Seq + Flash Attention 2 failed (%s), trying Vision2Seq + sdpa", e)
            load_kwargs["attn_implementation"] = "sdpa"
            try:
                self.model = AutoModelForVision2Seq.from_pretrained(model_name, **load_kwargs)
                LOGGER.info("Model loaded with Vision2Seq + SDPA")
            except Exception as e2:
                # LLaVA-OneVision-1.5-8B-Instruct on some transformers versions is exposed via CausalLM.
                LOGGER.warning("Vision2Seq path failed (%s), trying AutoModelForCausalLM fallback", e2)
                self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
                LOGGER.info("Model loaded with CausalLM fallback")

        # Enable eval mode and disable gradient computation
        self.model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile') and torch.__version__ >= "2.0":
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
                LOGGER.info("Model compiled with torch.compile()")
            except Exception as e:
                LOGGER.warning("torch.compile failed: %s", e)

        # Thread pool for parallel image loading
        self._executor = ThreadPoolExecutor(max_workers=num_workers)

    def _preprocess_single(self, img: Image.Image, prompt: str) -> Dict[str, torch.Tensor]:
        """Preprocess a single image+prompt pair"""
        content = [{"type": "text", "text": prompt}, {"type": "image", "image": img}]
        messages = [{"role": "user", "content": content}]
        chat_prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return self.processor(
            text=chat_prompt,
            images=[img],
            return_tensors="pt",
        )

    def _parallel_preprocess(self, images: List[Image.Image], prompt: str) -> List[Dict[str, torch.Tensor]]:
        """Preprocess images in parallel using thread pool"""
        futures = [
            self._executor.submit(self._preprocess_single, img, prompt)
            for img in images
        ]
        return [f.result() for f in futures]

    @torch.inference_mode()
    def infer_batch(
        self,
        prompt: str,
        post_imgs: List[Image.Image],
    ) -> List[str]:
        """
        Optimized batch inference with:
        - Parallel preprocessing
        - Proper padding
        - Efficient tensor operations
        """
        if len(post_imgs) == 0:
            return []

        # ========== Parallel preprocessing ==========
        all_inputs = self._parallel_preprocess(post_imgs, prompt)

        # ========== Efficient padding and batching ==========
        max_len = max(inp["input_ids"].shape[1] for inp in all_inputs)
        pad_token_id = self.processor.tokenizer.pad_token_id or 0

        batch_size = len(all_inputs)
        
        # Pre-allocate tensors
        batch_input_ids = torch.full(
            (batch_size, max_len), pad_token_id, dtype=torch.long
        )
        batch_attention_mask = torch.zeros(
            (batch_size, max_len), dtype=torch.long
        )

        all_pixel_values = []
        all_image_grid_thw = []

        for i, inp in enumerate(all_inputs):
            seq_len = inp["input_ids"].shape[1]
            # Right-align (left pad)
            batch_input_ids[i, max_len - seq_len:] = inp["input_ids"][0]
            batch_attention_mask[i, max_len - seq_len:] = inp["attention_mask"][0]
            
            if "pixel_values" in inp:
                all_pixel_values.append(inp["pixel_values"])
            if "image_grid_thw" in inp:
                all_image_grid_thw.append(inp["image_grid_thw"])

        # Move to GPU
        model_inputs = {
            "input_ids": batch_input_ids.to(self.model.device),
            "attention_mask": batch_attention_mask.to(self.model.device),
        }

        # Handle visual features
        if all_pixel_values:
            try:
                batch_pixel_values = torch.cat(all_pixel_values, dim=0).to(
                    device=self.model.device, dtype=self.dtype
                )
                model_inputs["pixel_values"] = batch_pixel_values
            except Exception:
                # Fallback for incompatible shapes
                model_inputs["pixel_values"] = torch.cat(
                    [pv.to(device=self.model.device, dtype=self.dtype) for pv in all_pixel_values], 
                    dim=0
                )

        if all_image_grid_thw:
            try:
                batch_grid = torch.cat(all_image_grid_thw, dim=0).to(self.model.device)
                model_inputs["image_grid_thw"] = batch_grid
            except Exception:
                model_inputs["image_grid_thw"] = torch.cat(
                    [g.to(self.model.device) for g in all_image_grid_thw], dim=0
                )

        # ========== Generation with optimized settings ==========
        generate_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": pad_token_id,
            "use_cache": True,  # Enable KV cache
        }
        
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
            generate_kwargs["do_sample"] = True
        else:
            generate_kwargs["do_sample"] = False

        with torch.cuda.amp.autocast(dtype=self.dtype):
            output_ids = self.model.generate(**model_inputs, **generate_kwargs)

        # Decode
        generated_ids = output_ids[:, max_len:]
        texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return [t.strip() for t in texts]

    @torch.inference_mode()
    def infer(self, prompt: str, post_img: Image.Image, pre_img: Optional[Image.Image] = None) -> str:
        """Single image inference - delegates to batch"""
        results = self.infer_batch(prompt, [post_img])
        return results[0] if results else ""


def strip_code_fences(text: str) -> str:
    text = text.strip()
    fence_pattern = re.compile(r"^```[a-zA-Z0-9_+-]*\s*(.*:)```$", re.DOTALL)
    m = fence_pattern.match(text)
    if m:
        return m.group(1).strip()
    return text


def extract_json_candidate(text: str) -> str:
    text = strip_code_fences(text)
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1].strip()
    return text.strip()


def try_parse_json_loose(text: str) -> Optional[Any]:
    candidate = extract_json_candidate(text)
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    fixed = candidate
    if "'" in fixed and fixed.count('"') < 2:
        fixed = fixed.replace("'", '"')

    lines: List[str] = []
    for line in fixed.splitlines():
        line = re.split(r"//|#", line, maxsplit=1)[0]
        lines.append(line)
    fixed = "\n".join(lines)
    fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)

    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        return None


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _resolve_output_path(output_dir: Path, image_path: Path) -> Path:
    base = image_path.stem
    out = output_dir / f"{base}.json"
    if not out.exists():
        return out
    i = 2
    while True:
        cand = output_dir / f"{base}_{i}.json"
        if not cand.exists():
            return cand
        i += 1


def load_images_parallel(paths: List[Path], num_workers: int = 4) -> List[tuple]:
    """Load images in parallel, returns list of (path, image) or (path, None) on failure"""
    results = []
    
    def _load(p: Path):
        try:
            return (p, load_image(p))
        except Exception as e:
            LOGGER.warning("Failed to load %s: %s", p.name, e)
            return (p, None)
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(_load, paths))
    
    return [(p, img) for p, img in results if img is not None]


def process_images(
    client: VLMClient,
    input_path: Path,
    output_dir: Path,
    limit: int = 0,
    batch_size: int = 4,
) -> None:
    if input_path.is_file():
        images: List[Path] = [input_path]
    else:
        assert input_path.exists(), f"Input directory not found: {input_path}"
        images = list_images(input_path)

    if limit and limit > 0:
        images = images[:limit]

    _ensure_dir(output_dir)

    prompt = build_single_image_prompt()
    fail_count = 0

    LOGGER.info("Found %d images.", len(images))

    # Filter already processed
    remaining_images = []
    skipped_count = 0
    for img_path in images:
        expected_output = output_dir / f"{img_path.stem}.json"
        if expected_output.exists():
            skipped_count += 1
            continue
        remaining_images.append(img_path)

    if skipped_count > 0:
        LOGGER.info("Skipping %d already processed. Remaining: %d", skipped_count, len(remaining_images))

    images = remaining_images
    if not images:
        LOGGER.info("All images already processed.")
        return

    total = len(images)
    idx = 0

    while idx < total:
        batch_paths = images[idx: idx + batch_size]

        # ========== Parallel image loading ==========
        loaded = load_images_parallel(batch_paths, num_workers=4)
        
        if not loaded:
            idx += batch_size
            continue

        loaded_paths = [p for p, _ in loaded]
        loaded_imgs = [img for _, img in loaded]

        try:
            raw_texts = client.infer_batch(prompt=prompt, post_imgs=loaded_imgs)
        except Exception as exc:
            LOGGER.warning("Batch inference failed: %s. Falling back to single.", exc)
            raw_texts = []
            for img in loaded_imgs:
                try:
                    raw_texts.append(client.infer(prompt=prompt, post_img=img))
                except Exception as single_exc:
                    LOGGER.error("Single inference failed: %s", single_exc)
                    raw_texts.append("")

        for p, raw_text in zip(loaded_paths, raw_texts):
            try:
                parsed = try_parse_json_loose(raw_text)
                if parsed is None:
                    fail_count += 1
                    LOGGER.warning("JSON parse failed for %s", p.name)
                    parsed = {"parse_error": True, "raw_output": raw_text.strip()}
                else:
                    if isinstance(parsed, dict):
                        parsed.pop("image_id", None)
                        parsed.pop("people_and_activities", None)

                out_path = _resolve_output_path(output_dir, p)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(parsed, f, ensure_ascii=False, indent=2)
            except Exception as exc:
                LOGGER.exception("Error saving %s: %s", p.name, exc)
                fail_count += 1

        done = min(idx + batch_size, total)
        LOGGER.info("Progress: %d / %d", done, total)
        idx += batch_size

    LOGGER.info("Complete. Total: %d, Parse failures: %d", len(images), fail_count)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=DEFAULT_INPUT, type=str)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, type=str)
    parser.add_argument("--limit", default=0, type=int)
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--model", default=MODEL_NAME, type=str)
    parser.add_argument("--device", default=DEVICE, type=str)
    parser.add_argument("--dtype", default=DTYPE, type=str)
    parser.add_argument("--max_new_tokens", default=MAX_NEW_TOKENS, type=int)
    parser.add_argument("--temperature", default=TEMPERATURE, type=float)
    parser.add_argument(
        "--gpu_ids",
        default="",
        type=str,
        help="Comma/space-separated CUDA device ids to use (e.g. '0,2,3'). Leave empty to allow all GPUs.",
    )
    parser.add_argument(
        "--max_memory_per_gpu",
        default="",
        type=str,
        help="Per-GPU max_memory cap passed to transformers (e.g. '16GiB' or '12000MiB'). Empty = infer from free VRAM.",
    )
    parser.add_argument(
        "--max_shard_gpus",
        default=4,
        type=int,
        help="Max number of GPUs to shard across (applied after --gpu_ids selection).",
    )
    parser.add_argument(
        "--max_memory_utilization",
        default=0.90,
        type=float,
        help="When inferring max_memory from free VRAM, use this fraction of (free - reserve).",
    )
    parser.add_argument(
        "--reserve_gib",
        default=1.0,
        type=float,
        help="When inferring max_memory from free VRAM, reserve this many GiB per GPU for safety.",
    )
    args = parser.parse_args()

    client = VLMClient(
        model_name=args.model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        token=HF_TOKEN,
        num_workers=4,
        gpu_ids=(args.gpu_ids or None),
        max_memory_per_gpu=(args.max_memory_per_gpu or None),
        max_shard_gpus=int(args.max_shard_gpus),
        max_memory_utilization=float(args.max_memory_utilization),
        reserve_gib=float(args.reserve_gib),
    )

    process_images(
        client=client,
        input_path=Path(args.input),
        output_dir=Path(args.output_dir),
        limit=int(args.limit),
        batch_size=int(args.batch_size),
    )


if __name__ == "__main__":
    main()