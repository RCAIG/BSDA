#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Set, Tuple


_GLOBAL_LOCAL_LLM_CACHE: Dict[str, Tuple[object, object, object]] = {}


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


def _sanitize_parallel_plans(obj: object, seen: Set[int] | None = None) -> None:
    """
    Recursively strip tensor/pipeline parallel plans from config objects.
    This avoids transformers v5 plan-validation issues with LOCAL_MODEL configs.
    """
    if obj is None:
        return
    if seen is None:
        seen = set()
    oid = id(obj)
    if oid in seen:
        return
    seen.add(oid)

    for attr in ("base_model_tp_plan", "base_model_pp_plan", "_tp_plan", "_pp_plan"):
        if hasattr(obj, attr):
            try:
                setattr(obj, attr, {})
            except Exception:
                pass

    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, (str, int, float, bool, bytes, bytearray, type(None))):
                continue
            if isinstance(v, (list, tuple, set)):
                for item in v:
                    _sanitize_parallel_plans(item, seen)
            elif isinstance(v, dict):
                for item in v.values():
                    _sanitize_parallel_plans(item, seen)
            else:
                _sanitize_parallel_plans(v, seen)


def get_shared_llm(model_path: str) -> Tuple[object, object, object]:
    global _GLOBAL_LOCAL_LLM_CACHE

    model_path = str(model_path)
    if model_path in _GLOBAL_LOCAL_LLM_CACHE:
        return _GLOBAL_LOCAL_LLM_CACHE[model_path]

    try:
        import torch  # type: ignore
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer  # type: ignore
    except Exception as e:
        raise RuntimeError("Missing dependencies. Please install: pip install -U transformers torch") from e

    _patch_flash_attn_symbol()
    _patch_layer_type_validation_symbol()

    print(f"[INFO] Loading shared local LLM model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # LOCAL_MODEL custom configs on transformers v5 may fail on parallel plan validation.
    _sanitize_parallel_plans(cfg)

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    for cfg_obj in (cfg, getattr(cfg, "text_config", None)):
        if cfg_obj is not None and getattr(cfg_obj, "pad_token_id", None) is None:
            try:
                cfg_obj.pad_token_id = pad_token_id
            except Exception:
                pass

    lower_path = model_path.lower()
    is_internvl = "internvl" in lower_path
    attn_impl = "eager" if is_internvl else "sdpa"
    model_loader = AutoModel if is_internvl else AutoModelForCausalLM

    common_kwargs = dict(
        config=cfg,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    try:
        model = model_loader.from_pretrained(model_path, attn_implementation=attn_impl, **common_kwargs)
    except Exception as e:
        # Final fallback: retry without forcing attention backend.
        print(f"[WARN] Load with attn_implementation='{attn_impl}' failed, retry default: {e}")
        model = model_loader.from_pretrained(model_path, **common_kwargs)

    if is_internvl:
        # LOCAL_MODEL's generate() asserts img_context_token_id is initialized.
        img_ctx_id = None
        for tok in ("<IMG_CONTEXT>", "<image>", "<img_context>"):
            try:
                tid = tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            if isinstance(tid, int) and tid >= 0:
                img_ctx_id = tid
                break
        if img_ctx_id is None:
            try:
                vocab = tokenizer.get_vocab()
                img_ctx_id = vocab.get("<IMG_CONTEXT>")
            except Exception:
                img_ctx_id = None
        if img_ctx_id is not None:
            for obj in (model, getattr(model, "language_model", None), getattr(model, "model", None)):
                if obj is None:
                    continue
                try:
                    setattr(obj, "img_context_token_id", int(img_ctx_id))
                except Exception:
                    pass

    try:
        model.eval()
    except Exception:
        pass

    _GLOBAL_LOCAL_LLM_CACHE[model_path] = (tokenizer, model, torch)
    return tokenizer, model, torch

