#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/config.py
details（details/details，details）。
"""

# details“details Agent details”：
# - details repo details（models/、RAG/、details）details Agent details（output/）
# - details
from __future__ import annotations

from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    """
    details repo details。
    details：details `models/` details `RAG/`。
    """
    cur = start.resolve()
    for p in [cur] + list(cur.parents):
        if (p / "models").exists() and (p / "RAG").exists():
            return p
    # fallback: details（details）
    return cur.parents[1] if len(cur.parents) > 1 else cur


_AGENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _find_repo_root(_AGENT_DIR)

# details：details GitHub details。
PRE_DESC_DIR = str(_REPO_ROOT / "data" / "pre_descriptions")

# details
POST_DESC_DIR = str(_REPO_ROOT / "data" / "post_descriptions")

# RAG artifacts details（details/details）
RAG_ARTIFACTS_DIR = str(_REPO_ROOT / "RAG" / "artifacts")

# details
OUTPUT_DIR = str(_AGENT_DIR / "output")

# details（details/details）
PAIR_MAP_CSV = str(_REPO_ROOT / "data" / "pair_map.csv")

# details（details）
LOCAL_LLM_MODEL_PATH = str(_REPO_ROOT / "models" / "LOCAL_MODEL")

# RAG details
RAG_TOP_K = 2

# RAG details：
# - "gov": details faiss_gov_text.index + faiss_gov_text_meta.jsonl + gov_docs_chunks.jsonl（details/details）
# - "street": details street_cases.faiss.index + street_cases.faiss_meta.jsonl + street_cases.jsonl（details）
RAG_CORPUS = "gov"

# RAG（FAISS）embedding details：details，details
# gov（rag_query_local.py details）
RAG_EMBED_MODEL_GOV = r"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# street（prepare_street_cases.py details）
RAG_EMBED_MODEL_STREET = r"sentence-transformers/all-MiniLM-L6-v2"

# details（details，details/details；details）
MAX_NEW_TOKENS = 800

# details：
# - Step1（details）details，details JSON details
# - Step3（details）details（details rag_hits），details（details），details
MAX_NEW_TOKENS_STEP1 = 2200
MAX_NEW_TOKENS_STEP3 = 2000

