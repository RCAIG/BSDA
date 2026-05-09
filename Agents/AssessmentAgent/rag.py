#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Assessment Agent RAG module (placed under AssessmentAgent/ as requested).

We expose:
- search_gov_rules(query_text, top_k): government/standard rule snippets
- search_history_cases(query_text, top_k): historical case snippets with damage labels

Implementation:
- Reuses the same artifacts directory layout under repo_root/RAG/artifacts
- Uses FAISS + SentenceTransformer if available, otherwise falls back to keyword matching.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from AssessmentAgent.config import RAG_ARTIFACTS_DIR


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    yield obj
            except Exception:
                continue


def _tokenize_for_match(text: str) -> List[str]:
    t = _safe_str(text)
    if not t:
        return []
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", t.lower())


def _truncate(text: str, max_chars: int = 1400) -> str:
    """
    Kept for backward compatibility, but no longer truncates by default.
    (User wants full-input / full-context behavior; truncation is treated as a display concern.)
    """
    return _safe_str(text)


def _is_garbage_text(text: str, *, min_alnum: int = 30) -> bool:
    t = _safe_str(text)
    if not t:
        return True
    kept = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", t)
    return len(kept) < int(min_alnum)


@dataclass
class _ArtifactsRAG:
    """
    Minimal RAG wrapper over existing artifacts.

    corpus:
      - "gov": gov docs chunks
      - "street": historical street cases (with labels)
    """

    artifacts_dir: str
    corpus: str
    embed_model: str
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_chars_per_hit: int = 1400
    dense_weight: float = 0.6
    bm25_weight: float = 0.4
    rerank_weight: float = 0.7
    dense_pool_size: int = 20
    bm25_pool_size: int = 20
    rerank_pool_size: int = 20
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    _encoder: Any = field(default=None, init=False, repr=False)
    _cross_encoder: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self._artifacts = Path(self.artifacts_dir)
        corpus = _safe_str(self.corpus).lower()
        if corpus not in {"gov", "street"}:
            corpus = "gov"
        self.corpus = corpus

        self._index = None
        self._meta: List[Dict[str, Any]] = []
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self._cases_by_doc_id: Dict[str, Dict[str, Any]] = {}
        self._entries: List[Dict[str, Any]] = []
        self._entry_by_id: Dict[str, Dict[str, Any]] = {}
        self._bm25_docs: List[Dict[str, Any]] = []
        self._bm25_doc_len: List[int] = []
        self._bm25_tf: List[Counter[str]] = []
        self._bm25_df: Dict[str, int] = {}
        self._bm25_avgdl: float = 0.0

        if corpus == "gov":
            faiss_index_name = "faiss_gov_text.index"
            faiss_meta_name = "faiss_gov_text_meta.jsonl"
            corpus_jsonl_name = "gov_docs_chunks.jsonl"
        else:
            faiss_index_name = "street_cases.faiss.index"
            faiss_meta_name = "street_cases.faiss_meta.jsonl"
            corpus_jsonl_name = "street_cases.jsonl"

        # 1) Try FAISS index + meta
        try:
            import faiss  # type: ignore

            idx_path = self._artifacts / faiss_index_name
            meta_path = self._artifacts / faiss_meta_name
            if idx_path.exists() and meta_path.exists():
                self._index = faiss.read_index(str(idx_path))
                self._meta = list(_iter_jsonl(meta_path))
        except Exception:
            self._index = None
            self._meta = []

        # 2) Load corpus JSONL for text backfill + fallback
        corpus_path = self._artifacts / corpus_jsonl_name
        if corpus_path.exists():
            if corpus == "gov":
                for row in _iter_jsonl(corpus_path):
                    cid = _safe_str(row.get("chunk_id", ""))
                    if cid:
                        self._chunks_by_id[cid] = row
            else:
                for row in _iter_jsonl(corpus_path):
                    doc_id = _safe_str(row.get("doc_id", ""))
                    if doc_id:
                        self._cases_by_doc_id[doc_id] = row

        self._build_entries()
        self._build_bm25_index()

    def _get_encoder(self):
        """
        Lazy-load SentenceTransformer once per RAG instance.
        """
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._encoder = SentenceTransformer(self.embed_model)
            return self._encoder
        except Exception:
            self._encoder = None
            return None

    def _get_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._cross_encoder = CrossEncoder(self.cross_encoder_model)
            return self._cross_encoder
        except Exception:
            self._cross_encoder = None
            return None

    def _build_entries(self) -> None:
        entries: List[Dict[str, Any]] = []
        if self.corpus == "gov":
            for cid, ch in self._chunks_by_id.items():
                text = _safe_str(ch.get("text", ""))
                if not text or _is_garbage_text(text):
                    continue
                meta = {
                    "doc_id": _safe_str(ch.get("doc_id", "")),
                    "chunk_id": cid,
                    "page": ch.get("page", None),
                    "raw_path": _safe_str(ch.get("raw_path", "")),
                    "title": _safe_str(ch.get("title", "")),
                    "section": _safe_str(ch.get("section", "")),
                }
                entries.append(
                    {
                        "entry_id": cid or meta["doc_id"],
                        "text": _truncate(text, int(self.max_chars_per_hit)),
                        "meta": meta,
                    }
                )
        else:
            for doc_id, case in self._cases_by_doc_id.items():
                text = _safe_str(case.get("text", "")) or _safe_str(case.get("title", ""))
                if not text:
                    continue
                entries.append(
                    {
                        "entry_id": doc_id,
                        "text": _truncate(text, int(self.max_chars_per_hit)),
                        "meta": dict(case),
                    }
                )
        self._entries = entries
        self._entry_by_id = {e["entry_id"]: e for e in entries if _safe_str(e.get("entry_id", ""))}

    def _build_bm25_index(self) -> None:
        docs: List[Dict[str, Any]] = []
        doc_lens: List[int] = []
        tfs: List[Counter[str]] = []
        df: defaultdict[str, int] = defaultdict(int)
        for entry in self._entries:
            tokens = _tokenize_for_match(_safe_str(entry.get("text", "")))
            if not tokens:
                continue
            tf = Counter(tokens)
            docs.append({"entry_id": entry["entry_id"], "tokens": tokens})
            doc_lens.append(len(tokens))
            tfs.append(tf)
            for tok in tf.keys():
                df[tok] += 1
        self._bm25_docs = docs
        self._bm25_doc_len = doc_lens
        self._bm25_tf = tfs
        self._bm25_df = dict(df)
        self._bm25_avgdl = (sum(doc_lens) / len(doc_lens)) if doc_lens else 0.0

    def _dense_search_candidates(self, query: str, top_n: int) -> Dict[str, float]:
        if self._index is None or not self._meta:
            return {}
        encoder = self._get_encoder()
        if encoder is None:
            return {}
        q = encoder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        scores, ids = self._index.search(q, int(top_n))
        out: Dict[str, float] = {}
        for s, idx in zip(scores[0].tolist(), ids[0].tolist()):
            if idx < 0 or idx >= len(self._meta):
                continue
            m = self._meta[int(idx)]
            entry_id = _safe_str(m.get("chunk_id", "")) or _safe_str(m.get("doc_id", ""))
            if not entry_id or entry_id not in self._entry_by_id:
                continue
            prev = out.get(entry_id)
            if prev is None or float(s) > prev:
                out[entry_id] = float(s)
        return out

    def _bm25_search_candidates(self, query: str, top_n: int) -> Dict[str, float]:
        q_tokens = _tokenize_for_match(query)
        if not q_tokens or not self._bm25_docs or self._bm25_avgdl <= 0:
            return {}
        scores: List[tuple[float, str]] = []
        N = len(self._bm25_docs)
        avgdl = self._bm25_avgdl
        for idx, doc in enumerate(self._bm25_docs):
            tf = self._bm25_tf[idx]
            dl = self._bm25_doc_len[idx]
            score = 0.0
            for tok in q_tokens:
                freq = tf.get(tok, 0)
                if freq <= 0:
                    continue
                df = self._bm25_df.get(tok, 0)
                idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                denom = freq + self.bm25_k1 * (1.0 - self.bm25_b + self.bm25_b * dl / avgdl)
                score += idf * (freq * (self.bm25_k1 + 1.0)) / max(denom, 1e-9)
            if score > 0.0:
                scores.append((float(score), doc["entry_id"]))
        scores.sort(key=lambda x: x[0], reverse=True)
        return {entry_id: score for score, entry_id in scores[: int(top_n)]}

    @staticmethod
    def _normalize_score_map(score_map: Dict[str, float]) -> Dict[str, float]:
        if not score_map:
            return {}
        vals = list(score_map.values())
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in score_map}
        return {k: (float(v) - lo) / (hi - lo) for k, v in score_map.items()}

    def _search_fallback_keyword(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_tokens = set(_tokenize_for_match(query))
        if not q_tokens:
            return []

        scored: List[tuple[int, str, Dict[str, Any]]] = []
        if self.corpus == "gov":
            for cid, ch in self._chunks_by_id.items():
                text = _safe_str(ch.get("text", ""))
                if _is_garbage_text(text):
                    continue
                t_tokens = set(_tokenize_for_match(text))
                score = len(q_tokens.intersection(t_tokens))
                if score > 0:
                    scored.append((score, text, ch))
        else:
            for doc_id, case in self._cases_by_doc_id.items():
                text = _safe_str(case.get("text", "")) or _safe_str(case.get("title", ""))
                if not text:
                    continue
                t_tokens = set(_tokenize_for_match(text))
                score = len(q_tokens.intersection(t_tokens))
                if score > 0:
                    scored.append((score, text, case))

        scored.sort(key=lambda x: x[0], reverse=True)
        out: List[Dict[str, Any]] = []
        for score, text, obj in scored[: int(top_k)]:
            out.append(
                {
                    "text": _truncate(text, int(self.max_chars_per_hit)),
                    "meta": dict(obj),
                    "score_hint": float(score),
                }
            )
        return out

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query = _safe_str(query)
        if not query:
            return []
        if not self._entries:
            return self._search_fallback_keyword(query, int(top_k))

        dense_scores = self._dense_search_candidates(query, max(int(top_k) * 4, int(self.dense_pool_size)))
        bm25_scores = self._bm25_search_candidates(query, max(int(top_k) * 4, int(self.bm25_pool_size)))
        if not dense_scores and not bm25_scores:
            return self._search_fallback_keyword(query, int(top_k))

        dense_norm = self._normalize_score_map(dense_scores)
        bm25_norm = self._normalize_score_map(bm25_scores)
        candidate_ids = list(dict.fromkeys(list(dense_scores.keys()) + list(bm25_scores.keys())))
        hybrid_scores: Dict[str, float] = {}
        for entry_id in candidate_ids:
            hybrid_scores[entry_id] = (
                self.dense_weight * dense_norm.get(entry_id, 0.0)
                + self.bm25_weight * bm25_norm.get(entry_id, 0.0)
            )

        ranked_candidate_ids = sorted(
            hybrid_scores.keys(),
            key=lambda eid: hybrid_scores[eid],
            reverse=True,
        )[: max(int(top_k) * 4, int(self.rerank_pool_size))]

        ce_scores: Dict[str, float] = {}
        cross_encoder = self._get_cross_encoder()
        if cross_encoder is not None and ranked_candidate_ids:
            try:
                pairs = [
                    [query, _safe_str(self._entry_by_id[eid].get("text", ""))]
                    for eid in ranked_candidate_ids
                ]
                preds = cross_encoder.predict(pairs)
                ce_scores = {eid: float(score) for eid, score in zip(ranked_candidate_ids, preds)}
            except Exception:
                ce_scores = {}

        ce_norm = self._normalize_score_map(ce_scores)
        final_scores: Dict[str, float] = {}
        for entry_id in ranked_candidate_ids:
            if ce_norm:
                final_scores[entry_id] = (
                    self.rerank_weight * ce_norm.get(entry_id, 0.0)
                    + (1.0 - self.rerank_weight) * hybrid_scores.get(entry_id, 0.0)
                )
            else:
                final_scores[entry_id] = hybrid_scores.get(entry_id, 0.0)

        out: List[Dict[str, Any]] = []
        for entry_id in sorted(final_scores.keys(), key=lambda eid: final_scores[eid], reverse=True)[: int(top_k)]:
            entry = dict(self._entry_by_id[entry_id])
            meta_out = dict(entry.get("meta", {}))
            meta_out.update(
                {
                    "dense_score": dense_scores.get(entry_id),
                    "bm25_score": bm25_scores.get(entry_id),
                    "hybrid_score": hybrid_scores.get(entry_id),
                    "rerank_score": ce_scores.get(entry_id),
                    "score_hint": final_scores.get(entry_id, 0.0),
                }
            )
            out.append(
                {
                    "text": _safe_str(entry.get("text", "")),
                    "meta": meta_out,
                    "score_hint": float(final_scores.get(entry_id, 0.0)),
                }
            )
        return out if out else self._search_fallback_keyword(query, int(top_k))


_GOV_RAG: Optional[_ArtifactsRAG] = None
_HISTORY_RAG: Optional[_ArtifactsRAG] = None


def _get_gov_rag() -> _ArtifactsRAG:
    global _GOV_RAG
    if _GOV_RAG is None:
        _GOV_RAG = _ArtifactsRAG(
            artifacts_dir=RAG_ARTIFACTS_DIR,
            corpus="gov",
            embed_model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
    return _GOV_RAG


def _get_history_rag() -> _ArtifactsRAG:
    global _HISTORY_RAG
    if _HISTORY_RAG is None:
        _HISTORY_RAG = _ArtifactsRAG(
            artifacts_dir=RAG_ARTIFACTS_DIR,
            corpus="street",
            embed_model="sentence-transformers/all-MiniLM-L6-v2",
        )
    return _HISTORY_RAG


def search_gov_rules(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    details，details query_text details‘details’details。

    details，details dict，details：
    - id: str
    - source_title: str
    - source_section: Optional[str]
    - text: str
    - meta: dict
    """
    rag = _get_gov_rag()
    hits = rag.search(query_text, top_k=int(top_k))
    out: List[Dict[str, Any]] = []
    for h in hits:
        meta = dict(h.get("meta", {}) or {})
        meta.setdefault("source", "gov")
        out.append(
            {
                "id": _safe_str(meta.get("chunk_id", "")) or _safe_str(meta.get("doc_id", "")) or "",
                "source_title": _safe_str(meta.get("title", "")) or _safe_str(meta.get("doc_id", "")) or "",
                "source_section": _safe_str(meta.get("section", "")) or None,
                "text": _safe_str(h.get("text", "")),
                "meta": meta,
            }
        )
    return out


def search_history_cases(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    details，details query_text details‘details’，details。

    details，details dict，details：
    - id: str
    - case_title: str
    - damage_level: str
    - text: str
    - meta: dict
    """
    rag = _get_history_rag()
    hits = rag.search(query_text, top_k=int(top_k))
    out: List[Dict[str, Any]] = []
    for h in hits:
        meta = dict(h.get("meta", {}) or {})
        meta.setdefault("source", "street")
        raw_label = _safe_str(meta.get("primary_label", "")) or _safe_str(meta.get("label", ""))
        dmg = raw_label.strip().lower()
        # normalize common variants
        if dmg in {"slight", "light"}:
            dmg = "minor"
        elif dmg in {"moderate"}:
            dmg = "moderate"
        elif dmg in {"severe", "major"}:
            dmg = "severe"
        out.append(
            {
                "id": _safe_str(meta.get("doc_id", "")) or "",
                "case_title": _safe_str(meta.get("title", "")) or "",
                "damage_level": dmg or "unknown",
                "text": _safe_str(h.get("text", "")),
                "meta": meta,
            }
        )
    return out

