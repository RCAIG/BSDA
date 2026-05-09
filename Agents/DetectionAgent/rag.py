#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/rag.py

details A details RAG details（details“details，details”）：

- details：
  - details SentenceTransformer details query -> FAISS（IndexFlatIP，details cosine）details
  - details：
    - gov: faiss_gov_text.index + faiss_gov_text_meta.jsonl + gov_docs_chunks.jsonl（details/details/details）
    - street: street_cases.faiss.index + street_cases.faiss_meta.jsonl + street_cases.jsonl（details）
  - details：details faiss/sentence-transformers details，details（details corpus jsonl）

- details（details Detection details A）：
  1) details Detection details“details query details”details：
     - build_detection_query(...)：details hazard_type / component / change_desc details
       details query，details：
         “details/details？”

  2) search() details query details；details search_for_detection_change()，
     details Detection Agent details“details”details，details。

  3) details：
     [
       {
         "text": "...details/details...",
         "source": "gov" | "street",
         "meta": {
            "doc_id": "...",
            "chunk_id": "...",
            "page": 12,
            "title": "...",
            "section": "...",
         }
       },
       ...
     ]
     details LLM details“details”details。

details（details Detection Agent details）：

    rag = DamageFeatureRAG(artifacts_dir="artifacts", corpus="gov")

    hits = rag.search_for_detection_change(
        hazard_type="flood",
        component="building facade near ground level",
        change_desc="post-disaster description reports new continuous water stains \
                     and mud accumulation at the bottom of the exterior wall \
                     that were absent in pre-disaster state",
        extra_context="urban area, building close to river, event date 2024-08-01"
    )

    # details hits details context details LLM，details：
    #   - details likely / possible / unlikely / unknown

"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable


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
    with path.open("r", encoding="utf-8") as f:
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
    """
    details：details/details/details，details。
    """
    t = _safe_str(text)
    if not t:
        return []
    # details or details
    return re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9]+", t.lower())


def _truncate(text: str, max_chars: int = 1400) -> str:
    t = _safe_str(text)
    if len(t) <= max_chars:
        return t
    return t[: max_chars - 20].rstrip() + " ...[details]"


def _is_garbage_text(text: str, *, min_alnum: int = 30) -> bool:
    """
    details PDF details“details/details”details。
    details：details（details）details，details。
    """
    t = _safe_str(text)
    if not t:
        return True
    kept = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "", t)
    return len(kept) < int(min_alnum)


@dataclass
class DamageFeatureRAG:
    """
    details“details”details RAG details。

    - artifacts_dir: details FAISS details JSONL details。
    - corpus: "gov"（details/details/details） or "street"（details）。
    - embed_model: details SentenceTransformer details。
    - max_chars_per_hit: details（details）。

    details Detection Agent details：
        search_for_detection_change(...)
    details search(query)，details query details。
    """
    artifacts_dir: str
    corpus: str = "gov"  # "gov" or "street"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
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

    def __post_init__(self) -> None:
        self._artifacts = Path(self.artifacts_dir)
        self._index = None
        self._meta: List[Dict[str, Any]] = []
        self._corpus_by_doc_id: Dict[str, str] = {}
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        self._entries: List[Dict[str, Any]] = []
        self._entry_by_id: Dict[str, Dict[str, Any]] = {}
        self._encoder = None
        self._cross_encoder = None
        self._bm25_docs: List[Dict[str, Any]] = []
        self._bm25_doc_len: List[int] = []
        self._bm25_tf: List[Counter[str]] = []
        self._bm25_df: Dict[str, int] = {}
        self._bm25_avgdl: float = 0.0

        # details corpus details
        corpus = _safe_str(self.corpus).lower() or "gov"
        if corpus not in {"gov", "street"}:
            corpus = "gov"
        self.corpus = corpus  # details

        if corpus == "gov":
            faiss_index_name = "faiss_gov_text.index"
            faiss_meta_name = "faiss_gov_text_meta.jsonl"
            corpus_jsonl_name = "gov_docs_chunks.jsonl"
        else:
            faiss_index_name = "street_cases.faiss.index"
            faiss_meta_name = "street_cases.faiss_meta.jsonl"
            corpus_jsonl_name = "street_cases.jsonl"

        # 1) details FAISS details + meta
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

        # 2) details（details & details）
        corpus_path = self._artifacts / corpus_jsonl_name
        if corpus_path.exists():
            if corpus == "gov":
                # gov_docs_chunks.jsonl: chunk_id -> chunk_obj（details text/raw_path/page/title/section）
                for row in _iter_jsonl(corpus_path):
                    cid = _safe_str(row.get("chunk_id", ""))
                    if cid:
                        self._chunks_by_id[cid] = row
            else:
                # street_cases.jsonl: doc_id -> text（details title）
                for row in _iter_jsonl(corpus_path):
                    doc_id = _safe_str(row.get("doc_id", ""))
                    text = _safe_str(row.get("text", "")) or _safe_str(row.get("title", ""))
                    if doc_id and text:
                        self._corpus_by_doc_id[doc_id] = text

        self._build_entries()
        self._build_bm25_index()

    def _get_encoder(self):
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
        if self.corpus == "gov" and self._chunks_by_id:
            for cid, ch in self._chunks_by_id.items():
                text = _safe_str(ch.get("text", ""))
                if not text or _is_garbage_text(text):
                    continue
                doc_id = _safe_str(ch.get("doc_id", ""))
                title = _safe_str(ch.get("title", ""))
                section = _safe_str(ch.get("section", ""))
                page = ch.get("page", None)
                raw_path = _safe_str(ch.get("raw_path", ""))
                header = f"[gov] {doc_id} | {cid}"
                if page is not None:
                    header += f" | page={page}"
                if raw_path:
                    header += f" | {raw_path}"
                meta_line = f"title={title} | section={section}".strip()
                merged_text = header + "\n" + meta_line + "\n" + text
                meta = {
                    "doc_id": doc_id,
                    "chunk_id": cid,
                    "page": page,
                    "raw_path": raw_path,
                    "title": title,
                    "section": section,
                }
                entries.append(
                    {
                        "entry_id": cid or doc_id,
                        "text": _truncate(merged_text, int(self.max_chars_per_hit)),
                        "source": "gov",
                        "meta": meta,
                    }
                )
        else:
            if self._corpus_by_doc_id:
                for doc_id, text in self._corpus_by_doc_id.items():
                    if not text:
                        continue
                    entries.append(
                        {
                            "entry_id": doc_id,
                            "text": _truncate(text, int(self.max_chars_per_hit)),
                            "source": "street",
                            "meta": {"doc_id": doc_id},
                        }
                    )
            elif self._meta:
                for m in self._meta:
                    doc_id = _safe_str(m.get("doc_id", ""))
                    text = _safe_str(m.get("text", "")) or _safe_str(m.get("title", ""))
                    if not doc_id or not text:
                        continue
                    entries.append(
                        {
                            "entry_id": doc_id,
                            "text": _truncate(text, int(self.max_chars_per_hit)),
                            "source": "street",
                            "meta": {"doc_id": doc_id},
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
            text = _safe_str(entry.get("text", ""))
            tokens = _tokenize_for_match(text)
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
        q = encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
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

    # ------------------------------------------------------------------
    # details：details
    # ------------------------------------------------------------------
    def _search_fallback_keyword(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        q_tokens = set(_tokenize_for_match(query))
        if not q_tokens:
            return []

        scored: List[tuple[int, str, Dict[str, Any]]] = []

        # gov: details chunk text；street: details case text
        if self.corpus == "gov" and self._chunks_by_id:
            for cid, ch in self._chunks_by_id.items():
                text = _safe_str(ch.get("text", ""))
                if _is_garbage_text(text):
                    continue
                t_tokens = set(_tokenize_for_match(text))
                score = len(q_tokens.intersection(t_tokens))
                if score > 0:
                    scored.append((score, text, ch))
        else:
            for doc_id, text in self._corpus_by_doc_id.items():
                if not text:
                    continue
                t_tokens = set(_tokenize_for_match(text))
                score = len(q_tokens.intersection(t_tokens))
                if score > 0:
                    scored.append((score, text, {"doc_id": doc_id}))

        scored.sort(key=lambda x: x[0], reverse=True)

        out: List[Dict[str, Any]] = []
        for score, text, meta in scored[: int(top_k)]:
            hit = {
                "text": _truncate(text, int(self.max_chars_per_hit)),
                "source": self.corpus,
                "meta": dict(meta),
            }
            # details（details）
            hit["meta"]["score_hint"] = float(score)
            out.append(hit)
        return out

    # ------------------------------------------------------------------
    # details：details search（details query）
    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        details：details query details RAG details。

        details“details/details”：
        [
          {
            "text": "...",
            "source": "gov" | "street",
            "meta": { ... }  # doc_id / chunk_id / page / title / section / score_hint details
          },
          ...
        ]
        """
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
                    "source": _safe_str(entry.get("source", self.corpus)),
                    "meta": meta_out,
                }
            )

        return out if out else self._search_fallback_keyword(query, int(top_k))

    # ------------------------------------------------------------------
    # Detection details A：details
    # ------------------------------------------------------------------
    @staticmethod
    def build_detection_query(
        *,
        hazard_type: Optional[str],
        component: Optional[str],
        change_desc: str,
        pre_desc: Optional[str] = None,
        post_desc: Optional[str] = None,
        extra_context: Optional[str] = None,
        language: str = "auto",
    ) -> str:
        """
        details“details + details + details”details，details
        details A details query，details“details”。

        - hazard_type: "earthquake" / "flood" / "wind" / "typhoon" / "wildfire" / details
        - component: details，details "building facade near ground level"
        - change_desc: details Detection Agent details（details/details）
        - pre_desc / post_desc: details，details/details
        - extra_context: details，details "urban area, near river, event date 2024-08-01"
        - language: "auto" / "en" / "zh"：
            - "auto": details change_desc/pre_desc/post_desc details，details；
            - details query details，details "en"，details。

        details：details self.search(...) details query details。
        """

        hz = _safe_str(hazard_type)
        comp = _safe_str(component)
        chg = _safe_str(change_desc)
        pre = _safe_str(pre_desc)
        post = _safe_str(post_desc)
        ctx = _safe_str(extra_context)

        # details prompt details“details”，details：
        #   “details + details + details，details/details”
        # details/details。
        if language.lower() == "zh":
            parts = []
            if hz:
                parts.append(f"details：{hz}")
            if comp:
                parts.append(f"details：{comp}")
            if pre:
                parts.append(f"details：{pre}")
            if post:
                parts.append(f"details：{post}")
            if chg:
                parts.append(f"details/details：{chg}")
            if ctx:
                parts.append(f"details（details/details/details）：{ctx}")

            parts.append(
                "details：details，details"
                "details、details、details/details。"
            )
            return "\n".join(parts).strip()

            # details“details vs details”，details。

        else:
            # details/auto，details“details”details，details embedding。
            parts_en: List[str] = []

            if hz:
                parts_en.append(f"Hazard type: {hz}.")
            if comp:
                parts_en.append(f"Relevant component or scene: {comp}.")
            if pre:
                parts_en.append(f"Pre-disaster state: {pre}.")
            if post:
                parts_en.append(f"Post-disaster state: {post}.")
            if chg:
                parts_en.append(
                    "Observed change between pre- and post-disaster descriptions: "
                    f"{chg}."
                )
            if ctx:
                parts_en.append(f"Additional context (location/time/environment): {ctx}.")

            parts_en.append(
                "Retrieval goal: find guidance, rules, or typical/atypical damage "
                "patterns for this hazard that describe whether such a change is "
                "commonly caused by the hazard, is a typical benign (non-disaster) "
                "phenomenon, or is ambiguous and prone to misclassification."
            )

            return " ".join(parts_en).strip()

    def search_for_detection_change(
        self,
        *,
        hazard_type: Optional[str],
        component: Optional[str],
        change_desc: str,
        pre_desc: Optional[str] = None,
        post_desc: Optional[str] = None,
        extra_context: Optional[str] = None,
        language: str = "auto",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Detection details：

        details：
          - hazard_type: hurricane）
          - component: details/details（details：details、details、details、details、details）
          - change_desc: details Detection Agent details“details”details
          - pre_desc/post_desc: details/details
          - extra_context: details/details/details，details
          - language: "auto"/"en"/"zh"（details query details）
          - top_k: details

        details：
          - details self.search() details：
            [
              {
                "text": "...",
                "source": "gov" | "street",
                "meta": { "doc_id": ..., "chunk_id": ..., "score_hint": ... }
              },
              ...
            ]

        details：
          - Detection Agent details“details”details，details，
            details LLM details“disaster_related likely/possible/unlikely/unknown”details。
        """
        q = self.build_detection_query(
            hazard_type=hazard_type,
            component=component,
            change_desc=change_desc,
            pre_desc=pre_desc,
            post_desc=post_desc,
            extra_context=extra_context,
            language=language,
        )
        return self.search(q, top_k=top_k)