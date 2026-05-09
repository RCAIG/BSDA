#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/rag_ws.py

RAG-WS (RAG with Web Search) details。

details：
1. Internal RAG: details（details）
2. External Web Search: details（details）

details MoRA-RAG details Agentic LLM Structure：
- Evidence Evaluator: details RAG details
- Query Rewriter: details
- Online Search: details Fallback details

details：
- details Fallback，details
- details RAG details
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .ddgs_search import DDGSSearchTool


# ----------------------------------------------------------------------
# details
# ----------------------------------------------------------------------

@dataclass
class RAGResult:
    """details RAG details"""
    chunks: List[Dict[str, Any]]
    scores: List[float] = field(default_factory=list)
    query: str = ""
    source: str = "internal"  # "internal" | "web" | "internal+web"
    
    @property
    def top1_score(self) -> float:
        return self.scores[0] if self.scores else 0.0
    
    @property
    def hit_count(self) -> int:
        return len(self.chunks)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": self.chunks,
            "scores": self.scores,
            "query": self.query,
            "source": self.source,
            "top1_score": self.top1_score,
            "hit_count": self.hit_count,
        }


@dataclass
class EvidenceEvaluation:
    """details"""
    is_sufficient: bool
    score: float  # 0-1，details
    missing_aspects: List[str] = field(default_factory=list)
    reason: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sufficient": self.is_sufficient,
            "score": self.score,
            "missing_aspects": self.missing_aspects,
            "reason": self.reason,
        }


@dataclass
class RAGWSResult:
    """RAG-WS details"""
    chunks: List[Dict[str, Any]]
    source: str  # "internal" | "internal+web"
    is_sufficient: bool
    evaluation: EvidenceEvaluation
    attempts: int = 1
    web_search_triggered: bool = False
    web_results: List[Dict[str, Any]] = field(default_factory=list)
    query_history: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunks": self.chunks,
            "source": self.source,
            "is_sufficient": self.is_sufficient,
            "evaluation": self.evaluation.to_dict(),
            "attempts": self.attempts,
            "web_search_triggered": self.web_search_triggered,
            "web_results_count": len(self.web_results),
            "query_history": self.query_history,
        }


# ----------------------------------------------------------------------
# Evidence Evaluator - details
# ----------------------------------------------------------------------

@dataclass
class EvidenceEvaluatorConfig:
    """details"""
    # details
    sufficient_top1_score: float = 0.65
    insufficient_top1_score: float = 0.40
    
    # details
    sufficient_hit_count: int = 3
    insufficient_hit_count: int = 1
    
    # details
    sufficient_coverage: float = 0.7
    insufficient_coverage: float = 0.3
    
    # details
    score_weight: float = 0.4
    hit_count_weight: float = 0.3
    coverage_weight: float = 0.3


# details Agent details
DETECTION_EVALUATOR_CONFIG = EvidenceEvaluatorConfig(
    sufficient_top1_score=0.60,
    insufficient_top1_score=0.35,
    sufficient_hit_count=2,
    insufficient_hit_count=1,
    sufficient_coverage=0.6,
    insufficient_coverage=0.25,
)

ASSESSMENT_EVALUATOR_CONFIG = EvidenceEvaluatorConfig(
    sufficient_top1_score=0.65,
    insufficient_top1_score=0.40,
    sufficient_hit_count=3,
    insufficient_hit_count=1,
    sufficient_coverage=0.7,
    insufficient_coverage=0.3,
)

CRITIC_EVALUATOR_CONFIG = EvidenceEvaluatorConfig(
    sufficient_top1_score=0.60,
    insufficient_top1_score=0.35,
    sufficient_hit_count=2,
    insufficient_hit_count=1,
    sufficient_coverage=0.5,
    insufficient_coverage=0.2,
)


class EvidenceEvaluator:
    """
    details。
    
    details RAG details。
    details MoRA-RAG details Evidence Evaluator Agent。
    """
    
    def __init__(self, config: Optional[EvidenceEvaluatorConfig] = None):
        self.config = config or EvidenceEvaluatorConfig()
    
    def evaluate(
        self,
        query: str,
        rag_result: RAGResult,
        required_aspects: Optional[List[str]] = None,
    ) -> EvidenceEvaluation:
        """
        details。
        
        Args:
            query: details
            rag_result: RAG details
            required_aspects: details（details）
        
        Returns:
            EvidenceEvaluation: details
        """
        cfg = self.config
        
        # 1. details
        top1_score = rag_result.top1_score
        if top1_score >= cfg.sufficient_top1_score:
            score_rating = 1.0
        elif top1_score <= cfg.insufficient_top1_score:
            score_rating = 0.0
        else:
            # details
            score_rating = (top1_score - cfg.insufficient_top1_score) / (
                cfg.sufficient_top1_score - cfg.insufficient_top1_score
            )
        
        # 2. details
        hit_count = rag_result.hit_count
        if hit_count >= cfg.sufficient_hit_count:
            hit_rating = 1.0
        elif hit_count <= cfg.insufficient_hit_count:
            hit_rating = 0.0
        else:
            hit_rating = (hit_count - cfg.insufficient_hit_count) / (
                cfg.sufficient_hit_count - cfg.insufficient_hit_count
            )
        
        # 3. details
        coverage = self._compute_coverage(query, rag_result.chunks, required_aspects)
        if coverage >= cfg.sufficient_coverage:
            coverage_rating = 1.0
        elif coverage <= cfg.insufficient_coverage:
            coverage_rating = 0.0
        else:
            coverage_rating = (coverage - cfg.insufficient_coverage) / (
                cfg.sufficient_coverage - cfg.insufficient_coverage
            )
        
        # 4. details
        overall_score = (
            cfg.score_weight * score_rating
            + cfg.hit_count_weight * hit_rating
            + cfg.coverage_weight * coverage_rating
        )
        
        # 5. details
        is_sufficient = overall_score >= 0.6  # details
        
        # 6. details
        missing_aspects = self._find_missing_aspects(query, rag_result.chunks, required_aspects)
        
        # 7. details
        reason = self._generate_reason(
            is_sufficient=is_sufficient,
            top1_score=top1_score,
            hit_count=hit_count,
            coverage=coverage,
            overall_score=overall_score,
            missing_aspects=missing_aspects,
        )
        
        return EvidenceEvaluation(
            is_sufficient=is_sufficient,
            score=overall_score,
            missing_aspects=missing_aspects,
            reason=reason,
        )
    
    def _compute_coverage(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        required_aspects: Optional[List[str]] = None,
    ) -> float:
        """details"""
        if not chunks:
            return 0.0
        
        if not required_aspects:
            # details：details chunk details
            total_text_len = sum(len(str(c.get("text", ""))) for c in chunks)
            # details 500 details
            return min(1.0, total_text_len / 500)
        
        # details
        covered = 0
        all_text = " ".join(str(c.get("text", "")) for c in chunks).lower()
        for aspect in required_aspects:
            if aspect.lower() in all_text:
                covered += 1
        
        return covered / len(required_aspects) if required_aspects else 1.0
    
    def _find_missing_aspects(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        required_aspects: Optional[List[str]] = None,
    ) -> List[str]:
        """details"""
        if not required_aspects:
            return []
        
        all_text = " ".join(str(c.get("text", "")) for c in chunks).lower()
        missing = []
        for aspect in required_aspects:
            if aspect.lower() not in all_text:
                missing.append(aspect)
        return missing
    
    def _generate_reason(
        self,
        is_sufficient: bool,
        top1_score: float,
        hit_count: int,
        coverage: float,
        overall_score: float,
        missing_aspects: List[str],
    ) -> str:
        """details"""
        if is_sufficient:
            return f"Evidence sufficient (score={overall_score:.2f}, top1={top1_score:.2f}, hits={hit_count})"
        
        reasons = []
        cfg = self.config
        
        if top1_score < cfg.insufficient_top1_score:
            reasons.append(f"low relevance score ({top1_score:.2f})")
        if hit_count < cfg.insufficient_hit_count:
            reasons.append(f"too few hits ({hit_count})")
        if coverage < cfg.insufficient_coverage:
            reasons.append(f"low coverage ({coverage:.2f})")
        if missing_aspects:
            reasons.append(f"missing aspects: {missing_aspects}")
        
        return f"Evidence insufficient: {'; '.join(reasons)}"


# ----------------------------------------------------------------------
# Query Rewriter - details
# ----------------------------------------------------------------------

class QueryRewriter:
    """
    details。
    
    details，details。
    details MoRA-RAG details Reflection & Question Rewriter Agent。
    """
    
    def __init__(self, llm_func: Optional[Callable[[str, str], str]] = None):
        """
        Args:
            llm_func: LLM details，details (system_prompt, user_prompt) -> response
                      details None，details
        """
        self.llm_func = llm_func
    
    def rewrite(
        self,
        original_query: str,
        rag_result: RAGResult,
        evaluation: EvidenceEvaluation,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        details。
        
        Args:
            original_query: details
            rag_result: details RAG details
            evaluation: details
            context: details（details、details）
        
        Returns:
            details
        """
        if self.llm_func is not None:
            return self._rewrite_with_llm(original_query, rag_result, evaluation, context)
        else:
            return self._rewrite_heuristic(original_query, rag_result, evaluation, context)
    
    def _rewrite_heuristic(
        self,
        original_query: str,
        rag_result: RAGResult,
        evaluation: EvidenceEvaluation,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """details"""
        new_query = original_query
        
        # 1. details，details
        if evaluation.missing_aspects:
            new_query = f"{original_query} {' '.join(evaluation.missing_aspects)}"
        
        # 2. details，details
        if context:
            hazard_type = context.get("hazard_type", "")
            component = context.get("component", "")
            if hazard_type and hazard_type.lower() not in new_query.lower():
                new_query = f"{hazard_type} {new_query}"
            if component and component.lower() not in new_query.lower():
                new_query = f"{new_query} {component}"
        
        # 3. details
        words = new_query.split()
        if len(words) > 20:
            new_query = " ".join(words[:20])
        
        return new_query.strip()
    
    def _rewrite_with_llm(
        self,
        original_query: str,
        rag_result: RAGResult,
        evaluation: EvidenceEvaluation,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """details LLM details"""
        system_prompt = (
            "You are a query rewriter for a disaster damage assessment RAG system. "
            "Your task is to rewrite the query to improve retrieval results."
        )
        
        user_prompt = f"""
Original query: {original_query}

Retrieved {rag_result.hit_count} chunks with top score {rag_result.top1_score:.2f}.

Evaluation: {evaluation.reason}

Missing aspects: {evaluation.missing_aspects}

Context: {json.dumps(context or {}, ensure_ascii=False)}

Please rewrite the query to:
1. Be more specific about the missing aspects
2. Use different keywords that might match better
3. Keep it concise (under 50 words)

Output only the rewritten query, nothing else.
"""
        
        try:
            rewritten = self.llm_func(system_prompt, user_prompt)
            return rewritten.strip() if rewritten else original_query
        except Exception:
            return self._rewrite_heuristic(original_query, rag_result, evaluation, context)


# ----------------------------------------------------------------------
# RAG-WS details
# ----------------------------------------------------------------------

@dataclass
class RAGWSConfig:
    """RAG-WS details"""
    max_rag_retries: int = 2  # details RAG details
    enable_web_search: bool = True  # details
    web_search_max_results: int = 5  # details
    web_search_type: str = "hybrid"  # text | news | images | hybrid
    web_search_parse_pdf: bool = True  # details PDF details
    enable_global_rerank: bool = True  # details
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_weight: float = 0.7
    rerank_top_k: int = 10
    
    # details
    web_search_min_score: float = 0.3  # details
    web_search_min_attempts: int = 2  # details N details RAG details


class RAGWS:
    """
    RAG-WS (RAG with Web Search) details。
    
    details：
    1. details RAG details
    2. Evidence Evaluator details
    3. details，Query Rewriter details，details RAG
    4. details，details Fallback
    """
    
    def __init__(
        self,
        rag_func: Callable[[str, int], RAGResult],
        evaluator: Optional[EvidenceEvaluator] = None,
        rewriter: Optional[QueryRewriter] = None,
        web_searcher: Optional[DDGSSearchTool] = None,
        config: Optional[RAGWSConfig] = None,
    ):
        """
        Args:
            rag_func: RAG details，details (query, top_k) -> RAGResult
            evaluator: details
            rewriter: details
            web_searcher: details
            config: details
        """
        self.rag_func = rag_func
        self.evaluator = evaluator or EvidenceEvaluator()
        self.rewriter = rewriter or QueryRewriter()
        self.web_searcher = web_searcher or DDGSSearchTool()
        self.config = config or RAGWSConfig()
        self._cross_encoder = None

    def _get_cross_encoder(self):
        if self._cross_encoder is not None:
            return self._cross_encoder
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._cross_encoder = CrossEncoder(self.config.rerank_model)
            return self._cross_encoder
        except Exception:
            self._cross_encoder = None
            return None

    @staticmethod
    def _extract_chunk_score(chunk: Dict[str, Any]) -> float:
        if not isinstance(chunk, dict):
            return 0.0
        meta = chunk.get("meta", {}) if isinstance(chunk.get("meta"), dict) else {}
        for key in ("score_hint", "hybrid_score", "rerank_score", "dense_score", "bm25_score"):
            val = meta.get(key)
            if isinstance(val, (int, float)):
                return float(val)
        val = chunk.get("score")
        if isinstance(val, (int, float)):
            return float(val)
        return 0.0

    @staticmethod
    def _normalize_scores(scores: Dict[int, float]) -> Dict[int, float]:
        if not scores:
            return {}
        vals = list(scores.values())
        lo = min(vals)
        hi = max(vals)
        if hi - lo < 1e-9:
            return {k: 1.0 for k in scores}
        return {k: (float(v) - lo) / (hi - lo) for k, v in scores.items()}

    def _rerank_evidence_pool(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        *,
        source: str,
        top_k: int,
    ) -> RAGResult:
        if not chunks:
            return RAGResult(chunks=[], scores=[], query=query, source=source)

        dedup: List[Dict[str, Any]] = []
        seen = set()
        for chunk in chunks:
            text = str(chunk.get("text", "")).strip()
            if not text or text in seen:
                continue
            dedup.append(dict(chunk))
            seen.add(text)

        if not dedup:
            return RAGResult(chunks=[], scores=[], query=query, source=source)

        base_scores = {idx: self._extract_chunk_score(chunk) for idx, chunk in enumerate(dedup)}
        base_norm = self._normalize_scores(base_scores)
        ce_scores: Dict[int, float] = {}

        if self.config.enable_global_rerank:
            cross_encoder = self._get_cross_encoder()
            if cross_encoder is not None:
                try:
                    pairs = [[query, str(chunk.get("text", ""))] for chunk in dedup]
                    preds = cross_encoder.predict(pairs)
                    ce_scores = {idx: float(score) for idx, score in enumerate(preds)}
                except Exception:
                    ce_scores = {}

        ce_norm = self._normalize_scores(ce_scores)
        final_scores: Dict[int, float] = {}
        for idx in range(len(dedup)):
            if ce_norm:
                final_scores[idx] = (
                    self.config.rerank_weight * ce_norm.get(idx, 0.0)
                    + (1.0 - self.config.rerank_weight) * base_norm.get(idx, 0.0)
                )
            else:
                final_scores[idx] = base_norm.get(idx, 0.0)

        ranked_indices = sorted(final_scores.keys(), key=lambda i: final_scores[i], reverse=True)[: int(top_k)]
        ranked_chunks: List[Dict[str, Any]] = []
        ranked_scores: List[float] = []
        for idx in ranked_indices:
            chunk = dict(dedup[idx])
            meta = dict(chunk.get("meta", {}) or {})
            meta.update(
                {
                    "global_base_score": base_scores.get(idx, 0.0),
                    "global_rerank_score": ce_scores.get(idx),
                    "global_final_score": final_scores.get(idx, 0.0),
                }
            )
            chunk["meta"] = meta
            chunk["score"] = float(final_scores.get(idx, 0.0))
            ranked_chunks.append(chunk)
            ranked_scores.append(float(final_scores.get(idx, 0.0)))

        return RAGResult(
            chunks=ranked_chunks,
            scores=ranked_scores,
            query=query,
            source=source,
        )
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        context: Optional[Dict[str, Any]] = None,
        required_aspects: Optional[List[str]] = None,
    ) -> RAGWSResult:
        """
        details RAG-WS details。
        
        Args:
            query: details
            top_k: details chunk details
            context: details（details、details）
            required_aspects: details
        
        Returns:
            RAGWSResult: details
        """
        cfg = self.config
        current_query = query
        query_history = [query]
        all_chunks: List[Dict[str, Any]] = []
        attempts = 0
        
        # 1. details RAG details
        for attempt in range(cfg.max_rag_retries + 1):
            attempts = attempt + 1
            
            # details RAG details
            rag_result = self.rag_func(current_query, top_k)
            
            # details（details）
            existing_texts = {str(c.get("text", "")) for c in all_chunks}
            for chunk in rag_result.chunks:
                text = str(chunk.get("text", ""))
                if text and text not in existing_texts:
                    all_chunks.append(chunk)
                    existing_texts.add(text)
            
            reranked_internal = self._rerank_evidence_pool(
                query=query,
                chunks=all_chunks,
                source="internal",
                top_k=max(int(top_k), int(cfg.rerank_top_k)),
            )

            # details
            evaluation = self.evaluator.evaluate(
                query=query,  # details query details
                rag_result=reranked_internal,
                required_aspects=required_aspects,
            )
            
            if evaluation.is_sufficient:
                return RAGWSResult(
                    chunks=reranked_internal.chunks[: int(top_k)],
                    source="internal",
                    is_sufficient=True,
                    evaluation=evaluation,
                    attempts=attempts,
                    web_search_triggered=False,
                    query_history=query_history,
                )
            
            # details，details
            if attempt < cfg.max_rag_retries:
                current_query = self.rewriter.rewrite(
                    original_query=query,
                    rag_result=rag_result,
                    evaluation=evaluation,
                    context=context,
                )
                if current_query != query_history[-1]:
                    query_history.append(current_query)
        
        # 2. RAG details，details
        if (
            cfg.enable_web_search
            and attempts >= cfg.web_search_min_attempts
            and evaluation.score < cfg.web_search_min_score
        ):
            web_results = self._do_web_search(query, context)
            
            if web_results:
                # details chunk details
                total_web = max(len(web_results), 1)
                for idx, wr in enumerate(web_results):
                    rank_score = 1.0 - (idx / total_web)
                    chunk = {
                        "text": f"{wr.get('title', '')}\n{wr.get('snippet', '')}",
                        "source": wr.get("url", "web"),
                        "meta": {
                            "source_type": "web_search",
                            "title": wr.get("title", ""),
                            "url": wr.get("url", ""),
                            "score_hint": float(rank_score),
                        },
                    }
                    all_chunks.append(chunk)
                
                reranked_combined = self._rerank_evidence_pool(
                    query=query,
                    chunks=all_chunks,
                    source="internal+web",
                    top_k=max(int(top_k), int(cfg.rerank_top_k)),
                )
                evaluation = self.evaluator.evaluate(
                    query=query,
                    rag_result=reranked_combined,
                    required_aspects=required_aspects,
                )
                
                return RAGWSResult(
                    chunks=reranked_combined.chunks[: int(top_k)],
                    source="internal+web",
                    is_sufficient=evaluation.is_sufficient,
                    evaluation=evaluation,
                    attempts=attempts,
                    web_search_triggered=True,
                    web_results=web_results,
                    query_history=query_history,
                )
        
        # 3. details（details）
        final_internal = self._rerank_evidence_pool(
            query=query,
            chunks=all_chunks,
            source="internal",
            top_k=max(int(top_k), int(cfg.rerank_top_k)),
        )
        return RAGWSResult(
            chunks=final_internal.chunks[: int(top_k)],
            source="internal",
            is_sufficient=evaluation.is_sufficient,
            evaluation=evaluation,
            attempts=attempts,
            web_search_triggered=False,
            query_history=query_history,
        )
    
    def _do_web_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """details"""
        # details
        search_query = query
        if context:
            hazard_type = context.get("hazard_type", "")
            location = context.get("location", "")
            if hazard_type:
                search_query = f"{hazard_type} {search_query}"
            if location:
                search_query = f"{search_query} {location}"
        
        # details
        result = self.web_searcher.execute(
            query=search_query,
            search_type=self.config.web_search_type,
            max_results=self.config.web_search_max_results,
            parse_pdf=self.config.web_search_parse_pdf,
        )
        
        if result.success and result.data:
            return result.data.get("results", [])
        return []


# ----------------------------------------------------------------------
# details - details Agent details RAG-WS details
# ----------------------------------------------------------------------

def create_detection_ragws(
    rag_func: Callable[[str, int], RAGResult],
    llm_func: Optional[Callable[[str, str], str]] = None,
    enable_web_search: bool = True,
) -> RAGWS:
    """details DetectionAgent details RAG-WS details"""
    return RAGWS(
        rag_func=rag_func,
        evaluator=EvidenceEvaluator(DETECTION_EVALUATOR_CONFIG),
        rewriter=QueryRewriter(llm_func),
        config=RAGWSConfig(
            max_rag_retries=2,
            enable_web_search=enable_web_search,
            web_search_max_results=5,
            web_search_min_score=0.35,
            web_search_min_attempts=2,
        ),
    )


def create_assessment_ragws(
    rag_func: Callable[[str, int], RAGResult],
    llm_func: Optional[Callable[[str, str], str]] = None,
    enable_web_search: bool = True,
) -> RAGWS:
    """details AssessmentAgent details RAG-WS details"""
    return RAGWS(
        rag_func=rag_func,
        evaluator=EvidenceEvaluator(ASSESSMENT_EVALUATOR_CONFIG),
        rewriter=QueryRewriter(llm_func),
        config=RAGWSConfig(
            max_rag_retries=2,
            enable_web_search=enable_web_search,
            web_search_max_results=5,
            web_search_min_score=0.40,
            web_search_min_attempts=2,
        ),
    )


def create_critic_ragws(
    rag_func: Callable[[str, int], RAGResult],
    llm_func: Optional[Callable[[str, str], str]] = None,
    enable_web_search: bool = True,
) -> RAGWS:
    """details CriticAgent details RAG-WS details"""
    return RAGWS(
        rag_func=rag_func,
        evaluator=EvidenceEvaluator(CRITIC_EVALUATOR_CONFIG),
        rewriter=QueryRewriter(llm_func),
        config=RAGWSConfig(
            max_rag_retries=1,  # Critic details
            enable_web_search=enable_web_search,
            web_search_max_results=3,
            web_search_min_score=0.35,
            web_search_min_attempts=1,
        ),
    )

