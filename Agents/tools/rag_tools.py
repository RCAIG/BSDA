#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/rag_tools.py

details RAG details，details RAG details Tool details。
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .registry import Tool, ToolResult

# details RAG details
_AGENTS_ROOT = Path(__file__).resolve().parent.parent
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


@dataclass
class InternalRAGTool(Tool):
    """
    details RAG details。
    details DetectionAgent details DamageFeatureRAG。
    """
    
    name: str = "search_internal_rag"
    description: str = (
        "Search the internal knowledge base for disaster damage patterns, rules, and guidelines. "
        "Use this to find official damage assessment criteria, FEMA guidelines, "
        "and historical damage patterns. This is the primary source for damage classification rules."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query describing the damage pattern or component to look up.",
            },
            "hazard_type": {
                "type": "string",
                "description": "Type of hazard (e.g., 'hurricane', 'earthquake', 'tornado', 'flood').",
                "default": "hurricane",
            },
            "component": {
                "type": "string",
                "description": "Building component affected (e.g., 'roof', 'facade', 'foundation').",
                "default": "",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    })
    
    # RAG details（details）
    _rag: Any = field(default=None, repr=False)
    _rag_initialized: bool = field(default=False, repr=False)
    
    # details
    artifacts_dir: str = ""
    corpus: str = "gov"
    embed_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    def _init_rag(self) -> bool:
        """details RAG"""
        if self._rag_initialized:
            return self._rag is not None
        
        self._rag_initialized = True
        
        try:
            # details DetectionAgent details RAG
            det_pkg_root = _AGENTS_ROOT / "DetectionAgent"
            if str(det_pkg_root) not in sys.path:
                sys.path.insert(0, str(det_pkg_root))
            
            from rag import DamageFeatureRAG
            
            # details artifacts details
            if not self.artifacts_dir:
                # details config details
                try:
                    import config as det_cfg
                    self.artifacts_dir = str(getattr(det_cfg, "RAG_ARTIFACTS_DIR", ""))
                except ImportError:
                    pass
            
            if not self.artifacts_dir:
                # details
                repo_root = _AGENTS_ROOT.parent
                default_artifacts = repo_root / "RAG" / "artifacts"
                if default_artifacts.exists():
                    self.artifacts_dir = str(default_artifacts)
            
            if not self.artifacts_dir or not Path(self.artifacts_dir).exists():
                return False
            
            self._rag = DamageFeatureRAG(
                artifacts_dir=self.artifacts_dir,
                corpus=self.corpus,
                embed_model=self.embed_model,
            )
            return True
        
        except Exception as e:
            print(f"[InternalRAGTool] Failed to initialize RAG: {e}")
            return False
    
    def execute(
        self,
        query: str,
        hazard_type: str = "hurricane",
        component: str = "",
        top_k: int = 5,
        **kwargs,
    ) -> ToolResult:
        """details RAG details"""
        
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Query cannot be empty",
                tool_name=self.name,
            )
        
        if not self._init_rag():
            return ToolResult(
                success=False,
                error="Internal RAG not available. Check artifacts directory.",
                tool_name=self.name,
            )
        
        query = query.strip()
        top_k = max(1, min(10, int(top_k)))
        
        try:
            # details search_for_detection_change details
            hits = self._rag.search_for_detection_change(
                hazard_type=_safe_str(hazard_type) or "hurricane",
                component=_safe_str(component),
                change_desc=query,
                pre_desc="",
                post_desc="",
                extra_context="",
                language="en",
                top_k=top_k,
            )
            
            if not hits:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "hazard_type": hazard_type,
                        "component": component,
                        "results": [],
                        "message": "No relevant rules or patterns found in internal knowledge base.",
                    },
                    tool_name=self.name,
                )
            
            # details
            results = []
            for hit in hits:
                if isinstance(hit, dict):
                    results.append({
                        "text": _safe_str(hit.get("text", "")),
                        "score": float(hit.get("score", 0.0)) if hit.get("score") else None,
                        "source": _safe_str(hit.get("source", "") or hit.get("doc_id", "")),
                        "meta": hit.get("meta", {}),
                    })
                elif isinstance(hit, str):
                    results.append({"text": hit, "score": None, "source": "", "meta": {}})
            
            # details
            top1_score = results[0].get("score") if results and results[0].get("score") else 0.0
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "hazard_type": hazard_type,
                    "component": component,
                    "result_count": len(results),
                    "top1_score": top1_score,
                    "results": results,
                },
                tool_name=self.name,
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"RAG search failed: {str(e)}",
                tool_name=self.name,
            )


@dataclass
class GovRulesRAGTool(Tool):
    """
    details/details RAG details。
    details AssessmentAgent details。
    """
    
    name: str = "search_gov_rules"
    description: str = (
        "Search official government rules and damage grading standards. "
        "Use this to find FEMA guidelines, building codes, and official damage level definitions "
        "(minor/moderate/severe). Best for determining how to classify damage severity."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query describing the damage type and severity to look up.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    })
    
    def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """details"""
        
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Query cannot be empty",
                tool_name=self.name,
            )
        
        try:
            # details AssessmentAgent details RAG
            from AssessmentAgent.rag import search_gov_rules
            
            results = search_gov_rules(query.strip(), top_k=int(top_k))
            
            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": [],
                        "message": "No government rules found for this query.",
                    },
                    tool_name=self.name,
                )
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                },
                tool_name=self.name,
            )
        
        except ImportError:
            return ToolResult(
                success=False,
                error="AssessmentAgent RAG module not available.",
                tool_name=self.name,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Gov rules search failed: {str(e)}",
                tool_name=self.name,
            )


@dataclass
class HistoryCasesRAGTool(Tool):
    """
    details RAG details。
    details。
    """
    
    name: str = "search_history_cases"
    description: str = (
        "Search historical damage cases with known labels. "
        "Use this to find similar past cases and see how they were classified. "
        "Helps calibrate damage level predictions based on precedent."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Description of the damage pattern to find similar cases for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    })
    
    def execute(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """details"""
        
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Query cannot be empty",
                tool_name=self.name,
            )
        
        try:
            from AssessmentAgent.rag import search_history_cases
            
            results = search_history_cases(query.strip(), top_k=int(top_k))
            
            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "results": [],
                        "message": "No similar historical cases found.",
                    },
                    tool_name=self.name,
                )
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "result_count": len(results),
                    "results": results,
                },
                tool_name=self.name,
            )
        
        except ImportError:
            return ToolResult(
                success=False,
                error="AssessmentAgent RAG module not available.",
                tool_name=self.name,
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"History cases search failed: {str(e)}",
                tool_name=self.name,
            )


@dataclass
class EventContextTool(Tool):
    """
    details。
    details。
    """
    
    name: str = "lookup_event_context"
    description: str = (
        "Look up disaster event context for a specific location and time. "
        "Use this to check if a disaster actually occurred at the given location "
        "around the given date. Combines internal records with external search if needed."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "lat": {
                "type": "number",
                "description": "Latitude of the location.",
            },
            "lon": {
                "type": "number",
                "description": "Longitude of the location.",
            },
            "pre_date": {
                "type": "string",
                "description": "Pre-disaster image date (YYYY-MM-DD or similar).",
            },
            "post_date": {
                "type": "string",
                "description": "Post-disaster image date (YYYY-MM-DD or similar).",
            },
            "hazard_type": {
                "type": "string",
                "description": "Expected hazard type to check for.",
                "default": "hurricane",
            },
        },
        "required": ["lat", "lon"],
    })
    
    def execute(
        self,
        lat: float,
        lon: float,
        pre_date: str = "",
        post_date: str = "",
        hazard_type: str = "hurricane",
        **kwargs,
    ) -> ToolResult:
        """details"""
        
        try:
            lat = float(lat)
            lon = float(lon)
        except (TypeError, ValueError):
            return ToolResult(
                success=False,
                error="Invalid lat/lon values",
                tool_name=self.name,
            )
        
        # details
        location_desc = f"coordinates ({lat:.4f}, {lon:.4f})"
        
        # details
        time_desc = ""
        if pre_date and post_date:
            time_desc = f"between {pre_date} and {post_date}"
        elif post_date:
            time_desc = f"around {post_date}"
        elif pre_date:
            time_desc = f"after {pre_date}"
        
        # details，details NOAA/FEMA API
        # details DDGS details
        context = {
            "location": {"lat": lat, "lon": lon, "description": location_desc},
            "time_range": {"pre_date": pre_date, "post_date": post_date, "description": time_desc},
            "hazard_type": hazard_type,
            "event_records": [],
            "note": "Event context lookup is a placeholder. Consider using web_search or search_disaster_event for actual event verification.",
        }
        
        return ToolResult(
            success=True,
            data=context,
            tool_name=self.name,
        )
