#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/ddgs_search.py

DDGS details。
details deedy5/ddgs details。
GitHub: http<LOCAL_PATH>

details：pip install ddgs
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .registry import Tool, ToolResult


@dataclass
class DDGSSearchTool(Tool):
    """
    DDGS details (deedy5/ddgs)。
    
    details：
    - text: details
    - news: details
    - images: details
    
    GitHub: http<LOCAL_PATH>
    """
    
    name: str = "web_search"
    description: str = (
        "Search the web using DDGS metasearch. "
        "Use this when you need external information about disaster events, "
        "official guidelines, or real-world context that is not in the internal knowledge base. "
        "Returns a list of search results with titles, snippets, and URLs."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and include relevant keywords like location, date, disaster type.",
            },
            "search_type": {
                "type": "string",
                "enum": ["text", "news", "images", "hybrid"],
                "description": "Type of search: text/news/images/hybrid (text+news+images).",
                "default": "text",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1-10).",
                "default": 5,
            },
            "region": {
                "type": "string",
                "description": "Region for search results (e.g., 'us-en', 'wt-wt' for worldwide).",
                "default": "wt-wt",
            },
            "parse_pdf": {
                "type": "boolean",
                "description": "If true, detect PDF links from text/news results and parse PDF text snippets.",
                "default": False,
            },
        },
        "required": ["query"],
    })
    
    # details
    timeout: int = 10
    safesearch: str = "moderate"
    max_pdf_pages: int = 3
    max_pdf_chars: int = 4000
    
    def execute(
        self,
        query: str,
        search_type: str = "text",
        max_results: int = 5,
        region: str = "wt-wt",
        parse_pdf: bool = False,
        **kwargs,
    ) -> ToolResult:
        """details"""
        
        if not query or not query.strip():
            return ToolResult(
                success=False,
                error="Query cannot be empty",
                tool_name=self.name,
            )
        
        query = query.strip()
        max_results = max(1, min(10, int(max_results)))
        search_type = search_type.lower() if search_type else "text"
        parse_pdf = bool(parse_pdf)
        
        try:
            from ddgs import DDGS
        except ImportError:
            return ToolResult(
                success=False,
                error="ddgs not installed. Run: pip install ddgs",
                tool_name=self.name,
            )
        
        try:
            results: List[Dict[str, Any]] = []
            image_results: List[Dict[str, Any]] = []
            pdf_results: List[Dict[str, Any]] = []

            # details deedy5/ddgs API
            ddgs = DDGS()

            def _append_text_results(raw_iter, *, source_type: str) -> None:
                for item in raw_iter:
                    if source_type == "news":
                        results.append(
                            {
                                "source_type": "news",
                                "title": item.get("title", ""),
                                "snippet": item.get("body", ""),
                                "url": item.get("url", ""),
                                "source": item.get("source", ""),
                                "date": item.get("date", ""),
                            }
                        )
                    else:
                        results.append(
                            {
                                "source_type": "web",
                                "title": item.get("title", ""),
                                "snippet": item.get("body", ""),
                                "url": item.get("href", ""),
                            }
                        )

            def _append_image_results(raw_iter) -> None:
                for item in raw_iter:
                    img = {
                        "source_type": "image",
                        "title": item.get("title", ""),
                        "snippet": item.get("body", "") or item.get("title", ""),
                        "url": item.get("url", "") or item.get("image", ""),
                        "image_url": item.get("image", ""),
                        "thumbnail": item.get("thumbnail", ""),
                        "width": item.get("width"),
                        "height": item.get("height"),
                        "source_page": item.get("source", ""),
                    }
                    image_results.append(img)
                    # details：details results
                    results.append(img)

            if search_type == "news":
                _append_text_results(
                    ddgs.news(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=max_results,
                    ),
                    source_type="news",
                )
            elif search_type == "images":
                _append_image_results(
                    ddgs.images(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=max_results,
                    )
                )
            elif search_type == "hybrid":
                text_n = max(1, max_results // 2)
                news_n = max(1, max_results // 3)
                image_n = max(1, max_results - text_n - news_n)
                _append_text_results(
                    ddgs.text(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=text_n,
                    ),
                    source_type="web",
                )
                _append_text_results(
                    ddgs.news(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=news_n,
                    ),
                    source_type="news",
                )
                _append_image_results(
                    ddgs.images(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=image_n,
                    )
                )
            else:
                _append_text_results(
                    ddgs.text(
                        query=query,
                        region=region,
                        safesearch=self.safesearch,
                        max_results=max_results,
                    ),
                    source_type="web",
                )

            if parse_pdf and results:
                pdf_candidates = self._collect_pdf_candidates(results)
                for pdf_url in pdf_candidates:
                    parsed = self._parse_pdf_url(pdf_url)
                    pdf_results.append(parsed)
                    if parsed.get("success"):
                        results.append(
                            {
                                "source_type": "pdf",
                                "title": parsed.get("title", "") or "PDF document",
                                "snippet": parsed.get("text_preview", ""),
                                "url": pdf_url,
                                "pdf_parsed": True,
                                "pdf_pages_parsed": parsed.get("pages_parsed", 0),
                            }
                        )
            
            if not results:
                return ToolResult(
                    success=True,
                    data={
                        "query": query,
                        "search_type": search_type,
                        "results": [],
                        "message": "No results found for this query.",
                    },
                    tool_name=self.name,
                )
            
            return ToolResult(
                success=True,
                data={
                    "query": query,
                    "search_type": search_type,
                    "result_count": len(results),
                    "results": results,
                    "image_results": image_results,
                    "pdf_results": pdf_results,
                },
                tool_name=self.name,
            )
        
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Search failed: {str(e)}",
                tool_name=self.name,
            )

    def _collect_pdf_candidates(self, results: List[Dict[str, Any]]) -> List[str]:
        cands: List[str] = []
        for r in results:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url", "")).strip()
            if not url:
                continue
            title = str(r.get("title", "")).lower()
            snippet = str(r.get("snippet", "")).lower()
            if url.lower().endswith(".pdf") or " pdf" in title or title.startswith("pdf") or ".pdf" in snippet:
                cands.append(url)
        # details，details
        seen = set()
        out: List[str] = []
        for u in cands:
            if u not in seen:
                out.append(u)
                seen.add(u)
        return out[:3]

    def _parse_pdf_url(self, pdf_url: str) -> Dict[str, Any]:
        try:
            import requests
        except Exception:
            return {
                "url": pdf_url,
                "success": False,
                "error": "requests_not_installed",
            }
        try:
            resp = requests.get(pdf_url, timeout=self.timeout)
            resp.raise_for_status()
            content = resp.content
        except Exception as e:
            return {
                "url": pdf_url,
                "success": False,
                "error": f"download_failed: {str(e)}",
            }
        try:
            from pypdf import PdfReader  # type: ignore
        except Exception:
            return {
                "url": pdf_url,
                "success": False,
                "error": "pypdf_not_installed",
            }
        try:
            reader = PdfReader(io.BytesIO(content))
            texts: List[str] = []
            max_pages = min(len(reader.pages), int(self.max_pdf_pages))
            for i in range(max_pages):
                page_text = reader.pages[i].extract_text() or ""
                if page_text:
                    texts.append(page_text.strip())
            merged = "\n".join(texts).strip()
            merged = merged[: int(self.max_pdf_chars)]
            return {
                "url": pdf_url,
                "success": True,
                "title": "",
                "pages_parsed": max_pages,
                "text_preview": merged,
            }
        except Exception as e:
            return {
                "url": pdf_url,
                "success": False,
                "error": f"parse_failed: {str(e)}",
            }


@dataclass
class RealTimeDisasterSearchTool(Tool):
    """
    details。
    
    details：
    - details（details、details、details）
    - details
    - details
    - details/details
    
    details：FEMA/NOAA details RAG，details RAG details。
    """
    
    name: str = "search_realtime_disaster_info"
    description: str = (
        "Search for REAL-TIME information about an ongoing or recent disaster event. "
        "Use this when you need current/live information that is NOT in the internal knowledge base, such as: "
        "1) Current disaster intensity, path, or affected areas; "
        "2) Latest news reports from the disaster zone; "
        "3) Real-time damage reports from specific locations; "
        "4) Emergency announcements and updates. "
        "Do NOT use this for general disaster guidelines (use RAG instead)."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Free-form search query. Be specific about what real-time info you need. "
                    "Examples: 'recent hurricane current wind speed', "
                    "'coastal town damage photos today', "
                    "'tornado affected neighborhoods'"
                ),
            },
            "search_intent": {
                "type": "string",
                "enum": ["event_details", "local_damage", "affected_areas", "latest_news", "custom"],
                "description": (
                    "Intent of the search: "
                    "'event_details' - disaster intensity, path, timeline; "
                    "'local_damage' - damage reports at specific location; "
                    "'affected_areas' - which areas were impacted; "
                    "'latest_news' - most recent news coverage; "
                    "'custom' - use query as-is"
                ),
                "default": "custom",
            },
            "location": {
                "type": "string",
                "description": "Optional: specific location to focus on.",
            },
            "disaster_name": {
                "type": "string",
                "description": "Optional: name of the disaster event.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-10).",
                "default": 5,
            },
        },
        "required": ["query"],
    })
    
    _ddgs_tool: DDGSSearchTool = field(default_factory=DDGSSearchTool)
    
    def _build_query(
        self,
        query: str,
        search_intent: str,
        location: Optional[str],
        disaster_name: Optional[str],
    ) -> str:
        """details query"""
        
        if search_intent == "custom":
            # details query
            return query
        
        parts = []
        
        if disaster_name:
            parts.append(disaster_name)
        
        if location:
            parts.append(location)
        
        # details
        intent_keywords = {
            "event_details": "intensity wind speed magnitude path timeline",
            "local_damage": "damage destruction photos assessment",
            "affected_areas": "affected areas impact zone neighborhoods",
            "latest_news": "latest news update today",
        }
        
        if search_intent in intent_keywords:
            parts.append(intent_keywords[search_intent])
        
        # details query，details
        if query and query.strip():
            parts.append(query.strip())
        
        return " ".join(parts) if parts else query
    
    def execute(
        self,
        query: str,
        search_intent: str = "custom",
        location: str = "",
        disaster_name: str = "",
        max_results: int = 5,
        **kwargs,
    ) -> ToolResult:
        """details"""
        
        if not query and not location and not disaster_name:
            return ToolResult(
                success=False,
                error="At least one of query, location, or disaster_name is required",
                tool_name=self.name,
            )
        
        # details query
        final_query = self._build_query(
            query=query,
            search_intent=search_intent,
            location=location if location else None,
            disaster_name=disaster_name if disaster_name else None,
        )
        
        # details（details）
        news_count = max(3, (max_results + 1) // 2)
        text_count = max(2, max_results - news_count)
        
        news_result = self._ddgs_tool.execute(
            query=final_query,
            search_type="news",
            max_results=news_count,
        )
        
        text_result = self._ddgs_tool.execute(
            query=final_query,
            search_type="text",
            max_results=text_count,
        )
        
        # details，details
        combined_results = []
        
        if news_result.success and news_result.data:
            for r in news_result.data.get("results", []):
                r["source_type"] = "news"
                r["timeliness"] = "recent"
                combined_results.append(r)
        
        if text_result.success and text_result.data:
            for r in text_result.data.get("results", []):
                r["source_type"] = "web"
                r["timeliness"] = "unknown"
                combined_results.append(r)
        
        return ToolResult(
            success=True,
            data={
                "query": final_query,
                "search_intent": search_intent,
                "location": location,
                "disaster_name": disaster_name,
                "result_count": len(combined_results),
                "results": combined_results,
                "note": "Results are from web search. For official guidelines and standards, use internal RAG.",
            },
            tool_name=self.name,
        )


@dataclass  
class DisasterContextSearchTool(Tool):
    """
    details。
    
    details RAG details，details：
    - details
    - details/details
    - details
    """
    
    name: str = "search_disaster_context"
    description: str = (
        "Search for supplementary context when RAG results are insufficient. "
        "Use this to find: "
        "1) Typical damage patterns for specific building types in this disaster; "
        "2) Geographic or climate characteristics of the affected area; "
        "3) Historical damage cases from similar disasters. "
        "This helps when you need more context to make accurate damage assessments."
    )
    parameters: Dict[str, Any] = field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "context_type": {
                "type": "string",
                "enum": ["damage_pattern", "geographic_context", "historical_case"],
                "description": (
                    "Type of context needed: "
                    "'damage_pattern' - how this disaster type damages specific structures; "
                    "'geographic_context' - local terrain, climate, building codes; "
                    "'historical_case' - past similar events for reference"
                ),
            },
            "disaster_type": {
                "type": "string",
                "description": "Type of disaster (hurricane, tornado, earthquake, flood, wildfire).",
            },
            "subject": {
                "type": "string",
                "description": (
                    "Subject of interest. For damage_pattern: building component (roof, wall, window). "
                    "For geographic_context: location name. "
                    "For historical_case: similar event or location."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results (1-10).",
                "default": 5,
            },
        },
        "required": ["context_type", "disaster_type", "subject"],
    })
    
    _ddgs_tool: DDGSSearchTool = field(default_factory=DDGSSearchTool)
    
    def execute(
        self,
        context_type: str,
        disaster_type: str,
        subject: str,
        max_results: int = 5,
        **kwargs,
    ) -> ToolResult:
        """details"""
        
        if not all([context_type, disaster_type, subject]):
            return ToolResult(
                success=False,
                error="context_type, disaster_type, and subject are all required",
                tool_name=self.name,
            )
        
        # details query
        if context_type == "damage_pattern":
            query = f"{disaster_type} {subject} damage pattern failure mode"
        elif context_type == "geographic_context":
            query = f"{subject} geography terrain climate building codes {disaster_type} vulnerability"
        elif context_type == "historical_case":
            query = f"{disaster_type} {subject} historical damage case study"
        else:
            query = f"{disaster_type} {subject}"
        
        result = self._ddgs_tool.execute(
            query=query,
            search_type="text",
            max_results=max_results,
        )
        
        if not result.success:
            return result
        
        return ToolResult(
            success=True,
            data={
                "query": query,
                "context_type": context_type,
                "disaster_type": disaster_type,
                "subject": subject,
                "result_count": result.data.get("result_count", 0) if result.data else 0,
                "results": result.data.get("results", []) if result.data else [],
            },
            tool_name=self.name,
        )


# Backward-compatible alias for older imports.
DisasterEventSearchTool = RealTimeDisasterSearchTool
