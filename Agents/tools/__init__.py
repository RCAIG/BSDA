#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/__init__.py

details，details。
"""

from .registry import ToolRegistry, Tool, ToolResult, get_default_registry, register_tool
from .ddgs_search import DDGSSearchTool, DisasterEventSearchTool
from .rag_tools import InternalRAGTool, EventContextTool, GovRulesRAGTool, HistoryCasesRAGTool
from .react_loop import (
    ReActAgentLoop,
    ReActLoopConfig,
    ReActLoopResult,
    ReActStep,
    parse_react_output,
    create_detection_agent_tools,
    create_assessment_agent_tools,
    create_critic_agent_tools,
)
from .config import (
    DETECTION_RAG_THRESHOLDS,
    ASSESSMENT_RAG_THRESHOLDS,
    CRITIC_RAG_THRESHOLDS,
    DETECTION_REACT_CONFIG,
    ASSESSMENT_REACT_CONFIG,
    CRITIC_REACT_CONFIG,
)

__all__ = [
    # Registry
    "ToolRegistry",
    "Tool",
    "ToolResult",
    "get_default_registry",
    "register_tool",
    
    # Search tools
    "DDGSSearchTool",
    "DisasterEventSearchTool",
    
    # RAG tools
    "InternalRAGTool",
    "EventContextTool",
    "GovRulesRAGTool",
    "HistoryCasesRAGTool",
    
    # ReAct
    "ReActAgentLoop",
    "ReActLoopConfig",
    "ReActLoopResult",
    "ReActStep",
    "parse_react_output",
    "create_detection_agent_tools",
    "create_assessment_agent_tools",
    "create_critic_agent_tools",
    
    # Config
    "DETECTION_RAG_THRESHOLDS",
    "ASSESSMENT_RAG_THRESHOLDS",
    "CRITIC_RAG_THRESHOLDS",
    "DETECTION_REACT_CONFIG",
    "ASSESSMENT_REACT_CONFIG",
    "CRITIC_REACT_CONFIG",
]
