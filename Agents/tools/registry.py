#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/registry.py

details：details，details ReAct / Tool-calling details。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union


@dataclass
class ToolResult:
    """details"""
    success: bool
    data: Any = None
    error: Optional[str] = None
    tool_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "tool_name": self.tool_name,
        }
    
    def to_observation_str(self, max_chars: int = 2000) -> str:
        """details ReAct details Observation details"""
        if not self.success:
            return f"[Tool Error] {self.error or 'Unknown error'}"
        
        if isinstance(self.data, str):
            text = self.data
        elif isinstance(self.data, (dict, list)):
            text = json.dumps(self.data, ensure_ascii=False, indent=2)
        else:
            text = str(self.data)
        
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n... [truncated, total {len(text)} chars]"
        
        return text


@dataclass
class Tool(ABC):
    """details"""
    name: str
    description: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """details"""
        pass
    
    def get_schema(self) -> Dict[str, Any]:
        """details JSON Schema（details prompt details function calling）"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
    
    def get_prompt_description(self) -> str:
        """details ReAct prompt details"""
        params_desc = ""
        if self.parameters.get("properties"):
            params_list = []
            for k, v in self.parameters["properties"].items():
                required = k in self.parameters.get("required", [])
                req_mark = " (required)" if required else " (optional)"
                params_list.append(f"  - {k}: {v.get('description', v.get('type', 'any'))}{req_mark}")
            params_desc = "\n" + "\n".join(params_list)
        
        return f"- {self.name}: {self.description}{params_desc}"


class ToolRegistry:
    """details"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """details"""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """details"""
        return self._tools.get(name)
    
    def execute(self, name: str, **kwargs) -> ToolResult:
        """details"""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                success=False,
                error=f"Tool '{name}' not found",
                tool_name=name,
            )
        try:
            result = tool.execute(**kwargs)
            result.tool_name = name
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                error=str(e),
                tool_name=name,
            )
    
    def list_tools(self) -> List[str]:
        """details"""
        return list(self._tools.keys())
    
    def get_all_schemas(self) -> List[Dict[str, Any]]:
        """details schema"""
        return [t.get_schema() for t in self._tools.values()]
    
    def get_tools_prompt(self) -> str:
        """details ReAct prompt details"""
        if not self._tools:
            return "No tools available."
        
        lines = ["Available tools:"]
        for tool in self._tools.values():
            lines.append(tool.get_prompt_description())
        
        return "\n".join(lines)


# details
_DEFAULT_REGISTRY: Optional[ToolRegistry] = None


def get_default_registry() -> ToolRegistry:
    """details"""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = ToolRegistry()
    return _DEFAULT_REGISTRY


def register_tool(tool: Tool) -> None:
    """details"""
    get_default_registry().register(tool)
