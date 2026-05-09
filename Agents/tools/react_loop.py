#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/react_loop.py

ReAct Agent Loop details。
details LLM details、details。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .registry import Tool, ToolRegistry, ToolResult

# details shared_llm
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


# ============================================================
# ReAct Prompt details
# ============================================================

REACT_SYSTEM_PROMPT_TEMPLATE = """You are an intelligent agent that can use tools to gather information before making decisions.

{tools_description}

## How to use tools

When you need to use a tool, output in this EXACT format:
```
Thought: [Your reasoning about what information you need]
Action: [tool_name]
Action Input: [JSON object with tool parameters]
```

After receiving the tool result (Observation), continue your reasoning.

When you have enough information to answer, output:
```
Thought: [Your final reasoning]
Final Answer: [Your final response in the required format]
```

## Important Rules

1. You can call multiple tools if needed, but call them ONE AT A TIME.
2. Always start with a Thought before taking an Action.
3. After each Observation, decide if you need more information or can give a Final Answer.
4. Maximum {max_iterations} tool calls allowed.
5. If internal knowledge base (RAG) results are insufficient, consider using web_search.
6. Always provide a Final Answer, even if you couldn't find all the information.

{additional_instructions}
"""

REACT_USER_PROMPT_TEMPLATE = """## Task
{task_description}

## Input
{input_data}

Now begin. Start with a Thought about what information you need.
"""


# ============================================================
# ReAct details
# ============================================================

@dataclass
class ReActStep:
    """ReAct details"""
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    final_answer: str = ""
    is_final: bool = False
    raw_output: str = ""


def parse_react_output(text: str) -> ReActStep:
    """
    details LLM details ReAct details。
    
    details：
    - Thought: ...
    - Action: tool_name
    - Action Input: {...}
    - Final Answer: ...
    """
    step = ReActStep(raw_output=text)
    text = _safe_str(text)
    
    if not text:
        return step
    
    # details Thought
    thought_match = re.search(r"Though<LOCAL_PATH>)(:=Action:|Final Answer:|$)", text, re.DOTALL | re.IGNORECASE)
    if thought_match:
        step.thought = thought_match.group(1).strip()
    
    # details Final Answer
    final_match = re.search(r"Final Answe<LOCAL_PATH>)", text, re.DOTALL | re.IGNORECASE)
    if final_match:
        step.final_answer = final_match.group(1).strip()
        step.is_final = True
        return step
    
    # details Action
    action_match = re.search(r"Actio<LOCAL_PATH>)", text, re.IGNORECASE)
    if action_match:
        step.action = action_match.group(1).strip()
    
    # details Action Input
    # details：JSON details、```json details、details JSON
    action_input_patterns = [
        r"Action Inpu<LOCAL_PATH>)\s*```",
        r"Action Inpu<LOCAL_PATH>)\s*```",
        r"Action Inpu<LOCAL_PATH>)",
        r"Action Inpu<LOCAL_PATH>)(:=Thought:|Action:|Final Answer:|Observation:|$)",
    ]
    
    for pattern in action_input_patterns:
        input_match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if input_match:
            input_str = input_match.group(1).strip()
            try:
                # details JSON
                step.action_input = json.loads(input_str)
                break
            except json.JSONDecodeError:
                # details
                try:
                    # details
                    fixed = re.sub(r",\s*}", "}", input_str)
                    fixed = re.sub(r",\s*]", "]", fixed)
                    step.action_input = json.loads(fixed)
                    break
                except json.JSONDecodeError:
                    # details，details
                    if step.action:
                        step.action_input = {"query": input_str}
                    continue
    
    return step


# ============================================================
# ReAct Agent Loop
# ============================================================

@dataclass
class ReActLoopConfig:
    """ReAct Loop details"""
    max_iterations: int = 5
    max_observation_chars: int = 2000
    temperature: float = 0.1
    max_new_tokens: int = 1500
    stop_on_final_answer: bool = True
    verbose: bool = False


@dataclass
class ReActLoopResult:
    """ReAct Loop details"""
    success: bool
    final_answer: str = ""
    steps: List[ReActStep] = field(default_factory=list)
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    total_iterations: int = 0
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "final_answer": self.final_answer,
            "total_iterations": self.total_iterations,
            "tool_calls": self.tool_calls,
            "error": self.error,
        }


class ReActAgentLoop:
    """
    ReAct Agent Loop details。
    
    details LLM details，details。
    """
    
    def __init__(
        self,
        registry: ToolRegistry,
        llm_generate_fn: Callable[[str, str], str],
        config: Optional[ReActLoopConfig] = None,
    ):
        """
        Args:
            registry: details
            llm_generate_fn: LLM details，details (system_prompt, user_prompt) -> str
            config: details
        """
        self.registry = registry
        self.llm_generate = llm_generate_fn
        self.config = config or ReActLoopConfig()
    
    def build_system_prompt(
        self,
        additional_instructions: str = "",
    ) -> str:
        """details prompt"""
        tools_desc = self.registry.get_tools_prompt()
        
        return REACT_SYSTEM_PROMPT_TEMPLATE.format(
            tools_description=tools_desc,
            max_iterations=self.config.max_iterations,
            additional_instructions=additional_instructions,
        )
    
    def build_user_prompt(
        self,
        task_description: str,
        input_data: str,
    ) -> str:
        """details prompt"""
        return REACT_USER_PROMPT_TEMPLATE.format(
            task_description=task_description,
            input_data=input_data,
        )
    
    def run(
        self,
        task_description: str,
        input_data: str,
        additional_instructions: str = "",
    ) -> ReActLoopResult:
        """
        details ReAct Loop。
        
        Args:
            task_description: details
            input_data: details
            additional_instructions: details
        
        Returns:
            ReActLoopResult
        """
        result = ReActLoopResult(success=False)
        
        system_prompt = self.build_system_prompt(additional_instructions)
        user_prompt = self.build_user_prompt(task_description, input_data)
        
        # details
        conversation = user_prompt
        
        for iteration in range(self.config.max_iterations):
            result.total_iterations = iteration + 1
            
            if self.config.verbose:
                print(f"\n[ReAct] Iteration {iteration + 1}/{self.config.max_iterations}")
            
            # details LLM
            try:
                llm_output = self.llm_generate(system_prompt, conversation)
            except Exception as e:
                result.error = f"LLM generation failed: {str(e)}"
                return result
            
            if self.config.verbose:
                print(f"[ReAct] LLM outpu<LOCAL_PATH>")
            
            # details
            step = parse_react_output(llm_output)
            result.steps.append(step)
            
            # details
            if step.is_final:
                result.success = True
                result.final_answer = step.final_answer
                if self.config.verbose:
                    print(f"[ReAct] Final answer received")
                return result
            
            # details
            if not step.action:
                # details action details final answer，details
                result.error = "LLM did not provide action or final answer"
                # details thought details
                if step.thought:
                    result.final_answer = step.thought
                    result.success = True
                return result
            
            # details
            tool_call_record = {
                "iteration": iteration + 1,
                "action": step.action,
                "action_input": step.action_input,
            }
            
            if self.config.verbose:
                print(f"[ReAct] Calling tool: {step.action}")
                print(f"[ReAct] Input: {json.dumps(step.action_input, ensure_ascii=False)}")
            
            tool_result = self.registry.execute(step.action, **step.action_input)
            observation = tool_result.to_observation_str(self.config.max_observation_chars)
            
            tool_call_record["observation"] = observation[:500] + "..." if len(observation) > 500 else observation
            tool_call_record["success"] = tool_result.success
            result.tool_calls.append(tool_call_record)
            
            if self.config.verbose:
                print(f"[ReAct] Observation: {observation[:300]}...")
            
            # details
            conversation += f"\n\n{llm_output}\n\nObservation: {observation}\n\nNow continue. If you have enough information, provide your Final Answer. Otherwise, use another tool."
        
        # details
        result.error = f"Max iterations ({self.config.max_iterations}) reached without final answer"
        
        # details
        if result.steps:
            last_step = result.steps[-1]
            if last_step.thought:
                result.final_answer = last_step.thought
                result.success = True
        
        return result


# ============================================================
# details：details Agent
# ============================================================

def create_detection_agent_tools(registry: ToolRegistry) -> None:
    """details DetectionAgent details"""
    from .ddgs_search import DDGSSearchTool, DisasterEventSearchTool
    from .rag_tools import InternalRAGTool, EventContextTool
    
    registry.register(InternalRAGTool())
    registry.register(DDGSSearchTool())
    registry.register(DisasterEventSearchTool())
    registry.register(EventContextTool())


def create_assessment_agent_tools(registry: ToolRegistry) -> None:
    """details AssessmentAgent details"""
    from .ddgs_search import DDGSSearchTool
    from .rag_tools import GovRulesRAGTool, HistoryCasesRAGTool
    
    registry.register(GovRulesRAGTool())
    registry.register(HistoryCasesRAGTool())
    registry.register(DDGSSearchTool())


def create_critic_agent_tools(registry: ToolRegistry) -> None:
    """details CriticAgent details"""
    from .ddgs_search import DDGSSearchTool
    from .rag_tools import GovRulesRAGTool
    
    registry.register(GovRulesRAGTool())
    registry.register(DDGSSearchTool())
