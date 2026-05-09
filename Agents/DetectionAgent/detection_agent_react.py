#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DetectionAgent/detection_agent_react.py

ReAct details DetectionAgent：
- details LLM details
- details RAG + details
- details
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# details
_AGENT_DIR = Path(__file__).resolve().parent
_AGENTS_ROOT = _AGENT_DIR.parent
if str(_AGENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTS_ROOT))
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))

from tools.registry import ToolRegistry, ToolResult
from tools.react_loop import (
    ReActAgentLoop,
    ReActLoopConfig,
    ReActLoopResult,
    create_detection_agent_tools,
)
from tools.config import DETECTION_REACT_CONFIG, DETECTION_RAG_THRESHOLDS

from shared_llm import get_shared_llm


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    try:
        return str(x).strip()
    except Exception:
        return ""


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    """details JSON details"""
    t = _safe_str(text)
    if not t:
        return None
    
    # details ```json details
    m = re.search(r"```json\s*([\s\S]*:)\s*```", t, flags=re.IGNORECASE)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    
    # details { ... }
    l = t.find("{")
    r = t.rfind("}")
    if l >= 0 and r > l:
        try:
            obj = json.loads(t[l:r+1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    
    return None


@dataclass
class DetectionMemoryStep:
    """
    details memory details（details agent loop details）：
    - deliberation: details
    - action: details
    - observation: details
    - memory_update: details
    """

    iteration: int
    deliberation: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    memory_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionMemory:
    """
    Detection Agent details memory schema。
    details/details。
    """

    goal: str = ""
    hazard_type: str = ""
    current_hypothesis: str = ""
    missing_evidence: List[str] = field(default_factory=list)
    evidence_sufficient: bool = False
    stop_reason: str = ""
    steps: List[DetectionMemoryStep] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.goal,
            "hazard_type": self.hazard_type,
            "current_hypothesis": self.current_hypothesis,
            "missing_evidence": self.missing_evidence,
            "evidence_sufficient": self.evidence_sufficient,
            "stop_reason": self.stop_reason,
            "steps": [asdict(s) for s in self.steps],
        }


# ============================================================
# ReAct Detection Agent
# ============================================================

DETECTION_TASK_DESCRIPTION = """You are a Detection Agent for disaster damage assessment.

Your task is to:
1. Analyze the changes between pre-disaster and post-disaster descriptions
2. Determine which changes are likely disaster-related vs. pseudo-changes (seasonal, viewpoint, etc.)
3. Classify each change into one of three categories:
   - confirmed_disaster_damage: High confidence disaster-related
   - likely_pseudo_change: High confidence NOT disaster-related
   - uncertain: Insufficient evidence to determine

You have access to tools to help you:
- search_internal_rag: Search internal knowledge base for damage patterns and rules
- web_search: Search the web for disaster event context
- search_disaster_event: Search for specific disaster events at a location/time
- lookup_event_context: Look up event context for coordinates

Strategy:
1. First, use search_internal_rag to find relevant damage patterns for the observed changes
2. If internal RAG is insufficient, use web_search or search_disaster_event to verify if a disaster actually occurred
3. Make your classification based on the evidence gathered

Output your final answer as a JSON object with this structure:
{
  "confirmed_disaster_damage": [...],
  "likely_pseudo_change": [...],
  "uncertain": [...],
  "reasoning_summary": "..."
}
"""

DETECTION_ADDITIONAL_INSTRUCTIONS = """
## Classification Guidelines

For each change, you must provide:
- id: unique identifier
- component: affected component (roof, facade, road, etc.)
- change_description: what changed
- disaster_related: "likely" | "possible" | "unlikely" | "unknown"
- reason: why you classified it this way
- confidence: 0.0-1.0

## When to use web search

Use web_search or search_disaster_event when:
- Internal RAG results are insufficient (low scores, few relevant hits)
- The change type is highly event-dependent (flooding, storm surge, debris)
- You need to verify if a disaster actually occurred at this location/time

Do NOT use web search when:
- Internal RAG provides clear, relevant rules
- The change is obviously structural damage (collapsed roof, wall failure)
- You already have enough evidence to classify

## Output Format

Your Final Answer MUST be a valid JSON object. Do not include any text outside the JSON.
"""


@dataclass
class DetectionAgentReAct:
    """
    ReAct details DetectionAgent。
    
    details LLM details。
    """
    
    model_path: str = ""
    hazard_type: str = "hurricane"
    max_iterations: int = DETECTION_REACT_CONFIG["max_iterations"]
    max_new_tokens: int = DETECTION_REACT_CONFIG["max_new_tokens"]
    temperature: float = DETECTION_REACT_CONFIG["temperature"]
    verbose: bool = DETECTION_REACT_CONFIG["verbose"]
    
    # details
    _registry: ToolRegistry = field(default_factory=ToolRegistry, repr=False)
    _tokenizer: Any = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _torch: Any = field(default=None, repr=False)
    _initialized: bool = field(default=False, repr=False)

    @staticmethod
    def memory_schema() -> Dict[str, Any]:
        """details memory schema（details）"""
        return DetectionMemory(
            goal="Classify changes into confirmed_disaster_damage / likely_pseudo_change / uncertain",
            hazard_type="hurricane",
        ).to_dict()
    
    def __post_init__(self):
        # details
        create_detection_agent_tools(self._registry)
    
    def _ensure_model(self) -> None:
        """details"""
        if self._initialized:
            return
        
        if not self.model_path:
            # details config details
            try:
                import config as det_cfg
                self.model_path = str(getattr(det_cfg, "LOCAL_LLM_MODEL_PATH", ""))
            except ImportError:
                pass
        
        if not self.model_path:
            raise RuntimeError("Model path not specified")
        
        self._tokenizer, self._model, self._torch = get_shared_llm(self.model_path)
        self._initialized = True
    
    def _llm_generate(self, system_prompt: str, user_prompt: str) -> str:
        """LLM details"""
        self._ensure_model()
        
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            text_input = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text_input = system_prompt + "\n\n" + user_prompt
        
        model_inputs = self._tokenizer([text_input], return_tensors="pt").to(self._model.device)
        
        do_sample = bool(float(self.temperature) > 1e-6)
        
        with self._torch.no_grad():
            generated_ids = self._model.generate(
                **model_inputs,
                max_new_tokens=int(self.max_new_tokens),
                temperature=float(self.temperature) if do_sample else None,
                do_sample=do_sample,
                top_p=0.9 if do_sample else None,
                repetition_penalty=1.05,
            )
        
        gen_only = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        return self._tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]
    
    def _build_input_data(
        self,
        pre_text: str,
        post_text: str,
        pair_meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """details"""
        parts = []
        
        parts.append("## Pre-disaster Description")
        parts.append(pre_text)
        parts.append("")
        parts.append("## Post-disaster Description")
        parts.append(post_text)
        
        if pair_meta:
            parts.append("")
            parts.append("## Metadata")
            if pair_meta.get("pre_date"):
                parts.append(f"- Pre-disaster date: {pair_meta['pre_date']}")
            if pair_meta.get("post_date"):
                parts.append(f"- Post-disaster date: {pair_meta['post_date']}")
            if pair_meta.get("lat") and pair_meta.get("lon"):
                parts.append(f"- Location: ({pair_meta['lat']}, {pair_meta['lon']})")
            if pair_meta.get("hazard_type"):
                parts.append(f"- Hazard type: {pair_meta['hazard_type']}")
        
        parts.append("")
        parts.append(f"## Target Hazard Type: {self.hazard_type}")
        
        return "\n".join(parts)
    
    def _parse_final_answer(self, answer: str) -> Dict[str, Any]:
        """details"""
        # details JSON
        obj = _extract_json_object(answer)
        
        if obj:
            # details
            if "confirmed_disaster_damage" not in obj:
                obj["confirmed_disaster_damage"] = []
            if "likely_pseudo_change" not in obj:
                obj["likely_pseudo_change"] = []
            if "uncertain" not in obj:
                obj["uncertain"] = []
            return obj
        
        # details，details
        return {
            "confirmed_disaster_damage": [],
            "likely_pseudo_change": [],
            "uncertain": [],
            "reasoning_summary": answer,
            "_parse_failed": True,
        }

    def _build_memory_trace(
        self,
        *,
        loop_result: ReActLoopResult,
        input_data: str,
    ) -> DetectionMemory:
        """
        details ReAct details memory trace，details agent loop details。
        """
        mem = DetectionMemory(
            goal="Classify post-disaster changes with tool-supported evidence",
            hazard_type=self.hazard_type,
        )

        # details iteration -> tool_call details
        tool_by_iter: Dict[int, Dict[str, Any]] = {}
        for call in loop_result.tool_calls:
            try:
                it = int(call.get("iteration", 0))
            except Exception:
                it = 0
            if it > 0:
                tool_by_iter[it] = call

        for i, step in enumerate(loop_result.steps, start=1):
            tool_call = tool_by_iter.get(i, {})
            observation = _safe_str(tool_call.get("observation", ""))
            if len(observation) > 500:
                observation = observation[:500] + "..."

            memory_update = {
                "tool_success": bool(tool_call.get("success", False)),
                "observation_summary": observation,
            }

            # details evidence insufficiency details
            low_signal = any(
                kw in observation.lower()
                for kw in ["no results", "insufficient", "failed", "error"]
            )
            if low_signal:
                mem.missing_evidence.append(f"iteration_{i}: retrieval quality low")

            mem.steps.append(
                DetectionMemoryStep(
                    iteration=i,
                    deliberation=_safe_str(step.thought),
                    action=_safe_str(step.action),
                    action_input=step.action_input if isinstance(step.action_input, dict) else {},
                    observation=observation,
                    memory_update=memory_update,
                )
            )

            # details hypothesis（details）
            if step.thought:
                mem.current_hypothesis = _safe_str(step.thought)[:300]

        mem.evidence_sufficient = bool(loop_result.success and not loop_result.error)
        if loop_result.error:
            mem.stop_reason = _safe_str(loop_result.error)
        elif loop_result.success:
            mem.stop_reason = "final_answer"
        else:
            mem.stop_reason = "max_iterations_or_no_action"
        return mem
    
    def run(
        self,
        pre_text: str,
        post_text: str,
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
        max_changes: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        details ReAct details。
        
        Args:
            pre_text: details
            post_text: details
            pair_meta: details（details、details）
            max_changes: details（details，ReAct details）
        
        Returns:
            details，details DetectionAgent details
        """
        pre_text = _safe_str(pre_text)
        post_text = _safe_str(post_text)
        
        if not pre_text or not post_text:
            return {
                "confirmed_disaster_damage": [],
                "likely_pseudo_change": [],
                "uncertain": [],
                "error": "Empty input text",
            }
        
        # details
        input_data = self._build_input_data(pre_text, post_text, pair_meta)
        
        # details ReAct Loop
        config = ReActLoopConfig(
            max_iterations=self.max_iterations,
            max_observation_chars=DETECTION_REACT_CONFIG["max_observation_chars"],
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            verbose=self.verbose,
        )
        
        loop = ReActAgentLoop(
            registry=self._registry,
            llm_generate_fn=self._llm_generate,
            config=config,
        )
        
        # details ReAct Loop
        result = loop.run(
            task_description=DETECTION_TASK_DESCRIPTION,
            input_data=input_data,
            additional_instructions=DETECTION_ADDITIONAL_INSTRUCTIONS,
        )
        
        # details
        output = self._parse_final_answer(result.final_answer)
        
        # details
        memory = self._build_memory_trace(loop_result=result, input_data=input_data)
        output["_react_meta"] = {
            "success": result.success,
            "total_iterations": result.total_iterations,
            "tool_calls": result.tool_calls,
            "error": result.error,
            "loop_pattern": [
                "deliberation",
                "tool_call",
                "observation",
                "memory_update",
            ],
            "memory_schema": self.memory_schema(),
            "memory_trace": memory.to_dict(),
        }
        
        return output
    
    def run_with_fallback(
        self,
        pre_text: str,
        post_text: str,
        *,
        pair_meta: Optional[Dict[str, Any]] = None,
        max_changes: Optional[int] = None,
        fallback_agent: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        details ReAct details，details Agent。
        
        Args:
            fallback_agent: details DetectionAgent details
        """
        try:
            result = self.run(pre_text, post_text, pair_meta=pair_meta, max_changes=max_changes)
            
            # details
            if result.get("_parse_failed") or result.get("error"):
                if fallback_agent:
                    print("[DetectionAgentReAct] ReAct failed, falling back to original agent")
                    return fallback_agent.run(pre_text, post_text, pair_meta=pair_meta, max_changes=max_changes)
            
            return result
        
        except Exception as e:
            if fallback_agent:
                print(f"[DetectionAgentReAct] ReAct exception: {e}, falling back to original agent")
                return fallback_agent.run(pre_text, post_text, pair_meta=pair_meta, max_changes=max_changes)
            raise


# ============================================================
# details
# ============================================================

_GLOBAL_REACT_AGENT: Optional[DetectionAgentReAct] = None


def get_detection_agent_react(model_path: str = "") -> DetectionAgentReAct:
    """details ReAct DetectionAgent details"""
    global _GLOBAL_REACT_AGENT
    
    if _GLOBAL_REACT_AGENT is None:
        _GLOBAL_REACT_AGENT = DetectionAgentReAct(model_path=model_path)
    
    return _GLOBAL_REACT_AGENT


def run_detection_react(
    pre_text: str,
    post_text: str,
    *,
    pair_meta: Optional[Dict[str, Any]] = None,
    model_path: str = "",
    verbose: bool = False,
) -> Dict[str, Any]:
    """details：details ReAct details"""
    agent = get_detection_agent_react(model_path)
    agent.verbose = verbose
    return agent.run(pre_text, post_text, pair_meta=pair_meta)
