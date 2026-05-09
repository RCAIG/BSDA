from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch

from src.knowledge_base import KnowledgeBase


@dataclass
class DGAConfig:
    """
    Baseline (non-RL) DGA config.
    """

    # If True: directly use KB suggested grade
    use_kb_rule_grade: bool = True


class DGAGradingAgent:
    """
    DGA (Damage Grading Agent).

    Baseline version:
    - call KnowledgeBase.get_grading_suggestion(damage_list)
    - output suggested_grade

    PPO version (next stage) will:
    - use DGAPolicy network
    - return action_log_prob/entropy tensors for PPO
    """

    def __init__(self, knowledge_base: KnowledgeBase, config: Optional[DGAConfig] = None) -> None:
        self.kb = knowledge_base
        self.config = config or DGAConfig()

    def decide_grade(
        self,
        damage_list: List[Dict[str, Any]],
        deterministic: bool = True,
    ) -> Tuple[int, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        sugg = self.kb.get_grading_suggestion(damage_list)
        grade = int(sugg["suggested_grade"])
        info = {"kb_suggestion": sugg}
        return grade, None, None, info


