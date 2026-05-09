#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Agents/tools/config.py

details，details RAG details。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


# ============================================================
# RAG details
# ============================================================

@dataclass
class RAGThresholds:
    """RAG details"""
    
    # sufficient: RAG details，details
    sufficient_top1_score: float
    sufficient_relevant_hits: int
    sufficient_coverage: float
    
    # insufficient: RAG details，details
    insufficient_top1_score: float
    insufficient_relevant_hits: int
    insufficient_coverage: float
    
    def judge_status(
        self,
        top1_score: float,
        relevant_hit_count: int,
        evidence_coverage: float,
    ) -> str:
        """
        details RAG details。
        
        Returns:
            "sufficient" | "borderline" | "insufficient"
        """
        # details
        if (
            top1_score < self.insufficient_top1_score
            or relevant_hit_count < self.insufficient_relevant_hits
            or evidence_coverage < self.insufficient_coverage
        ):
            return "insufficient"
        
        # details
        if (
            top1_score >= self.sufficient_top1_score
            and relevant_hit_count >= self.sufficient_relevant_hits
            and evidence_coverage >= self.sufficient_coverage
        ):
            return "sufficient"
        
        # details
        return "borderline"


# DetectionAgent details
DETECTION_RAG_THRESHOLDS = RAGThresholds(
    sufficient_top1_score=0.55,
    sufficient_relevant_hits=2,
    sufficient_coverage=0.60,
    insufficient_top1_score=0.45,
    insufficient_relevant_hits=0,
    insufficient_coverage=0.40,
)

# AssessmentAgent details（details）
ASSESSMENT_RAG_THRESHOLDS = RAGThresholds(
    sufficient_top1_score=0.60,
    sufficient_relevant_hits=3,
    sufficient_coverage=0.70,
    insufficient_top1_score=0.50,
    insufficient_relevant_hits=2,
    insufficient_coverage=0.50,
)

# CriticAgent details
CRITIC_RAG_THRESHOLDS = RAGThresholds(
    sufficient_top1_score=0.58,
    sufficient_relevant_hits=2,
    sufficient_coverage=0.60,
    insufficient_top1_score=0.48,
    insufficient_relevant_hits=0,
    insufficient_coverage=0.45,
)


# ============================================================
# ReAct Loop details
# ============================================================

# DetectionAgent ReAct details
DETECTION_REACT_CONFIG = {
    "max_iterations": 4,
    "max_observation_chars": 2000,
    "temperature": 0.1,
    "max_new_tokens": 1800,
    "verbose": False,
}

# AssessmentAgent ReAct details
ASSESSMENT_REACT_CONFIG = {
    "max_iterations": 3,
    "max_observation_chars": 1500,
    "temperature": 0.1,
    "max_new_tokens": 1500,
    "verbose": False,
}

# CriticAgent ReAct details
CRITIC_REACT_CONFIG = {
    "max_iterations": 3,
    "max_observation_chars": 1500,
    "temperature": 0.1,
    "max_new_tokens": 1200,
    "verbose": False,
}


# ============================================================
# details（details）
# ============================================================

EVENT_DEPENDENT_CHANGE_TYPES = {
    # details
    "flood", "flooding", "standing_water", "puddle", "water_mark",
    "mud", "silt", "sediment", "erosion", "storm_surge",
    "water_stain", "high_water_mark", "debris_line",
    
    # details/details
    "debris_clearing", "cleanup", "temporary_barrier", "reconstruction",
    "repair", "restoration", "recovery",
    
    # details
    "temporary_structure", "tarp", "plywood", "boarding",
    
    # details（details）
    "vegetation_change", "tree_damage", "fallen_tree", "defoliation",
}

# details/details（AssessmentAgent details）
RARE_OR_UNDERCOVERED_DAMAGE_TYPES = {
    "liquefaction", "landslide", "sinkhole",
    "tsunami", "volcanic",
    "industrial_damage", "infrastructure_failure",
    "bridge_damage", "dam_damage",
}


# ============================================================
# Web Search details
# ============================================================

WEB_SEARCH_CONFIG = {
    # details
    "default_max_results": 5,
    "default_region": "wt-wt",
    "timeout": 10,
    
    # details
    "official_source_keywords": [
        "FEMA", "NOAA", "USGS", "NWS",
        "official", "government", "gov",
        "damage report", "assessment report",
    ],
    
    # details
    "exclude_domains": [
        "reddit.com", "twitter.com", "facebook.com",
        "youtube.com", "tiktok.com",
    ],
}
