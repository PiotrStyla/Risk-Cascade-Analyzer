"""
Conditional Probability Table (CPT) templates and generators.

CPTs define P(child | parents) - how parent node states influence child node probabilities.
This module provides both pre-defined templates and functions to generate custom CPTs.
"""

from typing import Dict, List, Tuple, Callable
import numpy as np
from itertools import product


class CPTTemplate:
    """Base class for CPT generation strategies."""
    
    def generate(self, child_states: List[str], parent_states: Dict[str, List[str]]) -> Dict:
        """
        Generate CPT for a child node given its parents.
        
        Args:
            child_states: List of possible states for the child node
            parent_states: Dict mapping parent node IDs to their possible states
            
        Returns:
            Dictionary mapping parent state combinations to child state probabilities
        """
        raise NotImplementedError


class LinearCPT(CPTTemplate):
    """
    Linear influence model: more negative parent states increase probability of negative child states.
    
    Example: High fatigue + High stress → High probability of error
    """
    
    def __init__(self, influence_weights: Dict[str, float] = None):
        """
        Args:
            influence_weights: Optional weights for each parent's influence (default: equal weight)
        """
        self.influence_weights = influence_weights or {}
    
    def generate(self, child_states: List[str], parent_states: Dict[str, List[str]]) -> Dict:
        """Generate CPT with linear additive influence."""
        cpt = {}
        
        # Generate all combinations of parent states
        parent_ids = list(parent_states.keys())
        parent_state_lists = [parent_states[pid] for pid in parent_ids]
        
        for parent_combo in product(*parent_state_lists):
            # Calculate influence score (0 to 1, where 1 = maximum negative influence)
            influence_score = 0.0
            weight_sum = 0.0
            
            for parent_id, state in zip(parent_ids, parent_combo):
                weight = self.influence_weights.get(parent_id, 1.0)
                # Assume states are ordered from positive to negative
                state_index = parent_states[parent_id].index(state)
                state_score = state_index / (len(parent_states[parent_id]) - 1)
                influence_score += weight * state_score
                weight_sum += weight
            
            influence_score /= weight_sum if weight_sum > 0 else 1.0
            
            # Map influence score to child state probabilities
            child_probs = self._score_to_probabilities(influence_score, len(child_states))
            
            # Store as tuple key for parent combo
            key = tuple(parent_combo)
            cpt[key] = dict(zip(child_states, child_probs))
        
        return cpt
    
    def _score_to_probabilities(self, score: float, num_states: int) -> List[float]:
        """
        Convert influence score to probability distribution over child states.
        
        Uses a softmax-like distribution centered around score position.
        """
        # Map score (0 to 1) to state index (0 to num_states-1)
        center = score * (num_states - 1)
        
        # Create distribution peaked at center
        probs = []
        for i in range(num_states):
            distance = abs(i - center)
            prob = np.exp(-distance * 2)  # Exponential decay
            probs.append(prob)
        
        # Normalize
        total = sum(probs)
        return [p / total for p in probs]


class ThresholdCPT(CPTTemplate):
    """
    Threshold model: child state changes only when parent combination exceeds threshold.
    
    Example: Error occurs only when BOTH fatigue AND stress are high.
    """
    
    def __init__(self, threshold_rules: List[Tuple[Dict[str, str], Dict[str, float]]]):
        """
        Args:
            threshold_rules: List of (parent_conditions, child_probs) tuples
                Example: [
                    ({"fatigue": "high", "stress": "high"}, {"error": 0.8, "no_error": 0.2}),
                    ({"fatigue": "high"}, {"error": 0.3, "no_error": 0.7}),
                ]
        """
        self.threshold_rules = threshold_rules
    
    def generate(self, child_states: List[str], parent_states: Dict[str, List[str]]) -> Dict:
        """Generate CPT based on threshold rules."""
        cpt = {}
        
        parent_ids = list(parent_states.keys())
        parent_state_lists = [parent_states[pid] for pid in parent_ids]
        
        # Default probabilities (if no rule matches)
        default_probs = {state: 1.0 / len(child_states) for state in child_states}
        
        for parent_combo in product(*parent_state_lists):
            parent_dict = dict(zip(parent_ids, parent_combo))
            
            # Find matching rule (first match wins, rules should be ordered by specificity)
            matched_probs = default_probs
            for conditions, child_probs in self.threshold_rules:
                if self._matches_conditions(parent_dict, conditions):
                    matched_probs = child_probs
                    break
            
            key = tuple(parent_combo)
            cpt[key] = matched_probs.copy()
        
        return cpt
    
    def _matches_conditions(self, parent_dict: Dict[str, str], conditions: Dict[str, str]) -> bool:
        """Check if parent states match all conditions."""
        return all(parent_dict.get(pid) == state for pid, state in conditions.items())


class DeterministicCPT(CPTTemplate):
    """
    Deterministic relationship: child state is determined exactly by parent states.
    
    Example: Hazard + Initiator → Catastrophe (always)
    """
    
    def __init__(self, rule_function: Callable[[Dict[str, str]], str]):
        """
        Args:
            rule_function: Function mapping parent states dict to child state
        """
        self.rule_function = rule_function
    
    def generate(self, child_states: List[str], parent_states: Dict[str, List[str]]) -> Dict:
        """Generate deterministic CPT."""
        cpt = {}
        
        parent_ids = list(parent_states.keys())
        parent_state_lists = [parent_states[pid] for pid in parent_ids]
        
        for parent_combo in product(*parent_state_lists):
            parent_dict = dict(zip(parent_ids, parent_combo))
            
            # Determine child state
            determined_state = self.rule_function(parent_dict)
            
            # Create probability distribution (1.0 for determined state, 0.0 for others)
            child_probs = {state: 1.0 if state == determined_state else 0.0 
                          for state in child_states}
            
            key = tuple(parent_combo)
            cpt[key] = child_probs
        
        return cpt


# ============================================================================
# PRE-DEFINED CPT TEMPLATES FOR COMMON RELATIONSHIPS
# ============================================================================

def cpt_pressure_to_fatigue() -> ThresholdCPT:
    """Organizational pressure → Worker fatigue"""
    return ThresholdCPT([
        ({"org_output_pressure": "high", "org_resource_allocation": "insufficient"}, 
         {"rested": 0.05, "tired": 0.35, "exhausted": 0.60}),
        ({"org_output_pressure": "high"}, 
         {"rested": 0.15, "tired": 0.50, "exhausted": 0.35}),
        ({"org_output_pressure": "medium"}, 
         {"rested": 0.40, "tired": 0.45, "exhausted": 0.15}),
        ({}, 
         {"rested": 0.70, "tired": 0.25, "exhausted": 0.05}),
    ])


def cpt_pressure_to_stress() -> ThresholdCPT:
    """Organizational pressure + KPI design → Psychological stress"""
    return ThresholdCPT([
        ({"org_output_pressure": "high", "org_kpi_design": "unbalanced"}, 
         {"low": 0.05, "moderate": 0.25, "high": 0.70}),
        ({"org_output_pressure": "high"}, 
         {"low": 0.10, "moderate": 0.40, "high": 0.50}),
        ({"org_output_pressure": "medium"}, 
         {"low": 0.30, "moderate": 0.50, "high": 0.20}),
        ({}, 
         {"low": 0.60, "moderate": 0.30, "high": 0.10}),
    ])


def cpt_stress_to_time_pressure() -> LinearCPT:
    """Psychological stress → Perceived time pressure"""
    return LinearCPT(influence_weights={"human_stress": 1.5})


def cpt_fatigue_stress_to_attention() -> LinearCPT:
    """Fatigue + Stress → Attention degradation"""
    return LinearCPT(influence_weights={
        "human_fatigue": 1.2,
        "human_stress": 0.8
    })


def cpt_multiple_to_error() -> ThresholdCPT:
    """Fatigue + Stress + Time Pressure → Human Error"""
    return ThresholdCPT([
        # Critical combination: all factors negative
        ({"human_fatigue": "exhausted", "human_stress": "high", "human_time_pressure": "frantic"},
         {"no_error": 0.05, "minor_error": 0.30, "critical_error": 0.65}),
        
        # Two factors highly negative
        ({"human_fatigue": "exhausted", "human_stress": "high"},
         {"no_error": 0.15, "minor_error": 0.45, "critical_error": 0.40}),
        ({"human_stress": "high", "human_time_pressure": "frantic"},
         {"no_error": 0.20, "minor_error": 0.50, "critical_error": 0.30}),
        
        # One factor highly negative
        ({"human_fatigue": "exhausted"},
         {"no_error": 0.50, "minor_error": 0.40, "critical_error": 0.10}),
        ({"human_stress": "high"},
         {"no_error": 0.60, "minor_error": 0.30, "critical_error": 0.10}),
        
        # Default (favorable conditions)
        ({},
         {"no_error": 0.95, "minor_error": 0.04, "critical_error": 0.01}),
    ])


def cpt_error_to_hazard() -> DeterministicCPT:
    """Critical Error → Physical Hazard (deterministic)"""
    def rule(parents: Dict[str, str]) -> str:
        if parents.get("human_error") == "critical_error":
            return "hazardous"
        return "safe"
    
    return DeterministicCPT(rule)


def cpt_hazard_initiator_to_catastrophe() -> DeterministicCPT:
    """Physical Hazard + Initiating Event → Catastrophe"""
    def rule(parents: Dict[str, str]) -> str:
        if (parents.get("tech_hazard_present") == "hazardous" and 
            parents.get("tech_initiating_event") == "present"):
            return "occurred"
        return "prevented"
    
    return DeterministicCPT(rule)


# ============================================================================
# CPT REGISTRY
# ============================================================================

CPT_REGISTRY = {
    "pressure_to_fatigue": cpt_pressure_to_fatigue,
    "pressure_to_stress": cpt_pressure_to_stress,
    "stress_to_time_pressure": cpt_stress_to_time_pressure,
    "fatigue_stress_to_attention": cpt_fatigue_stress_to_attention,
    "multiple_to_error": cpt_multiple_to_error,
    "error_to_hazard": cpt_error_to_hazard,
    "hazard_initiator_to_catastrophe": cpt_hazard_initiator_to_catastrophe,
}


def get_cpt_template(template_id: str) -> CPTTemplate:
    """Retrieve a pre-defined CPT template by ID."""
    if template_id not in CPT_REGISTRY:
        raise KeyError(f"CPT template '{template_id}' not found in registry")
    return CPT_REGISTRY[template_id]()
