"""
Universal node types for Bayesian cascade networks.

Each node represents a state, event, or factor that can influence system behavior.
Nodes are organized into three hierarchical levels:
- Level 3: Organizational/System factors (root causes)
- Level 2: Human/Psychological factors (intermediate states)
- Level 1: Technical/Physical factors (observable outcomes)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Optional


class NodeLevel(Enum):
    """Hierarchical level of the node in cascade network."""
    ORGANIZATIONAL = 3  # Management decisions, policies, culture
    HUMAN = 2           # Worker states, psychological factors
    TECHNICAL = 1       # Physical systems, observable events


class NodeCategory(Enum):
    """Broad category of node."""
    # Organizational
    CULTURE = "culture"
    POLICY = "policy"
    RESOURCE = "resource"
    PRESSURE = "pressure"
    
    # Human
    PHYSIOLOGICAL = "physiological"
    PSYCHOLOGICAL = "psychological"
    BEHAVIORAL = "behavioral"
    COMPETENCY = "competency"
    
    # Technical
    ENVIRONMENTAL = "environmental"
    EQUIPMENT = "equipment"
    PROCESS = "process"
    INCIDENT = "incident"


@dataclass
class NodeDefinition:
    """Definition of a universal node type."""
    id: str
    name: str
    description: str
    level: NodeLevel
    category: NodeCategory
    states: List[str]  # Possible discrete states (e.g., ["low", "medium", "high"])
    default_probability: Optional[Dict[str, float]] = None  # Prior P(node) if root
    
    def __post_init__(self):
        """Validate that states are properly defined."""
        if not self.states:
            raise ValueError(f"Node {self.id} must have at least one state")
        if self.default_probability:
            if not abs(sum(self.default_probability.values()) - 1.0) < 0.001:
                raise ValueError(f"Node {self.id} probabilities must sum to 1.0")


# ============================================================================
# ORGANIZATIONAL NODES (Level 3)
# ============================================================================

ORGANIZATIONAL_NODES = [
    NodeDefinition(
        id="org_output_pressure",
        name="Output Pressure",
        description="Management pressure for productivity/performance over safety",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.PRESSURE,
        states=["low", "medium", "high"],
        default_probability={"low": 0.3, "medium": 0.5, "high": 0.2}
    ),
    NodeDefinition(
        id="org_safety_culture",
        name="Safety Culture Quality",
        description="How well organization prioritizes and normalizes safety practices",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.CULTURE,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.3, "adequate": 0.5, "excellent": 0.2}
    ),
    NodeDefinition(
        id="org_kpi_design",
        name="KPI Design Quality",
        description="Whether performance metrics balance multiple objectives or focus narrowly on output",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["unbalanced", "balanced"],
        default_probability={"unbalanced": 0.6, "balanced": 0.4}
    ),
    NodeDefinition(
        id="org_resource_allocation",
        name="Resource Allocation",
        description="Adequacy of resources (staff, time, equipment) for safe operations",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.RESOURCE,
        states=["insufficient", "adequate", "abundant"],
        default_probability={"insufficient": 0.4, "adequate": 0.5, "abundant": 0.1}
    ),
    NodeDefinition(
        id="org_training_quality",
        name="Training Program Quality",
        description="Effectiveness and recency of safety and operational training",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.COMPETENCY,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.3, "adequate": 0.5, "excellent": 0.2}
    ),
]


# ============================================================================
# HUMAN NODES (Level 2)
# ============================================================================

HUMAN_NODES = [
    NodeDefinition(
        id="human_fatigue",
        name="Worker Fatigue",
        description="Physical and mental exhaustion level",
        level=NodeLevel.HUMAN,
        category=NodeCategory.PHYSIOLOGICAL,
        states=["rested", "tired", "exhausted"],
    ),
    NodeDefinition(
        id="human_stress",
        name="Psychological Stress",
        description="Mental pressure and anxiety level",
        level=NodeLevel.HUMAN,
        category=NodeCategory.PSYCHOLOGICAL,
        states=["low", "moderate", "high"],
    ),
    NodeDefinition(
        id="human_time_pressure",
        name="Perceived Time Pressure",
        description="Worker's subjective sense of urgency and deadline stress",
        level=NodeLevel.HUMAN,
        category=NodeCategory.PSYCHOLOGICAL,
        states=["relaxed", "rushed", "frantic"],
    ),
    NodeDefinition(
        id="human_attention",
        name="Attention/Focus Level",
        description="Cognitive capacity to monitor environment and task",
        level=NodeLevel.HUMAN,
        category=NodeCategory.PSYCHOLOGICAL,
        states=["focused", "distracted", "autopilot"],
    ),
    NodeDefinition(
        id="human_risk_assessment",
        name="Risk Assessment Capability",
        description="Ability to accurately perceive and evaluate hazards",
        level=NodeLevel.HUMAN,
        category=NodeCategory.COMPETENCY,
        states=["impaired", "normal", "heightened"],
    ),
    NodeDefinition(
        id="human_error",
        name="Critical Human Error",
        description="Worker commits a safety-critical mistake",
        level=NodeLevel.HUMAN,
        category=NodeCategory.BEHAVIORAL,
        states=["no_error", "minor_error", "critical_error"],
    ),
]


# ============================================================================
# TECHNICAL NODES (Level 1)
# ============================================================================

TECHNICAL_NODES = [
    NodeDefinition(
        id="tech_hazard_present",
        name="Physical Hazard Present",
        description="Dangerous configuration or condition exists in environment",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.ENVIRONMENTAL,
        states=["safe", "hazardous"],
    ),
    NodeDefinition(
        id="tech_equipment_state",
        name="Equipment Condition",
        description="Operating state of critical equipment",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.EQUIPMENT,
        states=["normal", "degraded", "failed"],
    ),
    NodeDefinition(
        id="tech_initiating_event",
        name="Initiating Event",
        description="Random trigger event (spark, fault, etc.)",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.INCIDENT,
        states=["absent", "present"],
        default_probability={"absent": 0.99, "present": 0.01}
    ),
    NodeDefinition(
        id="tech_catastrophe",
        name="Catastrophic Outcome",
        description="Major incident occurs (fire, injury, system failure, etc.)",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.INCIDENT,
        states=["prevented", "occurred"],
    ),
]


# ============================================================================
# REGISTRY
# ============================================================================

NODE_REGISTRY: Dict[str, NodeDefinition] = {}

for node in ORGANIZATIONAL_NODES + HUMAN_NODES + TECHNICAL_NODES:
    NODE_REGISTRY[node.id] = node


def get_node(node_id: str) -> NodeDefinition:
    """Retrieve node definition by ID."""
    if node_id not in NODE_REGISTRY:
        raise KeyError(f"Node '{node_id}' not found in registry")
    return NODE_REGISTRY[node_id]


def list_nodes_by_level(level: NodeLevel) -> List[NodeDefinition]:
    """Get all nodes at a specific hierarchical level."""
    return [n for n in NODE_REGISTRY.values() if n.level == level]


def list_nodes_by_category(category: NodeCategory) -> List[NodeDefinition]:
    """Get all nodes in a specific category."""
    return [n for n in NODE_REGISTRY.values() if n.category == category]
