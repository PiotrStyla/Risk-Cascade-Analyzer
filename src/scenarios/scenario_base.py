"""
Base classes for scenario definitions.

Scenarios define domain-specific cascade networks using the universal node library.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..core.network_builder import CascadeNetworkBuilder


@dataclass
class ScenarioMetadata:
    """Metadata about a scenario."""
    id: str
    name: str
    description: str
    domain: str  # e.g., "manufacturing", "healthcare", "finance"
    difficulty: str  # "simple", "intermediate", "complex"
    real_world_example: Optional[str] = None


class CascadeScenario(ABC):
    """Base class for cascade risk scenarios."""
    
    def __init__(self):
        self.metadata: Optional[ScenarioMetadata] = None
        self.builder: Optional[CascadeNetworkBuilder] = None
    
    @abstractmethod
    def get_metadata(self) -> ScenarioMetadata:
        """Return scenario metadata."""
        pass
    
    @abstractmethod
    def build_network(self) -> CascadeNetworkBuilder:
        """
        Construct the Bayesian network for this scenario.
        
        Returns:
            CascadeNetworkBuilder with network structure and CPDs
        """
        pass
    
    def get_default_evidence(self) -> Dict[str, str]:
        """
        Get default evidence (typical baseline scenario).
        
        Override to provide domain-specific defaults.
        """
        return {}
    
    def get_stress_scenarios(self) -> Dict[str, Dict[str, str]]:
        """
        Get pre-defined stress scenarios for testing.
        
        Returns:
            Dictionary mapping scenario name → evidence
        """
        return {
            "baseline": self.get_default_evidence(),
            "low_stress": self._low_stress_scenario(),
            "medium_stress": self._medium_stress_scenario(),
            "high_stress": self._high_stress_scenario(),
            "crisis": self._crisis_scenario(),
        }
    
    def _low_stress_scenario(self) -> Dict[str, str]:
        """Low stress conditions."""
        return {}
    
    def _medium_stress_scenario(self) -> Dict[str, str]:
        """Medium stress conditions."""
        return {}
    
    def _high_stress_scenario(self) -> Dict[str, str]:
        """High stress conditions."""
        return {}
    
    def _crisis_scenario(self) -> Dict[str, str]:
        """Crisis conditions (maximum stress)."""
        return {}
    
    def get_intervention_options(self) -> Dict[str, Dict[str, str]]:
        """
        Get possible interventions managers can apply.
        
        Returns:
            Dictionary mapping intervention name → evidence changes
        """
        return {}
    
    def get_target_outcomes(self) -> List[tuple]:
        """
        Get list of target outcomes to monitor.
        
        Returns:
            List of (node_id, state) tuples representing catastrophic outcomes
        """
        return [("tech_catastrophe", "occurred")]


class ScenarioLibrary:
    """Registry of available scenarios."""
    
    def __init__(self):
        self._scenarios: Dict[str, CascadeScenario] = {}
    
    def register(self, scenario: CascadeScenario):
        """Register a scenario in the library."""
        metadata = scenario.get_metadata()
        self._scenarios[metadata.id] = scenario
    
    def get_scenario(self, scenario_id: str) -> CascadeScenario:
        """Retrieve a scenario by ID."""
        if scenario_id not in self._scenarios:
            raise KeyError(f"Scenario '{scenario_id}' not found in library")
        return self._scenarios[scenario_id]
    
    def list_scenarios(self, domain: Optional[str] = None) -> List[ScenarioMetadata]:
        """List all available scenarios, optionally filtered by domain."""
        scenarios = self._scenarios.values()
        if domain:
            scenarios = [s for s in scenarios if s.get_metadata().domain == domain]
        return [s.get_metadata() for s in scenarios]
    
    def list_domains(self) -> List[str]:
        """Get list of unique domains."""
        return list(set(s.get_metadata().domain for s in self._scenarios.values()))


# Global scenario library instance
SCENARIO_LIBRARY = ScenarioLibrary()
