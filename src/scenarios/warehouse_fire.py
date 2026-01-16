"""
Warehouse Fire Scenario - Reference implementation from main_thinking_chain.txt

This is the concrete example used to illustrate the Bayesian cascade framework:
- Organizational pressure → Worker fatigue/stress → Human error → Hazard → Fire

This serves as the reference template for building other domain scenarios.
"""

from ..core.network_builder import CascadeNetworkBuilder
from ..core.cpt_templates import (
    cpt_pressure_to_fatigue, cpt_pressure_to_stress,
    cpt_multiple_to_error, cpt_error_to_hazard,
    cpt_hazard_initiator_to_catastrophe
)
from .scenario_base import CascadeScenario, ScenarioMetadata, SCENARIO_LIBRARY


class WarehouseFireScenario(CascadeScenario):
    """
    Manufacturing/Warehouse fire prevention scenario.
    
    Models how organizational pressure during peak season leads to:
    - Worker taking overtime → Fatigue
    - Pressure from KPIs → Psychological stress
    - Fatigue + Stress → Human error (unsafe material placement)
    - Error + Equipment failure → Fire
    """
    
    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            id="warehouse_fire",
            name="Warehouse Fire Prevention",
            description=(
                "Models cascading risks in warehouse operations leading to fire. "
                "Organizational pressure for output → worker fatigue/stress → "
                "critical error (flammable material near battery charger) → fire."
            ),
            domain="manufacturing",
            difficulty="intermediate",
            real_world_example=(
                "Holiday season rush in distribution center. Management increases "
                "performance targets. Worker, under pressure, places solvent canister "
                "near lithium-ion battery charging station to save time. Battery fails, "
                "ignites vapors."
            )
        )
    
    def build_network(self) -> CascadeNetworkBuilder:
        """Construct the warehouse fire cascade network."""
        builder = CascadeNetworkBuilder()
        
        # =====================================================================
        # LEVEL 3: ORGANIZATIONAL FACTORS (Root Causes)
        # =====================================================================
        
        # These are the primary drivers - management decisions
        builder.add_node("org_output_pressure")
        builder.add_node("org_safety_culture")
        builder.add_node("org_kpi_design")
        builder.add_node("org_resource_allocation")
        
        # =====================================================================
        # LEVEL 2: HUMAN FACTORS (Intermediate States)
        # =====================================================================
        
        # Organizational pressure → Human states
        builder.add_edge(
            "org_output_pressure", "human_fatigue",
            cpt_pressure_to_fatigue()
        )
        
        builder.add_edge(
            "org_output_pressure", "human_stress",
            cpt_pressure_to_stress()
        )
        
        builder.add_node("human_time_pressure")
        # Stress amplifies perceived time pressure
        from ..core.cpt_templates import ThresholdCPT
        builder.add_edge(
            "human_stress", "human_time_pressure",
            ThresholdCPT([
                ({"human_stress": "high"}, {"relaxed": 0.05, "rushed": 0.35, "frantic": 0.60}),
                ({"human_stress": "moderate"}, {"relaxed": 0.25, "rushed": 0.55, "frantic": 0.20}),
                ({}, {"relaxed": 0.70, "rushed": 0.25, "frantic": 0.05}),
            ])
        )
        
        # Fatigue + Stress + Time Pressure → Human Error
        builder.add_node("human_error")
        builder.add_edge("human_fatigue", "human_error", cpt_multiple_to_error())
        builder.add_edge("human_stress", "human_error", cpt_multiple_to_error())
        builder.add_edge("human_time_pressure", "human_error", cpt_multiple_to_error())
        
        # =====================================================================
        # LEVEL 1: TECHNICAL FACTORS (Observable Outcomes)
        # =====================================================================
        
        # Human error → Physical hazard (dangerous configuration)
        builder.add_node("tech_hazard_present")
        builder.add_edge("human_error", "tech_hazard_present", cpt_error_to_hazard())
        
        # Random initiating event (battery failure)
        builder.add_node("tech_initiating_event")
        
        # Hazard + Initiator → Catastrophe (fire)
        builder.add_node("tech_catastrophe")
        builder.add_edge("tech_hazard_present", "tech_catastrophe", 
                        cpt_hazard_initiator_to_catastrophe())
        builder.add_edge("tech_initiating_event", "tech_catastrophe",
                        cpt_hazard_initiator_to_catastrophe())
        
        # Build the network
        builder.build()
        self.builder = builder
        
        return builder
    
    def get_default_evidence(self) -> dict:
        """Normal operating conditions."""
        return {
            "org_output_pressure": "medium",
            "org_safety_culture": "adequate",
            "org_kpi_design": "balanced",
            "org_resource_allocation": "adequate",
        }
    
    def _low_stress_scenario(self) -> dict:
        """Low stress: Good conditions."""
        return {
            "org_output_pressure": "low",
            "org_safety_culture": "excellent",
            "org_kpi_design": "balanced",
            "org_resource_allocation": "abundant",
        }
    
    def _medium_stress_scenario(self) -> dict:
        """Medium stress: Typical busy period."""
        return {
            "org_output_pressure": "medium",
            "org_safety_culture": "adequate",
            "org_kpi_design": "unbalanced",
            "org_resource_allocation": "adequate",
        }
    
    def _high_stress_scenario(self) -> dict:
        """High stress: Peak season (holiday rush)."""
        return {
            "org_output_pressure": "high",
            "org_safety_culture": "adequate",
            "org_kpi_design": "unbalanced",
            "org_resource_allocation": "insufficient",
        }
    
    def _crisis_scenario(self) -> dict:
        """Crisis: Everything goes wrong."""
        return {
            "org_output_pressure": "high",
            "org_safety_culture": "poor",
            "org_kpi_design": "unbalanced",
            "org_resource_allocation": "insufficient",
        }
    
    def get_intervention_options(self) -> dict:
        """Manager intervention options."""
        return {
            "Improve KPI design (balance safety + output)": {
                "org_kpi_design": "balanced"
            },
            "Reduce output pressure": {
                "org_output_pressure": "medium"
            },
            "Add more staff/resources": {
                "org_resource_allocation": "adequate"
            },
            "Strengthen safety culture": {
                "org_safety_culture": "excellent"
            },
            "Comprehensive intervention (all of the above)": {
                "org_kpi_design": "balanced",
                "org_output_pressure": "medium",
                "org_resource_allocation": "abundant",
                "org_safety_culture": "excellent",
            },
        }
    
    def get_target_outcomes(self) -> list:
        """Outcomes to monitor."""
        return [
            ("tech_catastrophe", "occurred"),
            ("human_error", "critical_error"),
            ("tech_hazard_present", "hazardous"),
        ]


# Register scenario in global library
SCENARIO_LIBRARY.register(WarehouseFireScenario())
