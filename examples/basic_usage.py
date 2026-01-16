"""
Basic usage examples for Retro Cascade framework.

Demonstrates core functionality without UI.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.scenarios.scenario_base import SCENARIO_LIBRARY
from src.core.inference_engine import CascadeInferenceEngine
from src.core.simulation_engine import MonteCarloSimulator


def example_1_basic_inference():
    """Example 1: Basic forward inference - predict risk given conditions."""
    
    print("=" * 70)
    print("EXAMPLE 1: Forward Inference (Prediction)")
    print("=" * 70)
    
    # Load warehouse fire scenario
    scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
    builder = scenario.build_network()
    network = builder.network
    
    # Create inference engine
    inference = CascadeInferenceEngine(network)
    
    # Scenario: High-stress conditions (holiday rush)
    evidence = {
        "org_output_pressure": "high",
        "org_kpi_design": "unbalanced",
        "org_resource_allocation": "insufficient"
    }
    
    print("\nGiven conditions:")
    for k, v in evidence.items():
        print(f"  - {k}: {v}")
    
    # Query: What's the probability of catastrophe?
    risk = inference.compute_risk_score(
        evidence=evidence,
        target_node="tech_catastrophe",
        target_state="occurred"
    )
    
    print(f"\n🎯 Probability of fire: {risk * 100:.2f}%")
    
    # Query: What's the probability of human error?
    error_risk = inference.compute_risk_score(
        evidence=evidence,
        target_node="human_error",
        target_state="critical_error"
    )
    
    print(f"🎯 Probability of critical error: {error_risk * 100:.2f}%")
    
    print("\n" + "-" * 70)


def example_2_backward_diagnosis():
    """Example 2: Backward inference - diagnose causes given outcome."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Backward Inference (Diagnosis)")
    print("=" * 70)
    
    # Load scenario
    scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
    builder = scenario.build_network()
    
    inference = CascadeInferenceEngine(builder.network)
    
    # Observed: A fire occurred
    outcome = {"tech_catastrophe": "occurred"}
    
    print("\nObserved outcome:")
    print(f"  - tech_catastrophe: occurred")
    
    # Diagnose: What were the most likely organizational causes?
    causes = inference.diagnose_backward(
        outcome=outcome,
        candidate_causes=["org_output_pressure", "org_kpi_design", "org_safety_culture"]
    )
    
    print("\n🔍 Most likely root causes:")
    for cause, probs in causes.items():
        most_likely_state = max(probs.items(), key=lambda x: x[1])
        print(f"  - {cause}: {most_likely_state[0]} (P={most_likely_state[1]:.2f})")
    
    print("\n" + "-" * 70)


def example_3_monte_carlo_simulation():
    """Example 3: Monte Carlo simulation for robust risk estimation."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Monte Carlo Simulation")
    print("=" * 70)
    
    # Load scenario
    scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
    builder = scenario.build_network()
    
    simulator = MonteCarloSimulator(builder)
    
    # Run simulation under crisis conditions
    evidence = {
        "org_output_pressure": "high",
        "org_kpi_design": "unbalanced",
        "org_resource_allocation": "insufficient",
        "org_safety_culture": "poor"
    }
    
    print("\nRunning 10,000 simulations under crisis conditions...")
    
    result = simulator.run_simulation(
        num_samples=10000,
        target_node="tech_catastrophe",
        target_state="occurred",
        evidence=evidence,
        track_paths=True
    )
    
    print(f"\n📊 Results:")
    print(f"  - Simulated fire probability: {result.simulated_probability * 100:.2f}%")
    print(f"  - Number of fires in simulation: {int(result.simulated_probability * result.num_simulations)}/{result.num_simulations}")
    
    print(f"\n🎯 Most important factors:")
    sorted_importance = sorted(
        result.node_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )
    for node, score in sorted_importance[:5]:
        print(f"  - {node}: {score:.3f}")
    
    print("\n" + "-" * 70)


def example_4_sensitivity_analysis():
    """Example 4: Sensitivity analysis - which factors matter most?"""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Sensitivity Analysis")
    print("=" * 70)
    
    # Load scenario
    scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
    builder = scenario.build_network()
    
    simulator = MonteCarloSimulator(builder)
    
    # Baseline: Medium stress
    baseline = {
        "org_output_pressure": "medium",
        "org_kpi_design": "unbalanced",
        "org_resource_allocation": "adequate"
    }
    
    print("\nBaseline conditions:")
    for k, v in baseline.items():
        print(f"  - {k}: {v}")
    
    print("\nAnalyzing sensitivity...")
    
    sensitivity = simulator.sensitivity_monte_carlo(
        num_samples=5000,
        target_node="tech_catastrophe",
        target_state="occurred",
        base_evidence=baseline,
        vary_nodes=["org_output_pressure", "org_kpi_design", "org_resource_allocation"]
    )
    
    print("\n📊 Impact of changing each factor:")
    for node, state_impacts in sensitivity.items():
        print(f"\n  {node}:")
        for state, impact in sorted(state_impacts.items(), key=lambda x: x[1]):
            if impact < 0:
                print(f"    → {state}: reduces risk by {abs(impact)*100:.1f}%")
            elif impact > 0:
                print(f"    → {state}: increases risk by {impact*100:.1f}%")
    
    print("\n" + "-" * 70)


def example_5_intervention_comparison():
    """Example 5: Compare effectiveness of different interventions."""
    
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Intervention Comparison")
    print("=" * 70)
    
    # Load scenario
    scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
    builder = scenario.build_network()
    
    simulator = MonteCarloSimulator(builder)
    
    # Crisis situation
    crisis = {
        "org_output_pressure": "high",
        "org_kpi_design": "unbalanced",
        "org_resource_allocation": "insufficient"
    }
    
    print("\nCrisis situation:")
    for k, v in crisis.items():
        print(f"  - {k}: {v}")
    
    # Define possible interventions
    interventions = [
        {"org_output_pressure": "medium"},  # Reduce pressure
        {"org_kpi_design": "balanced"},  # Fix KPIs
        {"org_resource_allocation": "adequate"},  # Add resources
        {"org_kpi_design": "balanced", "org_resource_allocation": "adequate"},  # Combined
    ]
    
    print("\nComparing interventions...")
    
    results = simulator.compare_interventions(
        num_samples=5000,
        target_node="tech_catastrophe",
        target_state="occurred",
        base_evidence=crisis,
        interventions=interventions
    )
    
    print("\n🏆 Intervention Rankings (by risk reduction):")
    for i, (name, reduction, intervention) in enumerate(results, 1):
        print(f"\n  #{i}: {reduction:.1f}% reduction")
        for k, v in intervention.items():
            print(f"    - {k} → {v}")
    
    print("\n" + "-" * 70)


if __name__ == "__main__":
    print("\n")
    print("🔗 RETRO CASCADE - Universal Bayesian Risk Framework")
    print("Demonstrating core capabilities...")
    print("\n")
    
    example_1_basic_inference()
    example_2_backward_diagnosis()
    example_3_monte_carlo_simulation()
    example_4_sensitivity_analysis()
    example_5_intervention_comparison()
    
    print("\n" + "=" * 70)
    print("✅ All examples completed!")
    print("=" * 70)
    print("\nTo launch the interactive dashboard, run:")
    print("  streamlit run app.py")
    print("\n")
