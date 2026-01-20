"""
Monte Carlo simulation engine for cascade risk analysis.

Runs millions of simulations to understand probability distributions,
identify critical paths, and quantify intervention impact.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from collections import defaultdict, Counter
from pgmpy.sampling import BayesianModelSampling

from .network_builder import CascadeNetworkBuilder
from .inference_engine import CascadeInferenceEngine


@dataclass
class SimulationResult:
    """Results from a Monte Carlo simulation run."""
    num_simulations: int
    target_node: str
    target_state: str
    
    # Basic statistics
    base_probability: float  # P(target_state) under baseline conditions
    simulated_probability: float  # Empirical probability from simulations
    
    # Path analysis
    critical_paths: List[Tuple[Dict[str, str], float]]  # Most common paths leading to target
    path_frequencies: Dict[str, int]  # How often each path pattern occurred
    
    # Node importance
    node_importance: Dict[str, float]  # Which nodes most influence outcome
    
    # Time-to-event (if temporal modeling enabled)
    mean_time_to_event: Optional[float] = None
    time_distribution: Optional[Dict[int, float]] = None


class MonteCarloSimulator:
    """Runs Monte Carlo simulations on Bayesian cascade networks."""
    
    def __init__(self, builder: CascadeNetworkBuilder, show_progress: bool = True):
        """
        Initialize simulator with a network builder.
        
        Args:
            builder: CascadeNetworkBuilder with constructed network
        """
        self.builder = builder
        self.network = builder.network
        self.sampler = BayesianModelSampling(self.network)
        self.show_progress = show_progress
    
    def run_simulation(
        self,
        num_samples: int,
        target_node: str,
        target_state: str,
        evidence: Optional[Dict[str, str]] = None,
        track_paths: bool = True,
        show_progress: Optional[bool] = None,
        compute_base_probability: bool = True,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation to estimate risk and analyze paths.
        
        Args:
            num_samples: Number of simulation runs
            target_node: Node to monitor (e.g., "tech_catastrophe")
            target_state: State of interest (e.g., "occurred")
            evidence: Fixed evidence for all simulations (optional)
            track_paths: Whether to track and analyze causal paths
            
        Returns:
            SimulationResult with comprehensive statistics
        """
        # Generate samples
        import pandas as pd
        progress = self.show_progress if show_progress is None else show_progress

        if evidence:
            evidence_list = [(node, state) for node, state in evidence.items()]
            samples = self.sampler.rejection_sample(
                evidence=evidence_list,
                size=num_samples,
                show_progress=progress,
            )
        else:
            samples = self.sampler.forward_sample(
                size=num_samples,
                show_progress=progress,
            )

        if not isinstance(samples, pd.DataFrame):
            samples = pd.DataFrame(samples)
        
        # Calculate target state frequency
        target_occurrences = (samples[target_node] == target_state).sum()
        simulated_prob = target_occurrences / num_samples
        
        # Analyze paths if requested
        critical_paths = []
        path_frequencies = {}
        node_importance = {}
        
        if track_paths:
            critical_paths, path_frequencies = self._analyze_critical_paths(
                samples, target_node, target_state
            )
            node_importance = self._compute_node_importance(
                samples, target_node, target_state
            )
        
        if compute_base_probability:
            inference = CascadeInferenceEngine(self.network)
            base_prob = inference.compute_risk_score(evidence or {}, target_node, target_state)
        else:
            base_prob = float("nan")
        
        return SimulationResult(
            num_simulations=num_samples,
            target_node=target_node,
            target_state=target_state,
            base_probability=base_prob,
            simulated_probability=simulated_prob,
            critical_paths=critical_paths,
            path_frequencies=path_frequencies,
            node_importance=node_importance
        )
    
    def sensitivity_monte_carlo(
        self,
        num_samples: int,
        target_node: str,
        target_state: str,
        base_evidence: Dict[str, str],
        vary_nodes: List[str],
        show_progress: Optional[bool] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Monte Carlo-based sensitivity analysis: vary each node and measure impact.
        
        Args:
            num_samples: Samples per variation
            target_node: Outcome to monitor
            target_state: State of interest
            base_evidence: Baseline scenario
            vary_nodes: Nodes to vary
            
        Returns:
            Dict mapping node → state → impact on target probability
        """
        baseline_result = self.run_simulation(
            num_samples,
            target_node,
            target_state,
            base_evidence,
            track_paths=False,
            show_progress=show_progress,
            compute_base_probability=False,
        )
        baseline_prob = baseline_result.simulated_probability
        
        sensitivity = {}
        
        for node in vary_nodes:
            node_states = self._get_node_states(node)
            state_impacts = {}
            
            for state in node_states:
                # Modify evidence
                modified_evidence = base_evidence.copy()
                modified_evidence[node] = state
                
                # Run simulation
                result = self.run_simulation(
                    num_samples,
                    target_node,
                    target_state,
                    modified_evidence,
                    track_paths=False,
                    show_progress=show_progress,
                    compute_base_probability=False,
                )
                prob = result.simulated_probability
                
                # Compute impact
                impact = prob - baseline_prob
                state_impacts[state] = impact
            
            sensitivity[node] = state_impacts
        
        return sensitivity
    
    def compare_interventions(
        self,
        num_samples: int,
        target_node: str,
        target_state: str,
        base_evidence: Dict[str, str],
        interventions: List[Dict[str, str]],
        show_progress: Optional[bool] = None,
    ) -> List[Tuple[str, float, Dict[str, str]]]:
        """
        Compare effectiveness of different interventions.
        
        Args:
            num_samples: Samples per intervention
            target_node: Outcome to monitor
            target_state: State to prevent
            base_evidence: Current situation
            interventions: List of intervention scenarios (as evidence dicts)
            
        Returns:
            List of (intervention_name, risk_reduction, intervention_dict) sorted by effectiveness
        """
        baseline_result = self.run_simulation(
            num_samples,
            target_node,
            target_state,
            base_evidence,
            track_paths=False,
            show_progress=show_progress,
            compute_base_probability=False,
        )
        baseline_risk = baseline_result.simulated_probability
        
        results = []
        
        for i, intervention in enumerate(interventions):
            # Apply intervention
            modified_evidence = base_evidence.copy()
            modified_evidence.update(intervention)
            
            # Simulate
            result = self.run_simulation(
                num_samples,
                target_node,
                target_state,
                modified_evidence,
                track_paths=False,
                show_progress=show_progress,
                compute_base_probability=False,
            )
            risk = result.simulated_probability
            
            # Calculate risk reduction
            risk_reduction = baseline_risk - risk
            risk_reduction_pct = (risk_reduction / baseline_risk * 100) if baseline_risk > 0 else 0
            
            intervention_name = f"Intervention {i+1}: " + ", ".join(
                f"{k}={v}" for k, v in intervention.items()
            )
            
            results.append((intervention_name, risk_reduction_pct, intervention))
        
        # Sort by effectiveness (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results
    
    def _analyze_critical_paths(
        self,
        samples,
        target_node: str,
        target_state: str,
        top_k: int = 10
    ) -> Tuple[List[Tuple[Dict[str, str], float]], Dict[str, int]]:
        """
        Identify most common paths leading to target state.
        
        Returns:
            (critical_paths, path_frequencies)
        """
        # Filter samples where target occurred
        target_samples = samples[samples[target_node] == target_state]
        
        if len(target_samples) == 0:
            return [], {}
        
        # Count state combinations
        path_counter = Counter()
        
        for _, row in target_samples.iterrows():
            # Create path signature (tuple of all states)
            path = tuple(row.items())
            path_counter[path] += 1
        
        # Get top-k most common paths
        top_paths = path_counter.most_common(top_k)
        
        # Convert to readable format
        critical_paths = [
            (dict(path), count / len(target_samples))
            for path, count in top_paths
        ]
        
        # Aggregate path patterns
        path_frequencies = {str(path): count for path, count in top_paths}
        
        return critical_paths, path_frequencies
    
    def _compute_node_importance(
        self,
        samples,
        target_node: str,
        target_state: str
    ) -> Dict[str, float]:
        """
        Compute importance score for each node in predicting target state.
        
        Uses mutual information between each node and the target.
        """
        from scipy.stats import chi2_contingency
        
        importance = {}
        
        for node in samples.columns:
            if node == target_node:
                continue
            
            # Create contingency table
            contingency = samples.groupby([node, target_node]).size().unstack(fill_value=0)
            
            # Compute chi-squared statistic (measure of dependence)
            chi2, p_value, dof, expected = chi2_contingency(contingency)
            
            # Normalized importance score (Cramér's V)
            n = len(samples)
            min_dim = min(contingency.shape) - 1
            
            # Guard against division by zero (happens when variable has only one category)
            if min_dim > 0:
                cramers_v = np.sqrt(chi2 / (n * min_dim))
            else:
                cramers_v = 0.0
            
            importance[node] = cramers_v
        
        return importance
    
    def _get_node_states(self, node_id: str) -> List[str]:
        """Get possible states for a node."""
        from .node_types import get_node
        node_def = get_node(node_id)
        return node_def.states


class TemporalSimulator(MonteCarloSimulator):
    """
    Extended simulator that models temporal evolution of cascades.
    
    Simulates how risk accumulates over time and identifies critical time windows.
    """
    
    def run_temporal_simulation(
        self,
        num_samples: int,
        target_node: str,
        target_state: str,
        time_steps: int,
        evidence_schedule: Optional[Dict[int, Dict[str, str]]] = None
    ) -> Dict[int, SimulationResult]:
        """
        Run simulation over multiple time steps.
        
        Args:
            num_samples: Samples per time step
            target_node: Outcome to monitor
            target_state: State of interest
            time_steps: Number of time steps to simulate
            evidence_schedule: Evidence that changes at specific times {time: evidence}
            
        Returns:
            Dictionary mapping time step → SimulationResult
        """
        results = {}
        
        for t in range(time_steps):
            # Get evidence for this time step
            evidence = {}
            if evidence_schedule:
                for time_point in sorted(evidence_schedule.keys()):
                    if time_point <= t:
                        evidence.update(evidence_schedule[time_point])
            
            # Run simulation for this time step
            result = self.run_simulation(
                num_samples, target_node, target_state, evidence, track_paths=False
            )
            results[t] = result
        
        return results
