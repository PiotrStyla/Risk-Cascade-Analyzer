"""
Inference engine for Bayesian cascade networks.

Provides both forward inference (prediction) and backward inference (diagnosis).
"""

from typing import Dict, List, Optional
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
import numpy as np


class CascadeInferenceEngine:
    """Performs probabilistic inference on cascade networks."""
    
    def __init__(self, network: DiscreteBayesianNetwork):
        """
        Initialize inference engine with a built network.
        
        Args:
            network: pgmpy DiscreteBayesianNetwork with CPDs
        """
        self.network = network
        self.inference = VariableElimination(network)
    
    def predict_forward(
        self, 
        evidence: Dict[str, str], 
        query_nodes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Forward inference: Given organizational/human factors, predict technical outcomes.
        
        Args:
            evidence: Dictionary of observed node states, e.g., {"org_output_pressure": "high"}
            query_nodes: List of nodes to query (default: all leaf nodes)
            
        Returns:
            Dictionary mapping query node IDs to their probability distributions
        """
        if query_nodes is None:
            # Default: query all nodes that have no children (leaf nodes)
            all_nodes = set(self.network.nodes())
            child_nodes = set(child for _, child in self.network.edges())
            query_nodes = list(all_nodes - child_nodes)
        
        results = {}
        for node in query_nodes:
            if node in evidence:
                # Node is observed, return deterministic result
                results[node] = {evidence[node]: 1.0}
            else:
                # Perform inference
                query_result = self.inference.query(variables=[node], evidence=evidence)
                probs = query_result.values
                states = query_result.state_names[node]
                results[node] = dict(zip(states, probs))
        
        return results
    
    def diagnose_backward(
        self,
        outcome: Dict[str, str],
        candidate_causes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Backward inference: Given an outcome, infer most likely causal states.
        
        Args:
            outcome: Observed outcome, e.g., {"tech_catastrophe": "occurred"}
            candidate_causes: List of potential cause nodes to diagnose (default: all root nodes)
            
        Returns:
            Dictionary mapping cause node IDs to their posterior probability distributions
        """
        if candidate_causes is None:
            # Default: diagnose all nodes that have no parents (root nodes)
            all_nodes = set(self.network.nodes())
            parent_nodes = set(parent for parent, _ in self.network.edges())
            candidate_causes = list(parent_nodes | (all_nodes - set(child for _, child in self.network.edges())))
        
        results = {}
        for cause_node in candidate_causes:
            if cause_node in outcome:
                # Cause is observed, return deterministic result
                results[cause_node] = {outcome[cause_node]: 1.0}
            else:
                # Perform inference given outcome
                query_result = self.inference.query(variables=[cause_node], evidence=outcome)
                probs = query_result.values
                states = query_result.state_names[cause_node]
                results[cause_node] = dict(zip(states, probs))
        
        return results
    
    def compute_risk_score(
        self,
        evidence: Dict[str, str],
        target_node: str,
        target_state: str
    ) -> float:
        """
        Compute probability of a specific target state given evidence.
        
        Args:
            evidence: Observed states
            target_node: Node to query
            target_state: Specific state of interest (e.g., "occurred" for catastrophe)
            
        Returns:
            Probability (0.0 to 1.0) of target state
        """
        if target_node in evidence:
            return 1.0 if evidence[target_node] == target_state else 0.0
        
        query_result = self.inference.query(variables=[target_node], evidence=evidence)
        state_idx = list(query_result.state_names[target_node]).index(target_state)
        return float(query_result.values[state_idx])
    
    def most_probable_explanation(
        self,
        evidence: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Find the most probable explanation (MPE) - joint assignment with highest probability.
        
        Args:
            evidence: Observed states for some nodes
            
        Returns:
            Dictionary with most probable state assignment for all nodes
        """
        # Get all unobserved nodes
        unobserved = [node for node in self.network.nodes() if node not in evidence]
        
        # Use MAP inference
        from pgmpy.inference import BeliefPropagation
        bp = BeliefPropagation(self.network)
        map_result = bp.map_query(variables=unobserved, evidence=evidence)
        
        # Combine with evidence
        full_assignment = evidence.copy()
        full_assignment.update(map_result)
        
        return full_assignment
    
    def sensitivity_analysis(
        self,
        base_evidence: Dict[str, str],
        target_node: str,
        target_state: str,
        vary_nodes: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Sensitivity analysis: How much does varying each factor change the risk?
        
        Args:
            base_evidence: Baseline scenario
            target_node: Outcome node to monitor
            target_state: State of interest
            vary_nodes: Nodes to vary (default: all root nodes)
            
        Returns:
            Dictionary mapping varied node IDs to impact scores:
            {node_id: {state: delta_probability}}
        """
        if vary_nodes is None:
            # Vary all root nodes not in evidence
            all_nodes = set(self.network.nodes())
            parent_nodes = set(parent for parent, _ in self.network.edges())
            vary_nodes = list((parent_nodes | (all_nodes - set(child for _, child in self.network.edges()))) - set(base_evidence.keys()))
        
        # Compute baseline risk
        baseline_risk = self.compute_risk_score(base_evidence, target_node, target_state)
        
        sensitivity_results = {}
        
        for node in vary_nodes:
            node_def = get_node_states(self.network, node)
            state_impacts = {}
            
            for state in node_def:
                # Vary this node to this state
                modified_evidence = base_evidence.copy()
                modified_evidence[node] = state
                
                # Compute new risk
                new_risk = self.compute_risk_score(modified_evidence, target_node, target_state)
                
                # Impact = change in probability
                impact = new_risk - baseline_risk
                state_impacts[state] = impact
            
            sensitivity_results[node] = state_impacts
        
        return sensitivity_results


def get_node_states(network: DiscreteBayesianNetwork, node_id: str) -> List[str]:
    """Extract possible states for a node from the network."""
    for cpd in network.get_cpds(node_id):
        if cpd.variable == node_id:
            return cpd.state_names[node_id]
    raise ValueError(f"Node {node_id} not found in network")
