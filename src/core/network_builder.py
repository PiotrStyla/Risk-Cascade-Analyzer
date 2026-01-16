"""
Bayesian Network builder for cascade risk modeling.

This module constructs pgmpy BayesianNetwork objects from node definitions and CPT templates.
"""

from typing import Dict, List, Tuple, Optional
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
import numpy as np

from .node_types import NodeDefinition, get_node
from .cpt_templates import CPTTemplate


class CascadeNetworkBuilder:
    """Builds and manages Bayesian networks for cascade risk scenarios."""
    
    def __init__(self):
        self.network = None
        self.nodes: Dict[str, NodeDefinition] = {}
        self.edges: List[Tuple[str, str]] = []
        self.cpds: Dict[str, TabularCPD] = {}
    
    def add_node(self, node_id: str):
        """Add a node to the network."""
        node_def = get_node(node_id)
        self.nodes[node_id] = node_def
    
    def add_edge(self, parent_id: str, child_id: str, cpt_template: CPTTemplate):
        """
        Add a causal edge from parent to child with specified CPT.
        
        Args:
            parent_id: ID of parent node
            child_id: ID of child node
            cpt_template: Template for generating conditional probabilities
        """
        if parent_id not in self.nodes:
            self.add_node(parent_id)
        if child_id not in self.nodes:
            self.add_node(child_id)
        
        self.edges.append((parent_id, child_id))
        
        # Store CPT template for later generation
        if child_id not in self.cpds:
            self.cpds[child_id] = {'template': cpt_template, 'parents': []}
        
        self.cpds[child_id]['parents'].append(parent_id)
    
    def build(self) -> DiscreteBayesianNetwork:
        """
        Construct the pgmpy DiscreteBayesianNetwork with all nodes, edges, and CPDs.
        
        Returns:
            Configured DiscreteBayesianNetwork ready for inference
        """
        # Create network structure with edges
        self.network = DiscreteBayesianNetwork(self.edges)
        
        # Add any isolated nodes (nodes without edges)
        for node_id in self.nodes.keys():
            if node_id not in self.network.nodes():
                self.network.add_node(node_id)
        
        # Generate and add CPDs for all nodes
        for node_id, node_def in self.nodes.items():
            if node_id in self.cpds:
                # Node has parents - generate CPD from template
                cpd = self._generate_cpd_from_template(node_id)
            else:
                # Root node - use prior probabilities
                cpd = self._generate_prior_cpd(node_id)
            
            self.network.add_cpds(cpd)
        
        # Validate network
        assert self.network.check_model(), "Network model validation failed"
        
        return self.network
    
    def _generate_prior_cpd(self, node_id: str) -> TabularCPD:
        """Generate CPD for a root node (no parents) using prior probabilities."""
        node_def = self.nodes[node_id]
        
        if node_def.default_probability:
            # Use specified priors
            probs = [node_def.default_probability[state] for state in node_def.states]
        else:
            # Uniform distribution
            probs = [1.0 / len(node_def.states)] * len(node_def.states)
        
        cpd = TabularCPD(
            variable=node_id,
            variable_card=len(node_def.states),
            values=[[p] for p in probs],
            state_names={node_id: node_def.states}
        )
        
        return cpd
    
    def _generate_cpd_from_template(self, node_id: str) -> TabularCPD:
        """Generate CPD for a child node using its template and parent information."""
        node_def = self.nodes[node_id]
        cpt_info = self.cpds[node_id]
        template = cpt_info['template']
        parent_ids = cpt_info['parents']
        
        # Get parent node definitions
        parent_defs = {pid: self.nodes[pid] for pid in parent_ids}
        parent_states = {pid: pdef.states for pid, pdef in parent_defs.items()}
        
        # Generate CPT using template
        cpt_dict = template.generate(node_def.states, parent_states)
        
        # Convert CPT dict to pgmpy TabularCPD format
        cpd = self._dict_to_tabular_cpd(
            node_id, 
            node_def.states,
            parent_ids,
            parent_states,
            cpt_dict
        )
        
        return cpd
    
    def _dict_to_tabular_cpd(
        self,
        child_id: str,
        child_states: List[str],
        parent_ids: List[str],
        parent_states: Dict[str, List[str]],
        cpt_dict: Dict
    ) -> TabularCPD:
        """
        Convert dictionary CPT to pgmpy TabularCPD format.
        
        pgmpy requires values as a 2D array where:
        - Rows = child states
        - Columns = all combinations of parent states (cartesian product)
        """
        from itertools import product
        
        # Generate all parent state combinations in order
        if parent_ids:
            parent_state_lists = [parent_states[pid] for pid in parent_ids]
            parent_combos = list(product(*parent_state_lists))
        else:
            parent_combos = [()]
        
        # Build probability matrix
        num_child_states = len(child_states)
        num_parent_combos = len(parent_combos)
        values = np.zeros((num_child_states, num_parent_combos))
        
        for col_idx, parent_combo in enumerate(parent_combos):
            prob_dist = cpt_dict[parent_combo]
            for row_idx, child_state in enumerate(child_states):
                values[row_idx, col_idx] = prob_dist[child_state]
        
        # Build state names dict
        state_names = {child_id: child_states}
        for pid in parent_ids:
            state_names[pid] = parent_states[pid]
        
        # Create TabularCPD
        cpd = TabularCPD(
            variable=child_id,
            variable_card=num_child_states,
            values=values.tolist(),
            evidence=parent_ids if parent_ids else None,
            evidence_card=[len(parent_states[pid]) for pid in parent_ids] if parent_ids else None,
            state_names=state_names
        )
        
        return cpd
    
    def get_network_summary(self) -> Dict:
        """Get summary statistics about the built network."""
        if not self.network:
            return {"status": "not_built"}
        
        return {
            "status": "built",
            "num_nodes": len(self.nodes),
            "num_edges": len(self.edges),
            "num_root_nodes": len([n for n in self.nodes if not any(e[1] == n for e in self.edges)]),
            "num_leaf_nodes": len([n for n in self.nodes if not any(e[0] == n for e in self.edges)]),
            "nodes_by_level": {
                "organizational": len([n for n in self.nodes.values() if n.level.value == 3]),
                "human": len([n for n in self.nodes.values() if n.level.value == 2]),
                "technical": len([n for n in self.nodes.values() if n.level.value == 1]),
            }
        }
