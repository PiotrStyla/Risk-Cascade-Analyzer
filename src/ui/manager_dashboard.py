"""
Manager Dashboard - Streamlit UI for cascade risk analysis.

Provides interactive interface for managers to:
- Select and configure scenarios
- Visualize Bayesian networks
- Run simulations and analyze results
- Compare intervention strategies
- Export reports
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import networkx as nx

from ..core.inference_engine import CascadeInferenceEngine
from ..core.simulation_engine import MonteCarloSimulator
from ..scenarios.scenario_base import SCENARIO_LIBRARY


def render_dashboard():
    """Main dashboard rendering function."""
    
    st.set_page_config(
        page_title="Retro Cascade - Risk Analysis",
        page_icon="🔗",
        layout="wide"
    )
    
    st.title("🔗 Retro Cascade: Universal Risk Cascade Analyzer")
    st.markdown("""
    **Bayesian framework for understanding how managerial decisions under stress 
    cascade into catastrophic outcomes.**
    
    Think backward from the future to prevent disasters today.
    """)
    
    # Sidebar: Scenario selection
    with st.sidebar:
        st.header("📋 Scenario Configuration")
        
        # List available scenarios
        domains = SCENARIO_LIBRARY.list_domains()
        selected_domain = st.selectbox("Domain", ["All"] + domains)
        
        if selected_domain == "All":
            scenarios = SCENARIO_LIBRARY.list_scenarios()
        else:
            scenarios = SCENARIO_LIBRARY.list_scenarios(domain=selected_domain)
        
        scenario_names = {s.name: s.id for s in scenarios}
        selected_name = st.selectbox("Scenario", list(scenario_names.keys()))
        scenario_id = scenario_names[selected_name]
        
        # Load scenario
        scenario = SCENARIO_LIBRARY.get_scenario(scenario_id)
        metadata = scenario.get_metadata()
        
        st.info(f"**{metadata.name}**\n\n{metadata.description}")
        
        if metadata.real_world_example:
            with st.expander("Real-world example"):
                st.write(metadata.real_world_example)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🌐 Network Visualization",
        "🎲 Monte Carlo Simulation", 
        "📊 Sensitivity Analysis",
        "🎯 Intervention Comparison"
    ])
    
    # Build network for selected scenario
    builder = scenario.build_network()
    network = builder.network
    
    with tab1:
        render_network_tab(scenario, builder)
    
    with tab2:
        render_simulation_tab(scenario, builder)
    
    with tab3:
        render_sensitivity_tab(scenario, builder)
    
    with tab4:
        render_intervention_tab(scenario, builder)


def render_network_tab(scenario, builder):
    """Visualize the Bayesian network structure."""
    st.header("Network Structure")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cascade Network Graph")
        fig = visualize_network(builder)
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("Network Statistics")
        summary = builder.get_network_summary()
        
        st.metric("Total Nodes", summary["num_nodes"])
        st.metric("Total Edges", summary["num_edges"])
        
        st.write("**Nodes by Level:**")
        st.write(f"- 🏢 Organizational: {summary['nodes_by_level']['organizational']}")
        st.write(f"- 👤 Human: {summary['nodes_by_level']['human']}")
        st.write(f"- ⚙️ Technical: {summary['nodes_by_level']['technical']}")
        
        st.subheader("Node Registry")
        nodes_df = pd.DataFrame([
            {
                "Node": n.name,
                "Level": n.level.value,
                "Category": n.category.value,
                "States": len(n.states)
            }
            for n in builder.nodes.values()
        ])
        st.dataframe(nodes_df, width='stretch')


def render_simulation_tab(scenario, builder):
    """Run Monte Carlo simulations."""
    st.header("Monte Carlo Risk Simulation")
    
    st.markdown("""
    Run thousands of simulations to estimate:
    - Probability of catastrophic outcomes
    - Most likely causal paths
    - Node importance rankings
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        num_samples = st.slider(
            "Number of simulations",
            min_value=1000,
            max_value=100000,
            value=10000,
            step=1000
        )
        
        st.subheader("📍 Current Situation")
        stress_scenarios = scenario.get_stress_scenarios()
        scenario_name = st.selectbox("Scenario preset", list(stress_scenarios.keys()))
        current_evidence = stress_scenarios[scenario_name]
        
        st.json(current_evidence)
        
        target_outcomes = scenario.get_target_outcomes()
        target_node, target_state = st.selectbox(
            "Outcome to monitor",
            target_outcomes,
            format_func=lambda x: f"{x[0]} = {x[1]}"
        )
        
        run_button = st.button("🚀 Run Simulation", type="primary")
    
    with col2:
        if run_button:
            with st.spinner("Running simulations..."):
                simulator = MonteCarloSimulator(builder)
                result = simulator.run_simulation(
                    num_samples=num_samples,
                    target_node=target_node,
                    target_state=target_state,
                    evidence=current_evidence,
                    track_paths=True
                )
            
            st.subheader("📈 Results")
            
            # Risk metrics
            risk_col1, risk_col2, risk_col3 = st.columns(3)
            
            with risk_col1:
                st.metric(
                    "Baseline Probability",
                    f"{result.base_probability * 100:.2f}%"
                )
            
            with risk_col2:
                st.metric(
                    "Simulated Probability",
                    f"{result.simulated_probability * 100:.2f}%",
                    delta=f"{(result.simulated_probability - result.base_probability) * 100:.2f}%"
                )
            
            with risk_col3:
                risk_level = "🟢 LOW" if result.simulated_probability < 0.1 else \
                            "🟡 MEDIUM" if result.simulated_probability < 0.3 else \
                            "🔴 HIGH"
                st.metric("Risk Level", risk_level)
            
            # Node importance
            st.subheader("🎯 Node Importance Rankings")
            importance_df = pd.DataFrame([
                {"Node": node, "Importance": score}
                for node, score in sorted(
                    result.node_importance.items(),
                    key=lambda x: x[1],
                    reverse=True
                )
            ])
            
            fig = px.bar(
                importance_df,
                x="Importance",
                y="Node",
                orientation='h',
                title="Which factors most influence the outcome?"
            )
            st.plotly_chart(fig, width='stretch')
            
            # Critical paths
            if result.critical_paths:
                st.subheader("🛤️ Most Common Paths to Catastrophe")
                for i, (path, freq) in enumerate(result.critical_paths[:5], 1):
                    with st.expander(f"Path {i} (occurs {freq*100:.1f}% of the time)"):
                        st.json(path)


def render_sensitivity_tab(scenario, builder):
    """Perform sensitivity analysis."""
    st.header("Sensitivity Analysis")
    
    st.markdown("""
    **Question:** Which organizational factors have the biggest impact on risk?
    
    This analysis varies each factor and measures how much the catastrophe probability changes.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        num_samples = st.slider(
            "Samples per variation",
            min_value=1000,
            max_value=50000,
            value=5000,
            step=1000,
            key="sensitivity_samples"
        )
        
        st.subheader("📍 Baseline Situation")
        stress_scenarios = scenario.get_stress_scenarios()
        baseline_name = st.selectbox(
            "Starting scenario",
            list(stress_scenarios.keys()),
            key="sensitivity_baseline"
        )
        baseline_evidence = stress_scenarios[baseline_name]
        
        st.json(baseline_evidence)
        
        target_outcomes = scenario.get_target_outcomes()
        target_node, target_state = st.selectbox(
            "Outcome to prevent",
            target_outcomes,
            format_func=lambda x: f"{x[0]} = {x[1]}",
            key="sensitivity_target"
        )
        
        # Select nodes to vary
        org_nodes = [n for n in builder.nodes.keys() if n.startswith("org_")]
        vary_nodes = st.multiselect(
            "Factors to vary",
            org_nodes,
            default=org_nodes
        )
        
        analyze_button = st.button("🔬 Run Sensitivity Analysis", type="primary")
    
    with col2:
        if analyze_button and vary_nodes:
            with st.spinner("Analyzing sensitivity..."):
                simulator = MonteCarloSimulator(builder)
                sensitivity = simulator.sensitivity_monte_carlo(
                    num_samples=num_samples,
                    target_node=target_node,
                    target_state=target_state,
                    base_evidence=baseline_evidence,
                    vary_nodes=vary_nodes
                )
            
            st.subheader("📊 Impact Analysis")
            
            # Create heatmap data
            heatmap_data = []
            for node, state_impacts in sensitivity.items():
                for state, impact in state_impacts.items():
                    heatmap_data.append({
                        "Factor": node.replace("org_", "").replace("_", " ").title(),
                        "State": state,
                        "Impact": impact * 100  # Convert to percentage
                    })
            
            df = pd.DataFrame(heatmap_data)
            
            if df.empty or len(heatmap_data) == 0:
                st.warning("No sensitivity data to display. Try adjusting the baseline scenario or selected factors.")
            else:
                pivot = df.pivot(index="Factor", columns="State", values="Impact")
                
                fig = px.imshow(
                    pivot,
                    labels=dict(x="State", y="Factor", color="Risk Change (%)"),
                    title="How each factor change affects catastrophe risk",
                    color_continuous_scale="RdYlGn_r",
                    aspect="auto"
                )
                st.plotly_chart(fig, width='stretch')
            
            # Recommendations
            st.subheader("💡 Key Insights")
            
            # Find most impactful positive changes
            required_cols = {"Factor", "State", "Impact"}
            if df.empty or not required_cols.issubset(set(df.columns)):
                st.info("Run the analysis to see insights.")
            else:
                positive_impacts = df[df["Impact"] < -1].sort_values("Impact")
                
                if not positive_impacts.empty:
                    st.success("**Most Effective Risk Reductions:**")
                    for _, row in positive_impacts.head(3).iterrows():
                        st.write(
                            f"- Change **{row['Factor']}** to **{row['State']}**: "
                            f"reduces risk by **{abs(row['Impact']):.1f}%**"
                        )


def render_intervention_tab(scenario, builder):
    """Compare intervention strategies."""
    st.header("Intervention Strategy Comparison")
    
    st.markdown("""
    **Question:** Which managerial interventions are most cost-effective?
    
    Compare different actions you can take and see their impact on preventing catastrophe.
    """)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("⚙️ Configuration")
        
        num_samples = st.slider(
            "Samples per intervention",
            min_value=1000,
            max_value=50000,
            value=10000,
            step=1000,
            key="intervention_samples"
        )
        
        st.subheader("📍 Current Crisis")
        stress_scenarios = scenario.get_stress_scenarios()
        crisis_name = st.selectbox(
            "Current situation",
            list(stress_scenarios.keys()),
            index=len(stress_scenarios) - 1,  # Default to worst
            key="intervention_crisis"
        )
        crisis_evidence = stress_scenarios[crisis_name]
        
        st.json(crisis_evidence)
        
        target_outcomes = scenario.get_target_outcomes()
        target_node, target_state = st.selectbox(
            "Outcome to prevent",
            target_outcomes,
            format_func=lambda x: f"{x[0]} = {x[1]}",
            key="intervention_target"
        )
        
        st.subheader("🎯 Available Interventions")
        interventions = scenario.get_intervention_options()
        selected_interventions = st.multiselect(
            "Select interventions to compare",
            list(interventions.keys()),
            default=list(interventions.keys())[:3]
        )
        
        compare_button = st.button("⚖️ Compare Interventions", type="primary")
    
    with col2:
        if compare_button and selected_interventions:
            with st.spinner("Comparing interventions..."):
                simulator = MonteCarloSimulator(builder)
                
                intervention_dicts = [interventions[name] for name in selected_interventions]
                
                results = simulator.compare_interventions(
                    num_samples=num_samples,
                    target_node=target_node,
                    target_state=target_state,
                    base_evidence=crisis_evidence,
                    interventions=intervention_dicts
                )
            
            st.subheader("📊 Intervention Rankings")
            
            # Create ranking visualization
            ranking_data = []
            for name, reduction_pct, intervention in results:
                # Extract intervention name (remove prefix)
                clean_name = name.split(": ")[1] if ": " in name else name
                ranking_data.append({
                    "Intervention": clean_name,
                    "Risk Reduction (%)": reduction_pct
                })
            
            df = pd.DataFrame(ranking_data)
            
            fig = px.bar(
                df,
                x="Risk Reduction (%)",
                y="Intervention",
                orientation='h',
                title="Which interventions are most effective?",
                color="Risk Reduction (%)",
                color_continuous_scale="RdYlGn"
            )
            st.plotly_chart(fig, width='stretch')
            
            # Detailed results
            st.subheader("📋 Detailed Results")
            
            for i, (name, reduction_pct, intervention) in enumerate(results, 1):
                with st.expander(f"#{i}: {name.split(': ')[1] if ': ' in name else name}"):
                    st.metric("Risk Reduction", f"{reduction_pct:.1f}%")
                    st.write("**Changes:**")
                    st.json(intervention)
            
            # Recommendation
            best_name, best_reduction, best_intervention = results[0]
            
            if best_reduction > 10:
                st.success(f"""
                **💡 Recommendation:**
                
                Implementing **{best_name.split(': ')[1]}** provides the greatest risk reduction 
                ({best_reduction:.1f}%). This should be your top priority.
                """)


def visualize_network(builder) -> go.Figure:
    """Create interactive network visualization using plotly."""
    
    # Build NetworkX graph
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for node_id, node_def in builder.nodes.items():
        G.add_node(
            node_id,
            label=node_def.name,
            level=node_def.level.value,
            category=node_def.category.value
        )
    
    # Add edges
    G.add_edges_from(builder.edges)
    
    # Layout: hierarchical by level
    pos = nx.spring_layout(G, k=2, iterations=50)
    
    # Adjust y-position by level for better hierarchy
    for node in G.nodes():
        level = G.nodes[node]['level']
        pos[node] = (pos[node][0], level * 3)
    
    # Create edge traces
    edge_trace = go.Scatter(
        x=[],
        y=[],
        line=dict(width=1, color='#888'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])
    
    # Create node traces (colored by level)
    node_traces = []
    
    level_colors = {3: '#FF6B6B', 2: '#4ECDC4', 1: '#45B7D1'}
    level_names = {3: 'Organizational', 2: 'Human', 1: 'Technical'}
    
    for level in [3, 2, 1]:
        level_nodes = [n for n in G.nodes() if G.nodes[n]['level'] == level]
        
        node_trace = go.Scatter(
            x=[pos[n][0] for n in level_nodes],
            y=[pos[n][1] for n in level_nodes],
            mode='markers+text',
            name=level_names[level],
            text=[G.nodes[n]['label'] for n in level_nodes],
            textposition='top center',
            marker=dict(
                size=20,
                color=level_colors[level],
                line=dict(width=2, color='white')
            ),
            hoverinfo='text',
            hovertext=[f"{G.nodes[n]['label']}<br>Level: {level_names[level]}" for n in level_nodes]
        )
        node_traces.append(node_trace)
    
    # Create figure
    fig = go.Figure(
        data=[edge_trace] + node_traces,
        layout=go.Layout(
            showlegend=True,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig


if __name__ == "__main__":
    render_dashboard()
