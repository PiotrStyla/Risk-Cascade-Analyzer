# Retro Cascade - Quick Start Guide

## Installation

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify Installation

Run the basic examples to ensure everything works:

```bash
python examples/basic_usage.py
```

You should see output demonstrating all core capabilities.

## Running the Dashboard

Launch the interactive manager dashboard:

```bash
streamlit run app.py
```

This opens a web interface at `http://localhost:8501` where you can:
- Select scenarios
- Visualize Bayesian networks
- Run Monte Carlo simulations
- Perform sensitivity analysis
- Compare intervention strategies

## Basic Programmatic Usage

### Example: Calculate Risk Given Conditions

```python
from src.scenarios.scenario_base import SCENARIO_LIBRARY
from src.core.inference_engine import CascadeInferenceEngine

# Load a scenario
scenario = SCENARIO_LIBRARY.get_scenario("warehouse_fire")
builder = scenario.build_network()

# Create inference engine
inference = CascadeInferenceEngine(builder.network)

# Define current conditions
evidence = {
    "org_output_pressure": "high",
    "org_kpi_design": "unbalanced",
    "org_resource_allocation": "insufficient"
}

# Calculate risk
risk = inference.compute_risk_score(
    evidence=evidence,
    target_node="tech_catastrophe",
    target_state="occurred"
)

print(f"Risk of catastrophe: {risk * 100:.2f}%")
```

### Example: Compare Interventions

```python
from src.core.simulation_engine import MonteCarloSimulator

simulator = MonteCarloSimulator(builder)

# Define current crisis and possible interventions
crisis = {"org_output_pressure": "high", "org_kpi_design": "unbalanced"}

interventions = [
    {"org_kpi_design": "balanced"},
    {"org_output_pressure": "medium"},
    {"org_kpi_design": "balanced", "org_output_pressure": "medium"}
]

# Compare effectiveness
results = simulator.compare_interventions(
    num_samples=10000,
    target_node="tech_catastrophe",
    target_state="occurred",
    base_evidence=crisis,
    interventions=interventions
)

# Print rankings
for name, reduction, intervention in results:
    print(f"{name}: {reduction:.1f}% risk reduction")
```

## Project Structure

```
Retro Cascade/
├── app.py                          # Dashboard entry point
├── main_thinking_chain.txt         # Conceptual foundation
├── requirements.txt                # Python dependencies
├── README.md                       # Project overview
├── QUICKSTART.md                   # This file
├── src/
│   ├── core/                       # Core Bayesian engine
│   │   ├── node_types.py          # Universal node library
│   │   ├── cpt_templates.py       # Probability templates
│   │   ├── network_builder.py     # Network construction
│   │   ├── inference_engine.py    # Forward/backward inference
│   │   └── simulation_engine.py   # Monte Carlo simulation
│   ├── scenarios/                  # Domain-specific scenarios
│   │   ├── scenario_base.py       # Base classes
│   │   └── warehouse_fire.py      # Reference implementation
│   └── ui/
│       └── manager_dashboard.py   # Streamlit dashboard
└── examples/
    └── basic_usage.py             # Usage examples
```

## Next Steps

### For Managers/Analysts
1. Launch dashboard: `streamlit run app.py`
2. Select the "warehouse_fire" scenario
3. Try different stress levels and see how risk changes
4. Compare intervention strategies
5. Export insights for decision-making

### For Developers/Researchers
1. Study `examples/basic_usage.py` for API patterns
2. Examine `src/scenarios/warehouse_fire.py` as a template
3. Create new scenarios by extending `CascadeScenario`
4. Add new node types to `src/core/node_types.py`
5. Define custom CPT templates in `src/core/cpt_templates.py`

### Creating a New Scenario

```python
from src.scenarios.scenario_base import CascadeScenario, ScenarioMetadata, SCENARIO_LIBRARY
from src.core.network_builder import CascadeNetworkBuilder

class MyScenario(CascadeScenario):
    def get_metadata(self):
        return ScenarioMetadata(
            id="my_scenario",
            name="My Risk Scenario",
            description="Description...",
            domain="my_domain",
            difficulty="intermediate"
        )
    
    def build_network(self):
        builder = CascadeNetworkBuilder()
        
        # Add nodes and edges using universal node library
        # See warehouse_fire.py for complete example
        
        builder.build()
        return builder

# Register it
SCENARIO_LIBRARY.register(MyScenario())
```

## Troubleshooting

### Import Errors
Make sure you're running from the project root directory and have installed all dependencies.

### Streamlit Issues
If the dashboard doesn't load, try:
```bash
streamlit cache clear
streamlit run app.py
```

### Performance
- Start with 1,000-10,000 simulations for testing
- Increase to 50,000+ for production analysis
- Sensitivity analysis is computationally expensive - use fewer samples initially

## Support & Extension

This is a **universal framework**. The warehouse fire scenario is just one example. The same mathematical foundation applies to:
- Healthcare (medical errors, patient safety)
- Finance (fraud cascades, operational risk)
- Software (security breaches, system failures)
- Any domain where stress-driven decisions create cascading risks

See `main_thinking_chain.txt` for the philosophical and mathematical foundation.
