import unittest
import logging

from src.core.inference_engine import CascadeInferenceEngine
from src.core.simulation_engine import MonteCarloSimulator
from src.scenarios.scenario_base import SCENARIO_LIBRARY

logging.getLogger("pgmpy").setLevel(logging.ERROR)


class TestHRUmbrellaMisdeliveryBreachScenario(unittest.TestCase):
    def test_scenario_builds_and_runs(self):
        scenario = SCENARIO_LIBRARY.get_scenario("hr_umbrella_misdelivery_breach")
        builder = scenario.build_network()

        self.assertIsNotNone(builder.network)
        self.assertTrue(builder.network.check_model())

        evidence = scenario.get_default_evidence()

        inference = CascadeInferenceEngine(builder.network)
        breach_prob = inference.compute_risk_score(
            evidence,
            "tech_cws_data_breach",
            "occurred",
        )
        self.assertGreaterEqual(breach_prob, 0.0)
        self.assertLessEqual(breach_prob, 1.0)

        simulator = MonteCarloSimulator(builder, show_progress=False)
        result = simulator.run_simulation(
            num_samples=500,
            target_node="tech_cws_data_breach",
            target_state="occurred",
            evidence=evidence,
            track_paths=False,
        )
        self.assertGreaterEqual(result.simulated_probability, 0.0)
        self.assertLessEqual(result.simulated_probability, 1.0)


if __name__ == "__main__":
    unittest.main()
