from ..core.network_builder import CascadeNetworkBuilder
from ..core.cpt_templates import ThresholdCPT
from ..core.node_types import NodeCategory, NodeDefinition, NodeLevel, register_nodes
from .scenario_base import CascadeScenario, ScenarioMetadata, SCENARIO_LIBRARY


_HR_NODES = [
    NodeDefinition(
        id="org_onboarding_workload",
        name="Onboarding Workload",
        description="Operational workload/volume in HR onboarding",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.PRESSURE,
        states=["low", "medium", "high"],
        default_probability={"low": 0.25, "medium": 0.55, "high": 0.20},
    ),
    NodeDefinition(
        id="org_process_discipline",
        name="Process Discipline (Onboarding)",
        description="Consistency of following onboarding steps and checks",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.20, "adequate": 0.60, "excellent": 0.20},
    ),
    NodeDefinition(
        id="org_tool_integration_quality",
        name="Tool Integration Quality",
        description="Quality of integrations/templates between ServiceNow, Umbrella, Amiqus and DocuSign",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.RESOURCE,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.25, "adequate": 0.55, "excellent": 0.20},
    ),
    NodeDefinition(
        id="org_second_pair_review",
        name="Second-Pair Review",
        description="Whether an external-send requires a second person verification",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["absent", "present"],
        default_probability={"absent": 0.70, "present": 0.30},
    ),
    NodeDefinition(
        id="org_vendor_controls",
        name="Vendor Controls",
        description="Controls and governance around external vendors (Amiqus, DocuSign)",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["weak", "medium", "strong"],
        default_probability={"weak": 0.20, "medium": 0.60, "strong": 0.20},
    ),
    NodeDefinition(
        id="human_misdelivery_error",
        name="Misdelivery Error",
        description="Accidental send to wrong umbrella/candidate/client",
        level=NodeLevel.HUMAN,
        category=NodeCategory.BEHAVIORAL,
        states=["none", "near_miss", "sent_wrong"],
    ),
    NodeDefinition(
        id="tech_security_controls",
        name="Security Controls (Email/DLP)",
        description="Effectiveness of technical controls: DLP, warnings, recipient validation, encryption",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.PROCESS,
        states=["weak", "standard", "strong"],
    ),
    NodeDefinition(
        id="tech_data_breach",
        name="Data Breach",
        description="Unauthorized disclosure of onboarding data (misdelivery or compromise)",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.INCIDENT,
        states=["prevented", "occurred"],
    ),
]


def _register_hr_nodes() -> None:
    register_nodes(_HR_NODES)


def _cpt_workload_to_stress() -> ThresholdCPT:
    return ThresholdCPT(
        [
            ({"org_onboarding_workload": "high"}, {"low": 0.10, "moderate": 0.40, "high": 0.50}),
            ({"org_onboarding_workload": "medium"}, {"low": 0.35, "moderate": 0.50, "high": 0.15}),
            ({}, {"low": 0.65, "moderate": 0.30, "high": 0.05}),
        ]
    )


def _cpt_workload_to_time_pressure() -> ThresholdCPT:
    return ThresholdCPT(
        [
            ({"org_onboarding_workload": "high"}, {"relaxed": 0.05, "rushed": 0.35, "frantic": 0.60}),
            ({"org_onboarding_workload": "medium"}, {"relaxed": 0.25, "rushed": 0.55, "frantic": 0.20}),
            ({}, {"relaxed": 0.70, "rushed": 0.25, "frantic": 0.05}),
        ]
    )

def _cpt_misdelivery_error() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {
                    "human_time_pressure": "frantic",
                    "human_attention": "autopilot",
                    "org_second_pair_review": "absent",
                },
                {"none": 0.25, "near_miss": 0.35, "sent_wrong": 0.40},
            ),
            (
                {
                    "human_time_pressure": "frantic",
                    "org_tool_integration_quality": "poor",
                },
                {"none": 0.35, "near_miss": 0.40, "sent_wrong": 0.25},
            ),
            (
                {
                    "org_second_pair_review": "present",
                },
                {"none": 0.80, "near_miss": 0.18, "sent_wrong": 0.02},
            ),
            ({}, {"none": 0.70, "near_miss": 0.25, "sent_wrong": 0.05}),
        ]
    )


def _cpt_data_breach() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "weak",
                    "org_vendor_controls": "weak",
                },
                {"prevented": 0.20, "occurred": 0.80},
            ),
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "weak",
                    "org_vendor_controls": "strong",
                },
                {"prevented": 0.40, "occurred": 0.60},
            ),
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "weak",
                },
                {"prevented": 0.30, "occurred": 0.70},
            ),
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "standard",
                    "org_vendor_controls": "weak",
                },
                {"prevented": 0.45, "occurred": 0.55},
            ),
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "standard",
                },
                {"prevented": 0.60, "occurred": 0.40},
            ),
            (
                {
                    "human_misdelivery_error": "sent_wrong",
                    "tech_security_controls": "strong",
                },
                {"prevented": 0.85, "occurred": 0.15},
            ),
            (
                {
                    "human_misdelivery_error": "near_miss",
                    "tech_security_controls": "strong",
                },
                {"prevented": 0.95, "occurred": 0.05},
            ),
            ({}, {"prevented": 0.92, "occurred": 0.08}),
        ]
    )


class HROnboardingDataBreachScenario(CascadeScenario):
    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            id="hr_onboarding_data_breach",
            name="HR Onboarding Data Breach",
            description=(
                "Large-corporation HR onboarding risk: wrong-recipient email sends "
                "(umbrella/candidate/client) can lead to a data breach. "
                "Integrations include ServiceNow notifications, Umbrella distribution, Amiqus vetting checklists, "
                "and DocuSign contracts."
            ),
            domain="hr",
            difficulty="intermediate",
            real_world_example=(
                "During onboarding, an email with onboarding details is sent from ServiceNow and distributed via an umbrella. "
                "Under workload and time pressure, the message is sent to a wrong umbrella/candidate/client, causing unauthorized disclosure. "
            ),
        )

    def build_network(self) -> CascadeNetworkBuilder:
        _register_hr_nodes()

        builder = CascadeNetworkBuilder()

        builder.add_node("org_onboarding_workload")
        builder.add_node("org_process_discipline")
        builder.add_node("org_tool_integration_quality")
        builder.add_node("org_second_pair_review")
        builder.add_node("org_vendor_controls")

        builder.add_edge("org_onboarding_workload", "human_stress", _cpt_workload_to_stress())
        builder.add_edge("org_onboarding_workload", "human_time_pressure", _cpt_workload_to_time_pressure())

        builder.add_node("human_attention")
        builder.add_edge(
            "human_stress",
            "human_attention",
            ThresholdCPT(
                [
                    ({"human_stress": "high"}, {"focused": 0.15, "distracted": 0.45, "autopilot": 0.40}),
                    ({"human_stress": "moderate"}, {"focused": 0.35, "distracted": 0.45, "autopilot": 0.20}),
                    ({}, {"focused": 0.60, "distracted": 0.30, "autopilot": 0.10}),
                ]
            ),
        )

        builder.add_edge("human_time_pressure", "human_misdelivery_error", _cpt_misdelivery_error())
        builder.add_edge("human_attention", "human_misdelivery_error", _cpt_misdelivery_error())
        builder.add_edge("org_second_pair_review", "human_misdelivery_error", _cpt_misdelivery_error())
        builder.add_edge("org_tool_integration_quality", "human_misdelivery_error", _cpt_misdelivery_error())

        builder.add_node("tech_security_controls")

        builder.add_edge("human_misdelivery_error", "tech_data_breach", _cpt_data_breach())
        builder.add_edge("tech_security_controls", "tech_data_breach", _cpt_data_breach())
        builder.add_edge("org_vendor_controls", "tech_data_breach", _cpt_data_breach())

        builder.build()
        self.builder = builder
        return builder

    def get_default_evidence(self) -> dict:
        return {
            "org_onboarding_workload": "medium",
            "org_process_discipline": "adequate",
            "org_tool_integration_quality": "adequate",
            "org_second_pair_review": "absent",
            "org_vendor_controls": "medium",
            "tech_security_controls": "standard",
        }

    def _low_stress_scenario(self) -> dict:
        return {
            "org_onboarding_workload": "low",
            "org_process_discipline": "excellent",
            "org_tool_integration_quality": "excellent",
            "org_second_pair_review": "present",
            "org_vendor_controls": "strong",
            "tech_security_controls": "strong",
        }

    def _medium_stress_scenario(self) -> dict:
        return {
            "org_onboarding_workload": "medium",
            "org_process_discipline": "adequate",
            "org_tool_integration_quality": "adequate",
            "org_second_pair_review": "absent",
            "org_vendor_controls": "medium",
            "tech_security_controls": "standard",
        }

    def _high_stress_scenario(self) -> dict:
        return {
            "org_onboarding_workload": "high",
            "org_process_discipline": "adequate",
            "org_tool_integration_quality": "poor",
            "org_second_pair_review": "absent",
            "org_vendor_controls": "medium",
            "tech_security_controls": "standard",
        }

    def _crisis_scenario(self) -> dict:
        return {
            "org_onboarding_workload": "high",
            "org_process_discipline": "poor",
            "org_tool_integration_quality": "poor",
            "org_second_pair_review": "absent",
            "org_vendor_controls": "weak",
            "tech_security_controls": "weak",
        }

    def get_intervention_options(self) -> dict:
        return {
            "Enable second-pair review for external sends": {
                "org_second_pair_review": "present"
            },
            "Improve tool integration and templates": {
                "org_tool_integration_quality": "excellent"
            },
            "Strengthen security controls (DLP/recipient warnings)": {
                "tech_security_controls": "strong"
            },
            "Improve vendor controls (Amiqus/DocuSign governance)": {
                "org_vendor_controls": "strong"
            },
            "Comprehensive hardening": {
                "org_second_pair_review": "present",
                "org_tool_integration_quality": "excellent",
                "tech_security_controls": "strong",
                "org_vendor_controls": "strong",
            },
        }

    def get_target_outcomes(self) -> list:
        return [
            ("tech_data_breach", "occurred"),
            ("human_misdelivery_error", "sent_wrong"),
        ]


SCENARIO_LIBRARY.register(HROnboardingDataBreachScenario())
