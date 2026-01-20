from ..core.network_builder import CascadeNetworkBuilder
from ..core.cpt_templates import ThresholdCPT
from ..core.node_types import NodeCategory, NodeDefinition, NodeLevel, register_nodes
from .scenario_base import CascadeScenario, ScenarioMetadata, SCENARIO_LIBRARY


_UMBRELLA_NODES = [
    NodeDefinition(
        id="org_cws_onboarding_workload",
        name="Onboarding Workload (CWS)",
        description="Operational workload/volume in AMS onboarding for CWS contractors",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.PRESSURE,
        states=["low", "medium", "high"],
        default_probability={"low": 0.25, "medium": 0.55, "high": 0.20},
    ),
    NodeDefinition(
        id="org_cws_process_discipline",
        name="Process Discipline (CWS Onboarding)",
        description="Consistency of following onboarding/vetting/contract handoff steps",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.20, "adequate": 0.60, "excellent": 0.20},
    ),
    NodeDefinition(
        id="org_cws_tool_handoff_quality",
        name="Tool Handoff Quality",
        description="Quality of handoffs/templates between ServiceNow/FieldGlass, Amiqus, Umbrella, and DocuSign",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.RESOURCE,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.25, "adequate": 0.55, "excellent": 0.20},
    ),
    NodeDefinition(
        id="org_cws_second_pair_review",
        name="Second-Pair Review",
        description="Whether external sends (Umbrella/DocuSign) require a second person verification",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["absent", "present"],
        default_probability={"absent": 0.85, "present": 0.15},
    ),
    NodeDefinition(
        id="org_cws_umbrella_variety",
        name="Umbrella Variety",
        description="Number/variety of Umbrella intermediaries used in the process (e.g., 4 different Umbrellas)",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.RESOURCE,
        states=["single", "multiple"],
        default_probability={"single": 0.05, "multiple": 0.95},
    ),
    NodeDefinition(
        id="org_cws_email_hygiene",
        name="Email Recipient Hygiene",
        description="How well recipient selection is controlled (saved recipients/autocomplete, disambiguation, attachment checks)",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["poor", "adequate", "excellent"],
        default_probability={"poor": 0.35, "adequate": 0.50, "excellent": 0.15},
    ),
    NodeDefinition(
        id="org_cws_vendor_controls",
        name="Vendor & Umbrella Controls",
        description="Governance and controls around Umbrella intermediaries and vendor tools (Amiqus/DocuSign): SLA, verification, escalation",
        level=NodeLevel.ORGANIZATIONAL,
        category=NodeCategory.POLICY,
        states=["weak", "medium", "strong"],
        default_probability={"weak": 0.25, "medium": 0.60, "strong": 0.15},
    ),
    NodeDefinition(
        id="human_cws_misdelivery_error",
        name="Misdelivery Error (CWS)",
        description="Accidental send/share to wrong umbrella, candidate, or recipient",
        level=NodeLevel.HUMAN,
        category=NodeCategory.BEHAVIORAL,
        states=["none", "near_miss", "sent_wrong"],
    ),
    NodeDefinition(
        id="human_cws_misdelivery_type",
        name="Misdelivery Type",
        description="Which kind of misdelivery happens when the process goes wrong",
        level=NodeLevel.HUMAN,
        category=NodeCategory.BEHAVIORAL,
        states=[
            "none",
            "wrong_recipient",
            "wrong_umbrella",
            "wrong_candidate_amiqus",
        ],
    ),
    NodeDefinition(
        id="tech_cws_security_controls",
        name="Security Controls (Email/DLP)",
        description="Effectiveness of technical controls: external recipient warnings, DLP, recipient validation, encryption",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.PROCESS,
        states=["weak", "standard", "strong"],
    ),
    NodeDefinition(
        id="tech_cws_data_breach",
        name="Onboarding Data Breach",
        description="Unauthorized disclosure of candidate/onboarding data due to misdelivery/sharing",
        level=NodeLevel.TECHNICAL,
        category=NodeCategory.INCIDENT,
        states=["prevented", "occurred"],
    ),
]


def _register_umbrella_nodes() -> None:
    register_nodes(_UMBRELLA_NODES)


def _cpt_workload_to_stress() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {"org_cws_onboarding_workload": "high"},
                {"low": 0.10, "moderate": 0.40, "high": 0.50},
            ),
            (
                {"org_cws_onboarding_workload": "medium"},
                {"low": 0.35, "moderate": 0.50, "high": 0.15},
            ),
            ({}, {"low": 0.65, "moderate": 0.30, "high": 0.05}),
        ]
    )


def _cpt_workload_to_time_pressure() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {"org_cws_onboarding_workload": "high"},
                {"relaxed": 0.05, "rushed": 0.35, "frantic": 0.60},
            ),
            (
                {"org_cws_onboarding_workload": "medium"},
                {"relaxed": 0.25, "rushed": 0.55, "frantic": 0.20},
            ),
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
                    "org_cws_second_pair_review": "absent",
                    "org_cws_email_hygiene": "poor",
                },
                {"none": 0.20, "near_miss": 0.35, "sent_wrong": 0.45},
            ),
            (
                {
                    "human_time_pressure": "frantic",
                    "org_cws_tool_handoff_quality": "poor",
                },
                {"none": 0.35, "near_miss": 0.40, "sent_wrong": 0.25},
            ),
            (
                {
                    "org_cws_second_pair_review": "present",
                },
                {"none": 0.82, "near_miss": 0.16, "sent_wrong": 0.02},
            ),
            ({}, {"none": 0.72, "near_miss": 0.23, "sent_wrong": 0.05}),
        ]
    )


def _cpt_misdelivery_type() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {"human_cws_misdelivery_error": "none"},
                {
                    "none": 1.00,
                    "wrong_recipient": 0.00,
                    "wrong_umbrella": 0.00,
                    "wrong_candidate_amiqus": 0.00,
                },
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "org_cws_umbrella_variety": "multiple",
                    "org_cws_email_hygiene": "poor",
                },
                {
                    "none": 0.00,
                    "wrong_recipient": 0.25,
                    "wrong_umbrella": 0.55,
                    "wrong_candidate_amiqus": 0.20,
                },
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "org_cws_process_discipline": "poor",
                    "org_cws_tool_handoff_quality": "poor",
                },
                {
                    "none": 0.00,
                    "wrong_recipient": 0.25,
                    "wrong_umbrella": 0.25,
                    "wrong_candidate_amiqus": 0.50,
                },
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "org_cws_email_hygiene": "poor",
                },
                {
                    "none": 0.00,
                    "wrong_recipient": 0.55,
                    "wrong_umbrella": 0.30,
                    "wrong_candidate_amiqus": 0.15,
                },
            ),
            (
                {"human_cws_misdelivery_error": "sent_wrong"},
                {
                    "none": 0.00,
                    "wrong_recipient": 0.35,
                    "wrong_umbrella": 0.45,
                    "wrong_candidate_amiqus": 0.20,
                },
            ),
            (
                {"human_cws_misdelivery_error": "near_miss"},
                {
                    "none": 0.05,
                    "wrong_recipient": 0.35,
                    "wrong_umbrella": 0.45,
                    "wrong_candidate_amiqus": 0.15,
                },
            ),
            (
                {},
                {
                    "none": 0.80,
                    "wrong_recipient": 0.07,
                    "wrong_umbrella": 0.08,
                    "wrong_candidate_amiqus": 0.05,
                },
            ),
        ]
    )


def _cpt_data_breach() -> ThresholdCPT:
    return ThresholdCPT(
        [
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "human_cws_misdelivery_type": "wrong_candidate_amiqus",
                    "tech_cws_security_controls": "weak",
                    "org_cws_vendor_controls": "weak",
                },
                {"prevented": 0.15, "occurred": 0.85},
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "human_cws_misdelivery_type": "wrong_umbrella",
                    "tech_cws_security_controls": "weak",
                },
                {"prevented": 0.25, "occurred": 0.75},
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "human_cws_misdelivery_type": "wrong_recipient",
                    "tech_cws_security_controls": "weak",
                },
                {"prevented": 0.30, "occurred": 0.70},
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "tech_cws_security_controls": "standard",
                },
                {"prevented": 0.60, "occurred": 0.40},
            ),
            (
                {
                    "human_cws_misdelivery_error": "sent_wrong",
                    "tech_cws_security_controls": "strong",
                },
                {"prevented": 0.85, "occurred": 0.15},
            ),
            (
                {
                    "human_cws_misdelivery_error": "near_miss",
                    "tech_cws_security_controls": "weak",
                },
                {"prevented": 0.90, "occurred": 0.10},
            ),
            (
                {
                    "human_cws_misdelivery_error": "near_miss",
                    "tech_cws_security_controls": "strong",
                },
                {"prevented": 0.98, "occurred": 0.02},
            ),
            ({}, {"prevented": 0.95, "occurred": 0.05}),
        ]
    )


class HRCWSUmbrellaMisdeliveryScenario(CascadeScenario):
    def get_metadata(self) -> ScenarioMetadata:
        return ScenarioMetadata(
            id="hr_umbrella_misdelivery_breach",
            name="HR CWS Umbrella Misdelivery & Data Breach",
            description=(
                "HR onboarding/vetting workflow risk (CWS): misdelivery of candidate data "
                "to the wrong recipient, wrong Umbrella intermediary, or wrong candidate "
                "workspace in Amiqus can lead to unauthorized disclosure."
            ),
            domain="hr",
            difficulty="intermediate",
            real_world_example=(
                "Candidate transitions from recruitment into onboarding (ServiceNow/FieldGlass work order). "
                "Onboarding contacts candidate and sends Amiqus vetting checklist. After vetting, "
                "Onboarding informs Umbrella intermediary and prepares DocuSign contracts. Under workload "
                "and time pressure, emails/attachments or recipient selection can be wrong (e.g., wrong Umbrella or "
                "wrong candidate), causing a data breach."
            ),
        )

    def build_network(self) -> CascadeNetworkBuilder:
        _register_umbrella_nodes()

        builder = CascadeNetworkBuilder()

        builder.add_node("org_cws_onboarding_workload")
        builder.add_node("org_cws_process_discipline")
        builder.add_node("org_cws_tool_handoff_quality")
        builder.add_node("org_cws_second_pair_review")
        builder.add_node("org_cws_umbrella_variety")
        builder.add_node("org_cws_email_hygiene")
        builder.add_node("org_cws_vendor_controls")

        builder.add_edge(
            "org_cws_onboarding_workload",
            "human_stress",
            _cpt_workload_to_stress(),
        )
        builder.add_edge(
            "org_cws_onboarding_workload",
            "human_time_pressure",
            _cpt_workload_to_time_pressure(),
        )

        builder.add_node("human_attention")
        builder.add_edge(
            "human_stress",
            "human_attention",
            ThresholdCPT(
                [
                    (
                        {"human_stress": "high"},
                        {"focused": 0.15, "distracted": 0.45, "autopilot": 0.40},
                    ),
                    (
                        {"human_stress": "moderate"},
                        {"focused": 0.35, "distracted": 0.45, "autopilot": 0.20},
                    ),
                    ({}, {"focused": 0.60, "distracted": 0.30, "autopilot": 0.10}),
                ]
            ),
        )

        builder.add_edge(
            "human_time_pressure",
            "human_cws_misdelivery_error",
            _cpt_misdelivery_error(),
        )
        builder.add_edge(
            "human_attention",
            "human_cws_misdelivery_error",
            _cpt_misdelivery_error(),
        )
        builder.add_edge(
            "org_cws_second_pair_review",
            "human_cws_misdelivery_error",
            _cpt_misdelivery_error(),
        )
        builder.add_edge(
            "org_cws_tool_handoff_quality",
            "human_cws_misdelivery_error",
            _cpt_misdelivery_error(),
        )
        builder.add_edge(
            "org_cws_email_hygiene",
            "human_cws_misdelivery_error",
            _cpt_misdelivery_error(),
        )

        builder.add_edge(
            "human_cws_misdelivery_error",
            "human_cws_misdelivery_type",
            _cpt_misdelivery_type(),
        )
        builder.add_edge(
            "org_cws_umbrella_variety",
            "human_cws_misdelivery_type",
            _cpt_misdelivery_type(),
        )
        builder.add_edge(
            "org_cws_process_discipline",
            "human_cws_misdelivery_type",
            _cpt_misdelivery_type(),
        )
        builder.add_edge(
            "org_cws_tool_handoff_quality",
            "human_cws_misdelivery_type",
            _cpt_misdelivery_type(),
        )
        builder.add_edge(
            "org_cws_email_hygiene",
            "human_cws_misdelivery_type",
            _cpt_misdelivery_type(),
        )

        builder.add_node("tech_cws_security_controls")

        builder.add_edge(
            "human_cws_misdelivery_error",
            "tech_cws_data_breach",
            _cpt_data_breach(),
        )
        builder.add_edge(
            "human_cws_misdelivery_type",
            "tech_cws_data_breach",
            _cpt_data_breach(),
        )
        builder.add_edge(
            "tech_cws_security_controls",
            "tech_cws_data_breach",
            _cpt_data_breach(),
        )
        builder.add_edge(
            "org_cws_vendor_controls",
            "tech_cws_data_breach",
            _cpt_data_breach(),
        )

        builder.build()
        self.builder = builder
        return builder

    def get_default_evidence(self) -> dict:
        return {
            "org_cws_onboarding_workload": "medium",
            "org_cws_process_discipline": "adequate",
            "org_cws_tool_handoff_quality": "adequate",
            "org_cws_second_pair_review": "absent",
            "org_cws_umbrella_variety": "multiple",
            "org_cws_email_hygiene": "poor",
            "org_cws_vendor_controls": "medium",
            "tech_cws_security_controls": "standard",
        }

    def _low_stress_scenario(self) -> dict:
        return {
            "org_cws_onboarding_workload": "low",
            "org_cws_process_discipline": "excellent",
            "org_cws_tool_handoff_quality": "excellent",
            "org_cws_second_pair_review": "present",
            "org_cws_umbrella_variety": "multiple",
            "org_cws_email_hygiene": "excellent",
            "org_cws_vendor_controls": "strong",
            "tech_cws_security_controls": "strong",
        }

    def _medium_stress_scenario(self) -> dict:
        return {
            "org_cws_onboarding_workload": "medium",
            "org_cws_process_discipline": "adequate",
            "org_cws_tool_handoff_quality": "adequate",
            "org_cws_second_pair_review": "absent",
            "org_cws_umbrella_variety": "multiple",
            "org_cws_email_hygiene": "poor",
            "org_cws_vendor_controls": "medium",
            "tech_cws_security_controls": "standard",
        }

    def _high_stress_scenario(self) -> dict:
        return {
            "org_cws_onboarding_workload": "high",
            "org_cws_process_discipline": "adequate",
            "org_cws_tool_handoff_quality": "poor",
            "org_cws_second_pair_review": "absent",
            "org_cws_umbrella_variety": "multiple",
            "org_cws_email_hygiene": "poor",
            "org_cws_vendor_controls": "medium",
            "tech_cws_security_controls": "standard",
        }

    def _crisis_scenario(self) -> dict:
        return {
            "org_cws_onboarding_workload": "high",
            "org_cws_process_discipline": "poor",
            "org_cws_tool_handoff_quality": "poor",
            "org_cws_second_pair_review": "absent",
            "org_cws_umbrella_variety": "multiple",
            "org_cws_email_hygiene": "poor",
            "org_cws_vendor_controls": "weak",
            "tech_cws_security_controls": "weak",
        }

    def get_intervention_options(self) -> dict:
        return {
            "Enable second-pair review for external sends": {
                "org_cws_second_pair_review": "present",
            },
            "Improve email recipient hygiene (reduce autocomplete risk)": {
                "org_cws_email_hygiene": "excellent",
            },
            "Improve handoffs/templates between tools": {
                "org_cws_tool_handoff_quality": "excellent",
            },
            "Strengthen security controls (DLP/recipient warnings)": {
                "tech_cws_security_controls": "strong",
            },
            "Improve vendor & Umbrella controls": {
                "org_cws_vendor_controls": "strong",
            },
            "Comprehensive hardening": {
                "org_cws_second_pair_review": "present",
                "org_cws_email_hygiene": "excellent",
                "org_cws_tool_handoff_quality": "excellent",
                "tech_cws_security_controls": "strong",
                "org_cws_vendor_controls": "strong",
            },
        }

    def get_target_outcomes(self) -> list:
        return [
            ("tech_cws_data_breach", "occurred"),
            ("human_cws_misdelivery_error", "sent_wrong"),
            ("human_cws_misdelivery_type", "wrong_umbrella"),
            ("human_cws_misdelivery_type", "wrong_candidate_amiqus"),
        ]


SCENARIO_LIBRARY.register(HRCWSUmbrellaMisdeliveryScenario())
