"""
clinical_state.py

Shared state definition for the Agentic Clinical Decision Support
and Diagnosis workflow.

Architecture:

    FastAPI
       |
    Router
       |
    Service
       |
    Implementation
       |
    LangGraph
       |
    ClinicalState
       |
    ├── Intake Agent
    ├── Context Agent
    ├── Evidence / RAG Agent
    ├── Differential Agent
    ├── Missing Information Agent
    ├── Diagnostic Test Agent
    ├── Safety Agent
    ├── Critic Agent
    └── Summary Agent

Important:
    - ClinicalState contains workflow state only.
    - Agents should update only the fields they own.
    - Do not store secrets, API keys, passwords, or raw prompts here.
    - Avoid writing raw PHI to application logs.
    - Human/clinician review remains mandatory.

```

This gives you one central state contract for all agents and also prevents each agent from inventing its own state structure.

A particularly important design point is that **`ClinicalState` is the shared memory of the LangGraph workflow**, while each agent's `run()` method remains responsible for its own reasoning and `node()` method adapts that reasoning into state updates.

For your workflow, the state now flows conceptually as:

`Intake → Context → Evidence → Differential → Missing Information → Tests → Safety → Critic → Summary → Human Review`

The `mark_agent_*()` and `log_state_summary()` helpers also give you a clean foundation for your **observability/agent telemetry layer** without putting PHI into ordinary Loguru logs.

"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from app.config.settings import settings
logger = settings.get_logger()

# ============================================================================
# Type Aliases
# ============================================================================

Severity = Literal[
    "low",
    "medium",
    "high",
    "critical",
]

Priority = Literal[
    "low",
    "medium",
    "high",
    "critical",
]

ReanalysisTarget = Literal[
    "evidence",
    "differential",
    "missing_information",
    "safety",
    "summary",
]


# ============================================================================
# ClinicalState
# ============================================================================

class ClinicalState(TypedDict, total=False):
    """
    Shared state passed between LangGraph nodes.

    The state represents the complete working context of one
    clinical decision-support workflow.

    total=False is intentional because LangGraph nodes progressively
    populate the state.
    """

    # ------------------------------------------------------------------------
    # Workflow Metadata
    # ------------------------------------------------------------------------

    workflow_id: str
    patient_id: str
    created_at: str
    updated_at: str

    # Current workflow status
    workflow_status: Literal[
        "initialized",
        "running",
        "completed",
        "requires_review",
        "failed",
    ]

    # ------------------------------------------------------------------------
    # Input Clinical Data
    # ------------------------------------------------------------------------

    symptoms: Dict[str, Any]
    history: Dict[str, Any]
    vitals: Dict[str, Any]

    medications: List[Dict[str, Any]]
    allergies: List[str]

    labs: List[Dict[str, Any]]
    imaging: List[Dict[str, Any]]

    # ------------------------------------------------------------------------
    # Intake Agent
    # ------------------------------------------------------------------------

    intake_result: Dict[str, Any]

    # Normalized / extracted clinical context
    clinical_context: Dict[str, Any]

    # Information explicitly identified as missing during intake
    intake_missing_information: List[str]

    # ------------------------------------------------------------------------
    # Context Agent
    # ------------------------------------------------------------------------

    active_problems: List[Dict[str, Any]]

    risk_factors: List[Dict[str, Any]]

    abnormal_findings: List[Dict[str, Any]]

    temporal_events: List[Dict[str, Any]]

    information_gaps: List[str]

    unresolved_issues: List[str]

    # ------------------------------------------------------------------------
    # Evidence / RAG Agent
    # ------------------------------------------------------------------------

    evidence_query: str

    evidence: List[Dict[str, Any]]

    evidence_gaps: List[str]

    evidence_source_count: int

    # Useful for tracking retrieval quality
    evidence_retrieval_status: Literal[
        "not_started",
        "completed",
        "no_results",
        "partial",
        "failed",
    ]

    # ------------------------------------------------------------------------
    # Differential Diagnosis Agent
    # ------------------------------------------------------------------------

    differential_diagnosis: List[Dict[str, Any]]

    key_discriminating_findings: List[str]

    diagnostic_uncertainties: List[str]

    diagnostic_reasoning: str

    # Important:
    # This is not a clinical probability.
    differential_review_required: bool

    # ------------------------------------------------------------------------
    # Missing Information Agent
    # ------------------------------------------------------------------------

    missing_information: List[Dict[str, Any]]

    critical_information_gaps: List[str]

    clinical_contradictions: List[str]

    information_completeness_assessment: str

    missing_information_review_required: bool

    # ------------------------------------------------------------------------
    # Diagnostic Test Agent
    # ------------------------------------------------------------------------

    suggested_tests: List[Dict[str, Any]]

    diagnostic_gaps_addressed: List[str]

    unresolved_uncertainties: List[str]

    test_assessment: str

    diagnostic_test_review_required: bool

    # ------------------------------------------------------------------------
    # Safety Agent
    # ------------------------------------------------------------------------

    safety_flags: List[Dict[str, Any]]

    immediate_attention_required: bool

    data_quality_concerns: List[str]

    medication_safety_concerns: List[str]

    diagnostic_safety_concerns: List[str]

    safety_assessment: str

    safety_review_required: bool

    # ------------------------------------------------------------------------
    # Critic Agent
    # ------------------------------------------------------------------------

    critique: Dict[str, Any]

    critic_passed: bool

    critic_findings: List[Dict[str, Any]]

    unsupported_claims: List[str]

    critic_contradictions: List[str]

    critic_evidence_gaps: List[str]

    critic_safety_gaps: List[str]

    critic_diagnostic_gaps: List[str]

    hallucination_detected: bool

    requires_reanalysis: bool

    reanalysis_target: ReanalysisTarget

    critic_review_required: bool

    # ------------------------------------------------------------------------
    # Summary Agent
    # ------------------------------------------------------------------------

    final_summary: Dict[str, Any]

    clinical_summary: str

    key_findings: List[str]

    differential_summary: List[Dict[str, Any]]

    summary_information_gaps: List[str]

    suggested_next_steps: List[str]

    safety_alerts: List[Dict[str, Any]]

    evidence_citations: List[Dict[str, Any]]

    limitations: List[str]

    # ------------------------------------------------------------------------
    # Human Review
    # ------------------------------------------------------------------------

    requires_human_review: bool

    human_review_reason: str

    human_review_status: Literal[
        "not_required",
        "pending",
        "approved",
        "rejected",
        "modified",
    ]

    human_reviewer_id: Optional[str]

    human_review_timestamp: Optional[str]

    # ------------------------------------------------------------------------
    # Error / Execution Information
    # ------------------------------------------------------------------------

    current_agent: str

    completed_agents: List[str]

    failed_agents: List[str]

    execution_errors: List[Dict[str, Any]]

    # ------------------------------------------------------------------------
    # Observability Metadata
    # ------------------------------------------------------------------------

    agent_execution_count: int

    graph_iteration: int


# ============================================================================
# State Factory
# ============================================================================

def create_initial_clinical_state(
    *,
    workflow_id: str,
    patient_id: str,
    symptoms: Dict[str, Any],
    history: Optional[Dict[str, Any]] = None,
    vitals: Optional[Dict[str, Any]] = None,
    medications: Optional[List[Dict[str, Any]]] = None,
    allergies: Optional[List[str]] = None,
    labs: Optional[List[Dict[str, Any]]] = None,
    imaging: Optional[List[Dict[str, Any]]] = None,
) -> ClinicalState:
    """
    Create the initial state for a clinical decision-support workflow.

    This function centralizes state initialization so that the
    implementation layer does not have to manually initialize
    dozens of fields.

    Args:
        workflow_id:
            Unique identifier for the workflow execution.

        patient_id:
            Patient identifier. This should not be written to ordinary
            application logs if it is considered sensitive in the
            deployment environment.

        symptoms:
            Presenting symptoms / chief complaint.

        history:
            Patient history.

        vitals:
            Current vital signs.

        medications:
            Current medications.

        allergies:
            Known allergies.

        labs:
            Laboratory results.

        imaging:
            Imaging results.

    Returns:
        ClinicalState
    """

    now = datetime.now(timezone.utc).isoformat()

    logger.debug(
        "Creating initial ClinicalState | workflow_id={} | "
        "has_history={} | has_vitals={} | medication_count={} | "
        "allergy_count={} | lab_count={} | imaging_count={}",
        workflow_id,
        bool(history),
        bool(vitals),
        len(medications or []),
        len(allergies or []),
        len(labs or []),
        len(imaging or []),
    )

    state: ClinicalState = {
        # --------------------------------------------------------------------
        # Workflow metadata
        # --------------------------------------------------------------------

        "workflow_id": workflow_id,
        "patient_id": patient_id,
        "created_at": now,
        "updated_at": now,
        "workflow_status": "initialized",

        # --------------------------------------------------------------------
        # Input
        # --------------------------------------------------------------------

        "symptoms": symptoms,
        "history": history or {},
        "vitals": vitals or {},
        "medications": medications or [],
        "allergies": allergies or [],
        "labs": labs or [],
        "imaging": imaging or [],

        # --------------------------------------------------------------------
        # Intake
        # --------------------------------------------------------------------

        "intake_result": {},
        "clinical_context": {},
        "intake_missing_information": [],

        # --------------------------------------------------------------------
        # Context
        # --------------------------------------------------------------------

        "active_problems": [],
        "risk_factors": [],
        "abnormal_findings": [],
        "temporal_events": [],
        "information_gaps": [],
        "unresolved_issues": [],

        # --------------------------------------------------------------------
        # Evidence
        # --------------------------------------------------------------------

        "evidence_query": "",
        "evidence": [],
        "evidence_gaps": [],
        "evidence_source_count": 0,
        "evidence_retrieval_status": "not_started",

        # --------------------------------------------------------------------
        # Differential
        # --------------------------------------------------------------------

        "differential_diagnosis": [],
        "key_discriminating_findings": [],
        "diagnostic_uncertainties": [],
        "diagnostic_reasoning": "",
        "differential_review_required": True,

        # --------------------------------------------------------------------
        # Missing information
        # --------------------------------------------------------------------

        "missing_information": [],
        "critical_information_gaps": [],
        "clinical_contradictions": [],
        "information_completeness_assessment": "",
        "missing_information_review_required": True,

        # --------------------------------------------------------------------
        # Diagnostic tests
        # --------------------------------------------------------------------

        "suggested_tests": [],
        "diagnostic_gaps_addressed": [],
        "unresolved_uncertainties": [],
        "test_assessment": "",
        "diagnostic_test_review_required": True,

        # --------------------------------------------------------------------
        # Safety
        # --------------------------------------------------------------------

        "safety_flags": [],
        "immediate_attention_required": False,
        "data_quality_concerns": [],
        "medication_safety_concerns": [],
        "diagnostic_safety_concerns": [],
        "safety_assessment": "",
        "safety_review_required": True,

        # --------------------------------------------------------------------
        # Critic
        # --------------------------------------------------------------------

        "critique": {},
        "critic_passed": False,
        "critic_findings": [],
        "unsupported_claims": [],
        "critic_contradictions": [],
        "critic_evidence_gaps": [],
        "critic_safety_gaps": [],
        "critic_diagnostic_gaps": [],
        "hallucination_detected": False,
        "requires_reanalysis": False,
        "reanalysis_target": "differential",
        "critic_review_required": True,

        # --------------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------------

        "final_summary": {},
        "clinical_summary": "",
        "key_findings": [],
        "differential_summary": [],
        "summary_information_gaps": [],
        "suggested_next_steps": [],
        "safety_alerts": [],
        "evidence_citations": [],
        "limitations": [],

        # --------------------------------------------------------------------
        # Human review
        # --------------------------------------------------------------------

        # Clinical workflows should remain clinician-controlled.
        "requires_human_review": True,
        "human_review_reason": (
            "Clinical decision-support output requires clinician review "
            "before clinical action."
        ),
        "human_review_status": "pending",
        "human_reviewer_id": None,
        "human_review_timestamp": None,

        # --------------------------------------------------------------------
        # Execution metadata
        # --------------------------------------------------------------------

        "current_agent": "",
        "completed_agents": [],
        "failed_agents": [],
        "execution_errors": [],

        # --------------------------------------------------------------------
        # Observability
        # --------------------------------------------------------------------

        "agent_execution_count": 0,
        "graph_iteration": 0,
    }

    logger.info(
        "Initial ClinicalState created | workflow_id={} | "
        "workflow_status={} | human_review_required={}",
        workflow_id,
        state["workflow_status"],
        state["requires_human_review"],
    )

    return state


# ============================================================================
# State Update Helpers
# ============================================================================

def mark_agent_started(
    state: ClinicalState,
    agent_name: str,
) -> ClinicalState:
    """
    Mark an agent as the currently executing agent.

    This helper updates execution metadata without modifying
    clinical information.
    """

    state["current_agent"] = agent_name
    state["workflow_status"] = "running"

    state["agent_execution_count"] = (
        state.get("agent_execution_count", 0) + 1
    )

    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.debug(
        "Clinical agent started | workflow_id={} | agent={} | "
        "execution_count={}",
        state.get("workflow_id"),
        agent_name,
        state["agent_execution_count"],
    )

    return state


def mark_agent_completed(
    state: ClinicalState,
    agent_name: str,
) -> ClinicalState:
    """
    Mark an agent as successfully completed.
    """

    completed_agents = state.setdefault("completed_agents", [])

    if agent_name not in completed_agents:
        completed_agents.append(agent_name)

    state["current_agent"] = ""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.debug(
        "Clinical agent completed | workflow_id={} | agent={} | "
        "completed_agent_count={}",
        state.get("workflow_id"),
        agent_name,
        len(completed_agents),
    )

    return state


def mark_agent_failed(
    state: ClinicalState,
    agent_name: str,
    error: Exception,
) -> ClinicalState:
    """
    Mark an agent as failed.

    The exception message is stored in workflow state for internal
    execution diagnostics. Ensure production deployments do not
    expose this field directly to end users.
    """

    failed_agents = state.setdefault("failed_agents", [])

    if agent_name not in failed_agents:
        failed_agents.append(agent_name)

    execution_errors = state.setdefault("execution_errors", [])

    execution_errors.append(
        {
            "agent": agent_name,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )

    state["workflow_status"] = "failed"
    state["current_agent"] = agent_name
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.exception(
        "Clinical agent failed | workflow_id={} | agent={} | "
        "error_type={} | error={}",
        state.get("workflow_id"),
        agent_name,
        error.__class__.__name__,
        str(error),
    )

    return state


def increment_graph_iteration(
    state: ClinicalState,
) -> ClinicalState:
    """
    Increment LangGraph iteration count.

    Useful when the Critic Agent routes the workflow back to
    an earlier node for re-analysis.
    """

    current_iteration = state.get("graph_iteration", 0)

    state["graph_iteration"] = current_iteration + 1
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.debug(
        "Clinical graph iteration incremented | workflow_id={} | "
        "iteration={}",
        state.get("workflow_id"),
        state["graph_iteration"],
    )

    return state


def mark_workflow_requires_review(
    state: ClinicalState,
    reason: str,
) -> ClinicalState:
    """
    Explicitly mark the workflow as requiring human review.
    """

    state["requires_human_review"] = True
    state["human_review_status"] = "pending"
    state["human_review_reason"] = reason
    state["workflow_status"] = "requires_review"
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Clinical workflow marked for human review | workflow_id={} | "
        "reason={}",
        state.get("workflow_id"),
        reason,
    )

    return state


def mark_workflow_completed(
    state: ClinicalState,
) -> ClinicalState:
    """
    Mark workflow as completed.

    Completion does NOT mean that the clinical decision is final.
    The workflow remains subject to clinician review.
    """

    state["workflow_status"] = "requires_review"
    state["requires_human_review"] = True
    state["human_review_status"] = "pending"
    state["current_agent"] = ""
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    logger.info(
        "Clinical workflow completed | workflow_id={} | "
        "requires_human_review={} | status={}",
        state.get("workflow_id"),
        state["requires_human_review"],
        state["workflow_status"],
    )

    return state


# ============================================================================
# State Inspection Helpers
# ============================================================================

def get_state_summary(
    state: ClinicalState,
) -> Dict[str, Any]:
    """
    Return a non-PHI operational summary of the state.

    This is useful for logging and observability without logging
    the complete clinical state.
    """

    summary = {
        "workflow_id": state.get("workflow_id"),
        "workflow_status": state.get("workflow_status"),
        "current_agent": state.get("current_agent"),
        "completed_agent_count": len(
            state.get("completed_agents", [])
        ),
        "failed_agent_count": len(
            state.get("failed_agents", [])
        ),
        "evidence_source_count": state.get(
            "evidence_source_count",
            0,
        ),
        "differential_count": len(
            state.get("differential_diagnosis", [])
        ),
        "missing_information_count": len(
            state.get("missing_information", [])
        ),
        "suggested_test_count": len(
            state.get("suggested_tests", [])
        ),
        "safety_flag_count": len(
            state.get("safety_flags", [])
        ),
        "immediate_attention_required": state.get(
            "immediate_attention_required",
            False,
        ),
        "critic_passed": state.get(
            "critic_passed",
            False,
        ),
        "requires_reanalysis": state.get(
            "requires_reanalysis",
            False,
        ),
        "graph_iteration": state.get(
            "graph_iteration",
            0,
        ),
        "requires_human_review": state.get(
            "requires_human_review",
            True,
        ),
    }

    return summary


def log_state_summary(
    state: ClinicalState,
) -> None:
    """
    Log a safe operational summary of the clinical workflow.

    Deliberately avoids logging raw symptoms, labs, medications,
    diagnoses, imaging, clinical notes, prompts, or LLM responses.
    """

    summary = get_state_summary(state)

    logger.info(
        "Clinical workflow state summary | {}",
        summary,
    )
