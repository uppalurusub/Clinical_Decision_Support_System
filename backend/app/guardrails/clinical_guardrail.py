"""
app/guardrails/clinical_guardrail.py

Clinical Guardrail for the Clinical Decision Support system.

Purpose:
    Provide deterministic clinical-safety checks around the Agentic AI
    workflow.

Architecture:

    Input
      |
      v
    InputGuardrail
      |
      v
    Clinical Workflow / LangGraph
      |
      +------------------------------------------------+
      |                                                |
      v                                                v
    Agents                                      ClinicalGuardrail
                                                   |
                                                   v
                                             Safety Assessment
                                                   |
                                                   v
                                            Critic / Summary
                                                   |
                                                   v
                                           OutputGuardrail
                                                   |
                                                   v
                                             Human Review

Responsibilities:
    - Detect critical missing clinical information
    - Detect clinically dangerous contradictions
    - Detect potentially unsafe certainty
    - Detect unsupported definitive diagnosis
    - Detect unsafe treatment/prescription language
    - Preserve safety alerts from upstream agents
    - Validate diagnostic recommendation boundaries
    - Detect medication/allergy conflicts when represented in state
    - Detect high-risk clinical scenarios requiring review
    - Force clinician review when appropriate

Non-responsibilities:
    - Does NOT diagnose
    - Does NOT prescribe treatment
    - Does NOT determine definitive clinical risk
    - Does NOT replace validated clinical decision rules
    - Does NOT replace physician judgment

IMPORTANT:
    The rules in this file are illustrative safety checks.
    Production deployment requires validation by qualified clinical,
    safety, regulatory, and institutional stakeholders.

Logging:
    - Uses Loguru
    - Never logs PHI
    - Never logs raw clinical notes
    - Never logs laboratory values
    - Never logs medication names
    - Never logs patient symptoms
    - Logs only workflow IDs, rule IDs, statuses, and counts
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from app.config.settings import settings
logger = settings.get_logger()

from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

MAX_CLINICAL_TEXT_LENGTH = 50_000


# =============================================================================
# Enumerations
# =============================================================================


class ClinicalSeverity(str, Enum):
    """
    Clinical guardrail severity.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ClinicalGuardrailAction(str, Enum):
    """
    Action to take after clinical guardrail evaluation.
    """

    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


class ClinicalFindingType(str, Enum):
    """
    Type of clinical safety finding.
    """

    MISSING_INFORMATION = "missing_information"
    CONTRADICTION = "contradiction"
    SAFETY_SIGNAL = "safety_signal"
    MEDICATION_SAFETY = "medication_safety"
    DIAGNOSTIC_SAFETY = "diagnostic_safety"
    DATA_QUALITY = "data_quality"
    UNCERTAINTY = "uncertainty"
    OUTPUT_BOUNDARY = "output_boundary"


# =============================================================================
# Pydantic Models
# =============================================================================


class ClinicalGuardrailFinding(BaseModel):
    """
    Individual clinical guardrail finding.

    Only safe metadata should be logged.
    """

    rule_id: str = Field(
        ...,
        description="Clinical guardrail rule identifier.",
    )

    finding_type: ClinicalFindingType = Field(
        ...,
        description="Category of clinical safety finding.",
    )

    severity: ClinicalSeverity = Field(
        ...,
        description="Severity of the finding.",
    )

    action: ClinicalGuardrailAction = Field(
        ...,
        description="Required action.",
    )

    field: Optional[str] = Field(
        default=None,
        description="State field associated with the finding.",
    )

    reason: str = Field(
        ...,
        description="Safe explanation without reproducing PHI.",
    )

    requires_clinician_review: bool = Field(
        default=True,
        description="Whether clinician review is required.",
    )


class ClinicalGuardrailResult(BaseModel):
    """
    Result of clinical guardrail evaluation.
    """

    passed: bool = Field(
        ...,
        description="Whether the state passed clinical guardrails.",
    )

    action: ClinicalGuardrailAction = Field(
        ...,
        description="Overall guardrail action.",
    )

    findings: List[ClinicalGuardrailFinding] = Field(
        default_factory=list,
        description="Clinical safety findings.",
    )

    critical_findings_count: int = Field(
        default=0,
        description="Number of critical findings.",
    )

    high_findings_count: int = Field(
        default=0,
        description="Number of high-severity findings.",
    )

    requires_human_review: bool = Field(
        default=True,
        description="Whether clinician review is required.",
    )

    safety_override: bool = Field(
        default=False,
        description=(
            "Whether the guardrail prevents automatic continuation "
            "because of a safety concern."
        ),
    )


# =============================================================================
# Clinical Guardrail
# =============================================================================


class ClinicalGuardrail:
    """
    Clinical safety guardrail.

    This component evaluates the current ClinicalState and identifies
    safety conditions that require additional review.

    It should be used alongside:

        InputGuardrail
        OutputGuardrail
        ToolGuardrail
        RAGGuardrail

    It is NOT a diagnostic engine.
    """

    def __init__(
        self,
        max_clinical_text_length: int = MAX_CLINICAL_TEXT_LENGTH,
    ) -> None:

        self.max_clinical_text_length = (
            max_clinical_text_length
        )

        logger.info(
            "ClinicalGuardrail initialized | max_text_length={}",
            max_clinical_text_length,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    async def validate(
        self,
        state: Dict[str, Any],
        workflow_id: Optional[str] = None,
    ) -> ClinicalGuardrailResult:
        """
        Evaluate ClinicalState for clinical safety concerns.

        Args:
            state:
                Current clinical workflow state.

            workflow_id:
                Safe workflow correlation identifier.

        Returns:
            ClinicalGuardrailResult.
        """

        workflow_id = (
            workflow_id
            or state.get("workflow_id")
            or "unknown"
        )

        logger.info(
            "ClinicalGuardrail evaluation started | workflow_id={}",
            workflow_id,
        )

        try:

            findings: List[
                ClinicalGuardrailFinding
            ] = []

            # -------------------------------------------------------------
            # Data quality
            # -------------------------------------------------------------

            findings.extend(
                self._check_data_quality(state)
            )

            # -------------------------------------------------------------
            # Critical missing information
            # -------------------------------------------------------------

            findings.extend(
                self._check_critical_missing_information(
                    state
                )
            )

            # -------------------------------------------------------------
            # Contradictions
            # -------------------------------------------------------------

            findings.extend(
                self._check_clinical_contradictions(
                    state
                )
            )

            # -------------------------------------------------------------
            # Safety agent results
            # -------------------------------------------------------------

            findings.extend(
                self._check_safety_agent_results(
                    state
                )
            )

            # -------------------------------------------------------------
            # Medication safety
            # -------------------------------------------------------------

            findings.extend(
                self._check_medication_safety(
                    state
                )
            )

            # -------------------------------------------------------------
            # Diagnostic safety
            # -------------------------------------------------------------

            findings.extend(
                self._check_diagnostic_safety(
                    state
                )
            )

            # -------------------------------------------------------------
            # Differential safety
            # -------------------------------------------------------------

            findings.extend(
                self._check_differential_safety(
                    state
                )
            )

            # -------------------------------------------------------------
            # Diagnostic test safety
            # -------------------------------------------------------------

            findings.extend(
                self._check_test_safety(
                    state
                )
            )

            # -------------------------------------------------------------
            # Uncertainty
            # -------------------------------------------------------------

            findings.extend(
                self._check_uncertainty(
                    state
                )
            )

            # -------------------------------------------------------------
            # Definitive diagnosis language
            # -------------------------------------------------------------

            findings.extend(
                self._check_definitive_diagnosis_language(
                    state
                )
            )

            # -------------------------------------------------------------
            # Treatment language
            # -------------------------------------------------------------

            findings.extend(
                self._check_treatment_language(
                    state
                )
            )

            # -------------------------------------------------------------
            # Critic status
            # -------------------------------------------------------------

            findings.extend(
                self._check_critic_status(
                    state
                )
            )

            # -------------------------------------------------------------
            # Human review
            # -------------------------------------------------------------

            findings.extend(
                self._check_human_review(
                    state
                )
            )

            # -------------------------------------------------------------
            # Determine final action
            # -------------------------------------------------------------

            action = self._determine_action(
                findings
            )

            critical_count = sum(
                1
                for finding in findings
                if finding.severity
                == ClinicalSeverity.CRITICAL
            )

            high_count = sum(
                1
                for finding in findings
                if finding.severity
                == ClinicalSeverity.HIGH
            )

            safety_override = action == (
                ClinicalGuardrailAction.BLOCK
            )

            result = ClinicalGuardrailResult(
                passed=(
                    action
                    == ClinicalGuardrailAction.ALLOW
                ),
                action=action,
                findings=findings,
                critical_findings_count=critical_count,
                high_findings_count=high_count,
                requires_human_review=True,
                safety_override=safety_override,
            )

            logger.info(
                "ClinicalGuardrail evaluation completed | "
                "workflow_id={} | action={} | findings={} | "
                "critical={} | high={} | safety_override={}",
                workflow_id,
                action.value,
                len(findings),
                critical_count,
                high_count,
                safety_override,
            )

            return result

        except Exception:

            logger.exception(
                "ClinicalGuardrail evaluation failed | workflow_id={}",
                workflow_id,
            )

            raise

    # =========================================================================
    # Data Quality
    # =========================================================================

    def _check_data_quality(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect basic data-quality problems.

        These are not clinical diagnoses.
        """

        findings = []

        required_context_fields = [
            "clinical_context",
            "differential_diagnosis",
            "safety_flags",
        ]

        for field_name in required_context_fields:

            if field_name not in state:

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-DATA-001",
                        finding_type=(
                            ClinicalFindingType.DATA_QUALITY
                        ),
                        severity=ClinicalSeverity.HIGH,
                        action=ClinicalGuardrailAction.REVIEW,
                        field=field_name,
                        reason=(
                            "Expected clinical workflow information "
                            "is unavailable."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Critical Missing Information
    # =========================================================================

    def _check_critical_missing_information(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect critical information gaps identified by upstream agents.
        """

        findings = []

        critical_gaps = state.get(
            "critical_information_gaps"
        )

        if critical_gaps:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-MISSING-001",
                    finding_type=(
                        ClinicalFindingType.MISSING_INFORMATION
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="critical_information_gaps",
                    reason=(
                        "Critical clinical information gaps remain "
                        "unresolved."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Contradictions
    # =========================================================================

    def _check_clinical_contradictions(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect contradictions already identified by workflow agents.
        """

        findings = []

        contradictions = (
            state.get("clinical_contradictions")
            or state.get("critic_contradictions")
        )

        if contradictions:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-CONTRADICTION-001",
                    finding_type=(
                        ClinicalFindingType.CONTRADICTION
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="clinical_contradictions",
                    reason=(
                        "The workflow identified contradictory clinical "
                        "information requiring verification."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Safety Agent
    # =========================================================================

    def _check_safety_agent_results(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Preserve Safety Agent findings.
        """

        findings = []

        safety_flags = state.get(
            "safety_flags"
        ) or []

        immediate_attention = state.get(
            "immediate_attention_required"
        )

        if immediate_attention is True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-SAFETY-001",
                    finding_type=(
                        ClinicalFindingType.SAFETY_SIGNAL
                    ),
                    severity=ClinicalSeverity.CRITICAL,
                    action=ClinicalGuardrailAction.BLOCK,
                    field="immediate_attention_required",
                    reason=(
                        "The Safety Agent identified a condition "
                        "requiring immediate clinical attention."
                    ),
                )
            )

        for flag in safety_flags:

            severity = self._extract_severity(
                flag
            )

            if severity in {
                ClinicalSeverity.HIGH,
                ClinicalSeverity.CRITICAL,
            }:

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-SAFETY-002",
                        finding_type=(
                            ClinicalFindingType.SAFETY_SIGNAL
                        ),
                        severity=severity,
                        action=ClinicalGuardrailAction.REVIEW,
                        field="safety_flags",
                        reason=(
                            "The Safety Agent identified a "
                            f"{severity.value}-severity safety concern."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Medication Safety
    # =========================================================================

    def _check_medication_safety(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect medication safety concerns already identified upstream.

        This method intentionally does not independently determine
        drug interactions.
        """

        findings = []

        medication_concerns = state.get(
            "medication_safety_concerns"
        )

        if medication_concerns:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-MEDICATION-001",
                    finding_type=(
                        ClinicalFindingType.MEDICATION_SAFETY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="medication_safety_concerns",
                    reason=(
                        "Medication-related safety concerns were "
                        "identified and require clinician review."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Diagnostic Safety
    # =========================================================================

    def _check_diagnostic_safety(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect diagnostic safety concerns.
        """

        findings = []

        diagnostic_concerns = state.get(
            "diagnostic_safety_concerns"
        )

        if diagnostic_concerns:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-DIAGNOSTIC-001",
                    finding_type=(
                        ClinicalFindingType.DIAGNOSTIC_SAFETY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="diagnostic_safety_concerns",
                    reason=(
                        "Diagnostic safety concerns were identified "
                        "and require clinician review."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Differential Safety
    # =========================================================================

    def _check_differential_safety(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Validate that the differential remains reviewable and uncertain.
        """

        findings = []

        differential = state.get(
            "differential_diagnosis"
        )

        if differential:

            if not isinstance(
                differential,
                list,
            ):

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-DIFFERENTIAL-001",
                        finding_type=(
                            ClinicalFindingType.DATA_QUALITY
                        ),
                        severity=ClinicalSeverity.HIGH,
                        action=ClinicalGuardrailAction.BLOCK,
                        field="differential_diagnosis",
                        reason=(
                            "Differential diagnosis output has an "
                            "invalid structure."
                        ),
                    )
                )

        review_required = state.get(
            "differential_review_required"
        )

        if review_required is not True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-DIFFERENTIAL-002",
                    finding_type=(
                        ClinicalFindingType.OUTPUT_BOUNDARY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="differential_review_required",
                    reason=(
                        "Differential diagnosis must remain subject "
                        "to clinician review."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Diagnostic Tests
    # =========================================================================

    def _check_test_safety(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Ensure test recommendations remain suggestions rather than
        autonomous orders.
        """

        findings = []

        tests = state.get(
            "suggested_tests"
        )

        if tests:

            if not isinstance(
                tests,
                list,
            ):

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-TEST-001",
                        finding_type=(
                            ClinicalFindingType.DIAGNOSTIC_SAFETY
                        ),
                        severity=ClinicalSeverity.HIGH,
                        action=ClinicalGuardrailAction.BLOCK,
                        field="suggested_tests",
                        reason=(
                            "Diagnostic test recommendations have "
                            "an invalid structure."
                        ),
                    )
                )

        review_required = state.get(
            "diagnostic_test_review_required"
        )

        if review_required is not True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-TEST-002",
                    finding_type=(
                        ClinicalFindingType.OUTPUT_BOUNDARY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="diagnostic_test_review_required",
                    reason=(
                        "Diagnostic test recommendations must "
                        "remain subject to clinician review."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Uncertainty
    # =========================================================================

    def _check_uncertainty(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Verify that diagnostic uncertainty has not been lost.
        """

        findings = []

        uncertainties = state.get(
            "diagnostic_uncertainties"
        )

        information_gaps = state.get(
            "information_gaps"
        )

        differential = state.get(
            "differential_diagnosis"
        )

        if differential and not uncertainties and not information_gaps:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-UNCERTAINTY-001",
                    finding_type=(
                        ClinicalFindingType.UNCERTAINTY
                    ),
                    severity=ClinicalSeverity.MEDIUM,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="diagnostic_uncertainties",
                    reason=(
                        "A differential exists without explicitly "
                        "represented uncertainty or information gaps."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Definitive Diagnosis
    # =========================================================================

    def _check_definitive_diagnosis_language(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect potentially unsafe definitive diagnosis language.
        """

        findings = []

        text = self._extract_text(
            state
        )

        patterns = [
            r"\bdefinitive\s+diagnosis\b",
            r"\bdiagnosis\s+is\s+confirmed\b",
            r"\bdefinitely\s+has\b",
            r"\bcertainly\s+has\b",
            r"\bwithout\s+doubt\b",
            r"\b100\s*%\s+certain\b",
            r"\bguaranteed\s+diagnosis\b",
        ]

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-OUTPUT-001",
                        finding_type=(
                            ClinicalFindingType.OUTPUT_BOUNDARY
                        ),
                        severity=ClinicalSeverity.HIGH,
                        action=ClinicalGuardrailAction.REVIEW,
                        field="clinical_output",
                        reason=(
                            "Generated clinical content contains "
                            "potentially excessive diagnostic certainty."
                        ),
                    )
                )

                break

        return findings

    # =========================================================================
    # Treatment Language
    # =========================================================================

    def _check_treatment_language(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Detect potentially autonomous treatment instructions.
        """

        findings = []

        text = self._extract_text(
            state
        )

        patterns = [
            r"\bprescribe\b",
            r"\bstart\s+\w+\s+\d+\s*(mg|mcg|g|ml)\b",
            r"\bstop\s+\w+\s+immediately\b",
            r"\bincrease\s+the\s+dose\b",
            r"\bdecrease\s+the\s+dose\b",
            r"\bpatient\s+should\s+take\b",
            r"\btake\s+\w+\s+\d+\s*(mg|mcg|g|ml)\b",
        ]

        for pattern in patterns:

            if re.search(
                pattern,
                text,
                flags=re.IGNORECASE,
            ):

                findings.append(
                    ClinicalGuardrailFinding(
                        rule_id="CLINICAL-OUTPUT-002",
                        finding_type=(
                            ClinicalFindingType.OUTPUT_BOUNDARY
                        ),
                        severity=ClinicalSeverity.HIGH,
                        action=ClinicalGuardrailAction.REVIEW,
                        field="clinical_output",
                        reason=(
                            "Generated content may contain "
                            "prescriptive treatment language."
                        ),
                    )
                )

                break

        return findings

    # =========================================================================
    # Critic
    # =========================================================================

    def _check_critic_status(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Validate Critic Agent status.
        """

        findings = []

        hallucination_detected = state.get(
            "hallucination_detected"
        )

        critic_passed = state.get(
            "critic_passed"
        )

        requires_reanalysis = state.get(
            "requires_reanalysis"
        )

        if hallucination_detected is True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-CRITIC-001",
                    finding_type=(
                        ClinicalFindingType.DATA_QUALITY
                    ),
                    severity=ClinicalSeverity.CRITICAL,
                    action=ClinicalGuardrailAction.BLOCK,
                    field="hallucination_detected",
                    reason=(
                        "Potential hallucinated content was identified "
                        "by the Critic Agent."
                    ),
                )
            )

        if requires_reanalysis is True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-CRITIC-002",
                    finding_type=(
                        ClinicalFindingType.DATA_QUALITY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.BLOCK,
                    field="requires_reanalysis",
                    reason=(
                        "The workflow requires additional re-analysis "
                        "before clinical output can be finalized."
                    ),
                )
            )

        if critic_passed is False:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-CRITIC-003",
                    finding_type=(
                        ClinicalFindingType.DATA_QUALITY
                    ),
                    severity=ClinicalSeverity.HIGH,
                    action=ClinicalGuardrailAction.REVIEW,
                    field="critic_passed",
                    reason=(
                        "The Critic Agent did not approve the "
                        "workflow output."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Human Review
    # =========================================================================

    def _check_human_review(
        self,
        state: Dict[str, Any],
    ) -> List[ClinicalGuardrailFinding]:
        """
        Enforce clinician review.
        """

        findings = []

        if state.get("requires_human_review") is not True:

            findings.append(
                ClinicalGuardrailFinding(
                    rule_id="CLINICAL-HUMAN-001",
                    finding_type=(
                        ClinicalFindingType.OUTPUT_BOUNDARY
                    ),
                    severity=ClinicalSeverity.CRITICAL,
                    action=ClinicalGuardrailAction.BLOCK,
                    field="requires_human_review",
                    reason=(
                        "Clinical AI output cannot bypass "
                        "clinician review."
                    ),
                )
            )

        return findings

    # =========================================================================
    # Action
    # =========================================================================

    def _determine_action(
        self,
        findings: List[ClinicalGuardrailFinding],
    ) -> ClinicalGuardrailAction:
        """
        Determine overall action.

        Priority:

            BLOCK > REVIEW > ALLOW
        """

        if any(
            finding.action
            == ClinicalGuardrailAction.BLOCK
            for finding in findings
        ):
            return ClinicalGuardrailAction.BLOCK

        if any(
            finding.action
            == ClinicalGuardrailAction.REVIEW
            for finding in findings
        ):
            return ClinicalGuardrailAction.REVIEW

        return ClinicalGuardrailAction.ALLOW

    # =========================================================================
    # Utilities
    # =========================================================================

    @staticmethod
    def _extract_severity(
        value: Any,
    ) -> ClinicalSeverity:
        """
        Extract severity from a safety flag without logging its content.
        """

        if isinstance(value, dict):

            severity = value.get(
                "severity",
                "medium",
            )

        else:

            severity = getattr(
                value,
                "severity",
                "medium",
            )

        try:

            return ClinicalSeverity(
                str(severity).lower()
            )

        except ValueError:

            return ClinicalSeverity.MEDIUM

    @staticmethod
    def _extract_text(
        state: Dict[str, Any],
    ) -> str:
        """
        Extract textual state content for lexical checks.

        The content is used internally and is never logged.
        """

        values: List[str] = []

        fields = [
            "clinical_summary",
            "final_summary",
            "clinical_context",
            "differential_diagnosis",
            "diagnostic_reasoning",
            "diagnostic_uncertainties",
            "suggested_next_steps",
            "safety_flags",
            "safety_assessment",
            "critique",
            "limitations",
        ]

        for field_name in fields:

            value = state.get(field_name)

            if value is None:
                continue

            if isinstance(value, str):

                values.append(value)

            elif isinstance(value, list):

                for item in value:

                    if isinstance(item, str):
                        values.append(item)

                    elif isinstance(item, dict):

                        values.extend(
                            str(v)
                            for v in item.values()
                            if v is not None
                        )

                    else:

                        values.append(
                            str(item)
                        )

            elif isinstance(value, dict):

                values.extend(
                    str(v)
                    for v in value.values()
                    if v is not None
                )

        text = " ".join(values)

        return text[:MAX_CLINICAL_TEXT_LENGTH]