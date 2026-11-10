"""
app/guardrails/input_guardrail.py

Input Guardrail for the Clinical Decision Support system.

Purpose:
    Validate, normalize, and safety-check incoming clinical requests
    before they enter the Agentic AI workflow.

Architecture:

    FastAPI Router
          |
          v
    ClinicalDecisionService
          |
          v
    InputGuardrail
          |
          v
    ClinicalDecisionImplementation
          |
          v
    ClinicalGraph
          |
          v
    Clinical Agents

Responsibilities:
    - Validate required input
    - Validate data types and basic ranges
    - Detect obviously malformed input
    - Detect prompt-injection style instructions
    - Detect attempts to manipulate system behavior
    - Preserve clinical uncertainty
    - Return structured validation results
    - Never diagnose
    - Never recommend treatment
    - Never make clinical decisions

Security:
    - Do not log raw clinical input
    - Do not log symptoms
    - Do not log medications
    - Do not log laboratory values
    - Do not log clinical notes
    - Do not log prompts or LLM responses
    - Only log safe metadata such as workflow ID, field names,
      validation status, and counts
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional

from app.config.settings import settings
logger = settings.get_logger()

from pydantic import BaseModel, Field


# =============================================================================
# Constants
# =============================================================================

MAX_TEXT_LENGTH = 20_000
MAX_LIST_ITEMS = 500

MIN_HEART_RATE = 20
MAX_HEART_RATE = 300

MIN_RESPIRATORY_RATE = 4
MAX_RESPIRATORY_RATE = 100

MIN_TEMPERATURE = 25.0
MAX_TEMPERATURE = 45.0

MIN_SPO2 = 0
MAX_SPO2 = 100

MIN_SYSTOLIC_BP = 40
MAX_SYSTOLIC_BP = 300

MIN_DIASTOLIC_BP = 20
MAX_DIASTOLIC_BP = 200


# =============================================================================
# Result Types
# =============================================================================


class GuardrailSeverity(str, Enum):
    """
    Severity of an input guardrail finding.
    """

    INFO = "info"
    WARNING = "warning"
    BLOCK = "block"


class GuardrailAction(str, Enum):
    """
    Action requested by the guardrail.
    """

    ALLOW = "allow"
    SANITIZE = "sanitize"
    REVIEW = "review"
    BLOCK = "block"


class GuardrailFinding(BaseModel):
    """
    Individual input guardrail finding.

    The finding intentionally contains field names and reason codes
    rather than the patient's actual clinical content.
    """

    rule_id: str = Field(
        ...,
        description="Unique guardrail rule identifier.",
    )

    field: Optional[str] = Field(
        default=None,
        description="Input field associated with the finding.",
    )

    severity: GuardrailSeverity = Field(
        ...,
        description="Severity of the finding.",
    )

    action: GuardrailAction = Field(
        ...,
        description="Recommended guardrail action.",
    )

    reason: str = Field(
        ...,
        description="Safe explanation of the validation finding.",
    )


class InputGuardrailResult(BaseModel):
    """
    Result returned by InputGuardrail.
    """

    allowed: bool = Field(
        ...,
        description="Whether processing may continue.",
    )

    action: GuardrailAction = Field(
        ...,
        description="Overall guardrail action.",
    )

    findings: List[GuardrailFinding] = Field(
        default_factory=list,
        description="Guardrail findings.",
    )

    sanitized_input: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Sanitized input when sanitization is appropriate.",
    )

    requires_human_review: bool = Field(
        default=False,
        description="Whether human review is required.",
    )

    rule_count: int = Field(
        default=0,
        description="Number of rules evaluated.",
    )


# =============================================================================
# Input Guardrail
# =============================================================================


class InputGuardrail:
    """
    Deterministic input guardrail for clinical requests.

    This guardrail does NOT determine whether a patient has a disease.

    It validates:
        - Required fields
        - Data structure
        - Basic numeric ranges
        - Text size
        - Suspicious instruction patterns
        - Potential prompt injection
        - Malformed content

    Clinical safety decisions remain outside this component.
    """

    def __init__(
        self,
        max_text_length: int = MAX_TEXT_LENGTH,
        max_list_items: int = MAX_LIST_ITEMS,
    ) -> None:

        self.max_text_length = max_text_length
        self.max_list_items = max_list_items

        logger.info(
            "InputGuardrail initialized | max_text_length={} | "
            "max_list_items={}",
            max_text_length,
            max_list_items,
        )

    # =========================================================================
    # Public API
    # =========================================================================

    async def validate(
        self,
        request: Any,
        workflow_id: Optional[str] = None,
    ) -> InputGuardrailResult:
        """
        Validate incoming clinical request.

        Args:
            request:
                Clinical request object or dictionary.

            workflow_id:
                Safe workflow correlation identifier.

        Returns:
            InputGuardrailResult
        """

        logger.info(
            "InputGuardrail validation started | workflow_id={}",
            workflow_id or "unknown",
        )

        try:

            data = self._to_dict(request)

            findings: List[GuardrailFinding] = []

            # -------------------------------------------------------------
            # Required fields
            # -------------------------------------------------------------

            findings.extend(
                self._validate_required_fields(data)
            )

            # -------------------------------------------------------------
            # Structure
            # -------------------------------------------------------------

            findings.extend(
                self._validate_structure(data)
            )

            # -------------------------------------------------------------
            # Text length
            # -------------------------------------------------------------

            findings.extend(
                self._validate_text_lengths(data)
            )

            # -------------------------------------------------------------
            # Collection sizes
            # -------------------------------------------------------------

            findings.extend(
                self._validate_collection_sizes(data)
            )

            # -------------------------------------------------------------
            # Vital sign ranges
            # -------------------------------------------------------------

            findings.extend(
                self._validate_vitals(
                    data.get("vitals")
                )
            )

            # -------------------------------------------------------------
            # Suspicious instructions / prompt injection
            # -------------------------------------------------------------

            findings.extend(
                self._detect_prompt_injection(data)
            )

            # -------------------------------------------------------------
            # Suspicious control instructions
            # -------------------------------------------------------------

            findings.extend(
                self._detect_control_manipulation(data)
            )

            # -------------------------------------------------------------
            # Determine overall action
            # -------------------------------------------------------------

            action = self._determine_action(findings)

            allowed = action != GuardrailAction.BLOCK

            requires_human_review = (
                action == GuardrailAction.REVIEW
            )

            sanitized_input = None

            if allowed:
                sanitized_input = self._sanitize_input(data)

            result = InputGuardrailResult(
                allowed=allowed,
                action=action,
                findings=findings,
                sanitized_input=sanitized_input,
                requires_human_review=requires_human_review,
                rule_count=len(findings),
            )

            logger.info(
                "InputGuardrail validation completed | workflow_id={} | "
                "allowed={} | action={} | findings={} | "
                "human_review={}",
                workflow_id or "unknown",
                allowed,
                action.value,
                len(findings),
                requires_human_review,
            )

            return result

        except Exception:

            logger.exception(
                "InputGuardrail validation failed | workflow_id={}",
                workflow_id or "unknown",
            )

            raise

    # =========================================================================
    # Required Fields
    # =========================================================================

    def _validate_required_fields(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Validate required request fields.

        Adjust the required fields if your ClinicalDecisionRequest
        schema differs.
        """

        findings = []

        patient_id = data.get("patient_id")

        if patient_id is None or not str(patient_id).strip():

            findings.append(
                GuardrailFinding(
                    rule_id="INPUT-001",
                    field="patient_id",
                    severity=GuardrailSeverity.BLOCK,
                    action=GuardrailAction.BLOCK,
                    reason="Required patient identifier is missing.",
                )
            )

        symptoms = data.get("symptoms")

        if symptoms is None or not self._has_content(symptoms):

            findings.append(
                GuardrailFinding(
                    rule_id="INPUT-002",
                    field="symptoms",
                    severity=GuardrailSeverity.BLOCK,
                    action=GuardrailAction.BLOCK,
                    reason="Required clinical symptom information is missing.",
                )
            )

        return findings

    # =========================================================================
    # Structure Validation
    # =========================================================================

    def _validate_structure(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Validate basic input structure.
        """

        findings = []

        expected_collections = [
            "symptoms",
            "history",
            "medications",
            "allergies",
            "labs",
            "imaging",
        ]

        for field_name in expected_collections:

            if field_name not in data:
                continue

            value = data[field_name]

            if value is None:
                continue

            if not isinstance(value, (str, list, dict)):

                findings.append(
                    GuardrailFinding(
                        rule_id="INPUT-003",
                        field=field_name,
                        severity=GuardrailSeverity.BLOCK,
                        action=GuardrailAction.BLOCK,
                        reason=(
                            "Input field has an unsupported data structure."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Text Validation
    # =========================================================================

    def _validate_text_lengths(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Detect excessively large text fields.
        """

        findings = []

        text_fields = [
            "symptoms",
            "history",
            "medications",
            "allergies",
            "labs",
            "imaging",
        ]

        for field_name in text_fields:

            value = data.get(field_name)

            if value is None:
                continue

            text_length = self._calculate_text_length(value)

            if text_length > self.max_text_length:

                findings.append(
                    GuardrailFinding(
                        rule_id="INPUT-004",
                        field=field_name,
                        severity=GuardrailSeverity.BLOCK,
                        action=GuardrailAction.BLOCK,
                        reason=(
                            "Input exceeds the configured maximum "
                            "text length."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Collection Validation
    # =========================================================================

    def _validate_collection_sizes(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Detect excessively large collections.
        """

        findings = []

        for field_name, value in data.items():

            if not isinstance(value, (list, tuple)):
                continue

            if len(value) > self.max_list_items:

                findings.append(
                    GuardrailFinding(
                        rule_id="INPUT-005",
                        field=field_name,
                        severity=GuardrailSeverity.BLOCK,
                        action=GuardrailAction.BLOCK,
                        reason=(
                            "Input collection exceeds the configured "
                            "maximum number of items."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Vital Validation
    # =========================================================================

    def _validate_vitals(
        self,
        vitals: Any,
    ) -> List[GuardrailFinding]:
        """
        Validate basic physiological value ranges.

        These checks detect malformed or implausible data.

        They are NOT clinical risk assessments.
        """

        if vitals is None:
            return []

        if not isinstance(vitals, dict):
            return [
                GuardrailFinding(
                    rule_id="INPUT-006",
                    field="vitals",
                    severity=GuardrailSeverity.BLOCK,
                    action=GuardrailAction.BLOCK,
                    reason="Vitals must be represented as a mapping.",
                )
            ]

        findings = []

        range_rules = {
            "heart_rate": (
                MIN_HEART_RATE,
                MAX_HEART_RATE,
            ),
            "respiratory_rate": (
                MIN_RESPIRATORY_RATE,
                MAX_RESPIRATORY_RATE,
            ),
            "temperature": (
                MIN_TEMPERATURE,
                MAX_TEMPERATURE,
            ),
            "spo2": (
                MIN_SPO2,
                MAX_SPO2,
            ),
            "systolic_bp": (
                MIN_SYSTOLIC_BP,
                MAX_SYSTOLIC_BP,
            ),
            "diastolic_bp": (
                MIN_DIASTOLIC_BP,
                MAX_DIASTOLIC_BP,
            ),
        }

        for field_name, (
            minimum,
            maximum,
        ) in range_rules.items():

            value = vitals.get(field_name)

            if value is None:
                continue

            try:
                numeric_value = float(value)

            except (TypeError, ValueError):

                findings.append(
                    GuardrailFinding(
                        rule_id="INPUT-007",
                        field=f"vitals.{field_name}",
                        severity=GuardrailSeverity.BLOCK,
                        action=GuardrailAction.BLOCK,
                        reason="Vital sign must be numeric.",
                    )
                )

                continue

            if not minimum <= numeric_value <= maximum:

                findings.append(
                    GuardrailFinding(
                        rule_id="INPUT-008",
                        field=f"vitals.{field_name}",
                        severity=GuardrailSeverity.WARNING,
                        action=GuardrailAction.REVIEW,
                        reason=(
                            "Vital sign is outside the configured "
                            "plausibility range and requires verification."
                        ),
                    )
                )

        return findings

    # =========================================================================
    # Prompt Injection Detection
    # =========================================================================

    def _detect_prompt_injection(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Detect common prompt-injection patterns.

        This is a security filter, not a clinical classifier.

        Detection should be treated as a signal rather than proof of
        malicious intent.
        """

        findings = []

        suspicious_patterns = [
            r"\bignore\s+(all|any|previous|prior)\s+instructions\b",
            r"\bdisregard\s+(all|any|previous|prior)\s+instructions\b",
            r"\bforget\s+(all|any|previous|prior)\s+instructions\b",
            r"\boverride\s+(the\s+)?system\b",
            r"\b(system|developer)\s+prompt\b",
            r"\breveal\s+(your|the)\s+(system\s+)?prompt\b",
            r"\bshow\s+(me\s+)?your\s+instructions\b",
            r"\bdo\s+not\s+follow\s+(the\s+)?instructions\b",
            r"\bexecute\s+this\s+instruction\b",
        ]

        combined_text = self._extract_text(data)

        for pattern in suspicious_patterns:

            if re.search(
                pattern,
                combined_text,
                flags=re.IGNORECASE,
            ):

                findings.append(
                    GuardrailFinding(
                        rule_id="SECURITY-001",
                        field=None,
                        severity=GuardrailSeverity.REVIEW,
                        action=GuardrailAction.REVIEW,
                        reason=(
                            "Input contains an instruction pattern that "
                            "may attempt to manipulate system behavior."
                        ),
                    )
                )

                break

        return findings

    # =========================================================================
    # Control Manipulation Detection
    # =========================================================================

    def _detect_control_manipulation(
        self,
        data: Dict[str, Any],
    ) -> List[GuardrailFinding]:
        """
        Detect attempts to force an outcome.

        Example concepts:
            - force diagnosis
            - bypass safety
            - skip review
            - suppress warnings
            - disable guardrails
        """

        findings = []

        patterns = [
            r"\bskip\s+(human|clinician)\s+review\b",
            r"\bbypass\s+(safety|guardrail|validation)\b",
            r"\bdisable\s+(safety|guardrail|validation)\b",
            r"\bhide\s+(the\s+)?warning\b",
            r"\bsuppress\s+(the\s+)?warning\b",
            r"\bforce\s+(a\s+)?diagnosis\b",
            r"\bguarantee\s+(a\s+)?diagnosis\b",
        ]

        combined_text = self._extract_text(data)

        for pattern in patterns:

            if re.search(
                pattern,
                combined_text,
                flags=re.IGNORECASE,
            ):

                findings.append(
                    GuardrailFinding(
                        rule_id="SECURITY-002",
                        field=None,
                        severity=GuardrailSeverity.BLOCK,
                        action=GuardrailAction.BLOCK,
                        reason=(
                            "Input attempts to bypass or manipulate "
                            "clinical safety controls."
                        ),
                    )
                )

                break

        return findings

    # =========================================================================
    # Sanitization
    # =========================================================================

    def _sanitize_input(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Perform conservative input sanitization.

        Important:
            Do not aggressively modify clinical content.

        Clinical meaning must not be changed by a generic security
        sanitizer.
        """

        sanitized = dict(data)

        # Normalize simple text fields.
        text_fields = [
            "symptoms",
            "history",
            "medications",
            "allergies",
            "labs",
            "imaging",
        ]

        for field_name in text_fields:

            value = sanitized.get(field_name)

            if isinstance(value, str):

                sanitized[field_name] = (
                    value.strip()
                )

        return sanitized

    # =========================================================================
    # Action Resolution
    # =========================================================================

    def _determine_action(
        self,
        findings: List[GuardrailFinding],
    ) -> GuardrailAction:
        """
        Determine overall guardrail action.

        Priority:

            BLOCK > REVIEW > SANITIZE > ALLOW
        """

        if any(
            finding.action == GuardrailAction.BLOCK
            for finding in findings
        ):
            return GuardrailAction.BLOCK

        if any(
            finding.action == GuardrailAction.REVIEW
            for finding in findings
        ):
            return GuardrailAction.REVIEW

        if findings:
            return GuardrailAction.SANITIZE

        return GuardrailAction.ALLOW

    # =========================================================================
    # Utility Methods
    # =========================================================================

    @staticmethod
    def _to_dict(
        request: Any,
    ) -> Dict[str, Any]:
        """
        Convert Pydantic model or dictionary to a dictionary.
        """

        if isinstance(request, dict):
            return dict(request)

        if hasattr(request, "model_dump"):
            return request.model_dump()

        if hasattr(request, "dict"):
            return request.dict()

        raise TypeError(
            "Input request must be a dictionary or Pydantic model."
        )

    @staticmethod
    def _has_content(
        value: Any,
    ) -> bool:
        """
        Determine whether a value contains usable content.
        """

        if value is None:
            return False

        if isinstance(value, str):
            return bool(value.strip())

        if isinstance(value, (list, tuple, dict)):
            return len(value) > 0

        return True

    @staticmethod
    def _calculate_text_length(
        value: Any,
    ) -> int:
        """
        Calculate approximate textual representation length.
        """

        if isinstance(value, str):
            return len(value)

        if isinstance(value, (list, tuple)):
            return sum(
                len(str(item))
                for item in value
            )

        if isinstance(value, dict):
            return sum(
                len(str(key)) + len(str(item))
                for key, item in value.items()
            )

        return len(str(value))

    @staticmethod
    def _extract_text(
        data: Dict[str, Any],
    ) -> str:
        """
        Extract text for security pattern detection.

        The resulting text is used internally only and is NEVER logged.
        """

        values: List[str] = []

        for value in data.values():

            if isinstance(value, str):
                values.append(value)

            elif isinstance(value, list):
                values.extend(
                    str(item)
                    for item in value
                )

            elif isinstance(value, dict):
                values.extend(
                    str(item)
                    for item in value.values()
                )

        return " ".join(values)