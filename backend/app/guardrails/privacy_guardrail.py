"""
app/guardrails/privacy_guardrail.py

Privacy Guardrail for the Clinical Decision Support system.

Purpose:
    Detect, minimize, redact, and control sensitive PHI/PII before
    information is sent to LLMs, tools, logs, external services,
    or downstream components.

Architecture:

    Clinical Request
          |
          v
    PrivacyGuardrail
          |
          +-------------------------+
          |                         |
       ALLOW                    REDACT/BLOCK
          |                         |
          v                         v
    Agentic Workflow          Privacy Review
          |
          v
        LLM/RAG
          |
          v
    Output Privacy Check
          |
          v
    API Response

Responsibilities:
    - Detect common PHI/PII patterns
    - Identify sensitive fields
    - Redact sensitive values when appropriate
    - Minimize unnecessary personal information
    - Detect identifiers embedded in free text
    - Protect credentials/secrets accidentally supplied by users
    - Prevent sensitive data from being written to application logs
    - Provide privacy findings to the workflow
    - Preserve clinical meaning where possible

Non-responsibilities:
    - Does NOT diagnose
    - Does NOT determine clinical risk
    - Does NOT replace organizational privacy policy
    - Does NOT replace HIPAA/GDPR/local regulatory compliance
    - Does NOT provide legal advice

IMPORTANT:
    Regex-based detection is only a first layer.
    Production healthcare systems should combine:
        - deterministic patterns
        - schema-aware field classification
        - enterprise DLP
        - approved PHI/PII detection models
        - data classification
        - access controls
        - encryption
        - audit logging
        - retention policies
        - human/privacy review where appropriate

Logging:
    - Uses Loguru
    - NEVER logs detected PHI/PII
    - NEVER logs raw clinical text
    - NEVER logs redacted values
    - NEVER logs tokens/API keys/passwords
    - Logs only field names, rule IDs, counts and statuses
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from app.config.settings import settings
logger = settings.get_logger()

from pydantic import BaseModel, Field


# =============================================================================
# Configuration
# =============================================================================

MAX_TEXT_LENGTH = 50_000

DEFAULT_REDACTION_TOKEN = "[REDACTED]"

# Common sensitive fields in healthcare applications.
SENSITIVE_FIELD_NAMES = {
    "name",
    "first_name",
    "last_name",
    "full_name",
    "middle_name",
    "patient_name",
    "date_of_birth",
    "dob",
    "birth_date",
    "address",
    "street",
    "city",
    "state",
    "postal_code",
    "zip_code",
    "phone",
    "phone_number",
    "mobile",
    "email",
    "email_address",
    "ssn",
    "social_security_number",
    "mrn",
    "medical_record_number",
    "patient_id",
    "member_id",
    "insurance_id",
    "policy_number",
    "account_number",
    "passport",
    "passport_number",
    "driver_license",
    "national_id",
    "ip_address",
    "device_id",
    "biometric",
}

# Fields that can contain clinical content and therefore should NOT
# automatically be fully redacted. Instead, free-text scanning is applied.
CLINICAL_TEXT_FIELDS = {
    "symptoms",
    "history",
    "clinical_history",
    "clinical_context",
    "clinical_notes",
    "notes",
    "assessment",
    "diagnosis",
    "diagnoses",
    "labs",
    "laboratory_results",
    "imaging",
    "radiology_report",
    "medications",
    "allergies",
    "message",
    "question",
    "prompt",
    "conversation",
}


# =============================================================================
# Enumerations
# =============================================================================


class PrivacySeverity(str, Enum):
    """
    Privacy finding severity.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class PrivacyAction(str, Enum):
    """
    Privacy action.
    """

    ALLOW = "allow"
    REDACT = "redact"
    REVIEW = "review"
    BLOCK = "block"


class PrivacyFindingType(str, Enum):
    """
    Type of sensitive information detected.
    """

    PHI = "phi"
    PII = "pii"
    IDENTIFIER = "identifier"
    CREDENTIAL = "credential"
    SECRET = "secret"
    DIRECT_IDENTIFIER = "direct_identifier"
    QUASI_IDENTIFIER = "quasi_identifier"
    DATA_MINIMIZATION = "data_minimization"


# =============================================================================
# Pydantic Models
# =============================================================================


class PrivacyFinding(BaseModel):
    """
    Individual privacy finding.

    IMPORTANT:
        This model stores metadata about the detection, not the
        detected sensitive value itself.
    """

    rule_id: str = Field(
        ...,
        description="Privacy rule identifier.",
    )

    finding_type: PrivacyFindingType = Field(
        ...,
        description="Type of sensitive information detected.",
    )

    field: Optional[str] = Field(
        default=None,
        description="Field where sensitive data was detected.",
    )

    severity: PrivacySeverity = Field(
        ...,
        description="Privacy severity.",
    )

    action: PrivacyAction = Field(
        ...,
        description="Required privacy action.",
    )

    match_count: int = Field(
        default=1,
        description="Number of matches detected.",
    )

    reason: str = Field(
        ...,
        description="Safe reason without exposing sensitive values.",
    )


class PrivacyGuardrailResult(BaseModel):
    """
    Result of privacy guardrail evaluation.
    """

    allowed: bool = Field(
        ...,
        description="Whether processing may continue.",
    )

    action: PrivacyAction = Field(
        ...,
        description="Overall privacy action.",
    )

    findings: List[PrivacyFinding] = Field(
        default_factory=list,
        description="Privacy findings.",
    )

    sanitized_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Privacy-sanitized representation of the input.",
    )

    redaction_count: int = Field(
        default=0,
        description="Total number of redactions.",
    )

    requires_privacy_review: bool = Field(
        default=False,
        description="Whether privacy review is required.",
    )


# =============================================================================
# Privacy Guardrail
# =============================================================================


class PrivacyGuardrail:
    """
    PHI/PII privacy guardrail.

    The guardrail supports two complementary operations:

        1. validate()
           Detect privacy issues and return findings.

        2. sanitize()
           Create a privacy-sanitized copy of the input.

    The original object is never modified.

    The guardrail should be used at multiple trust boundaries:

        API input
        -> before LLM
        -> before external tool
        -> before logging
        -> before external integration
        -> before output
    """

    def __init__(
        self,
        redaction_token: str = DEFAULT_REDACTION_TOKEN,
        max_text_length: int = MAX_TEXT_LENGTH,
    ) -> None:

        self.redaction_token = redaction_token
        self.max_text_length = max_text_length

        self._patterns = self._build_patterns()

        logger.info(
            "PrivacyGuardrail initialized | patterns={} | "
            "max_text_length={}",
            len(self._patterns),
            max_text_length,
        )

    # =========================================================================
    # Public Validation API
    # =========================================================================

    async def validate(
        self,
        data: Any,
        workflow_id: Optional[str] = None,
    ) -> PrivacyGuardrailResult:
        """
        Validate data for PHI/PII/privacy concerns.

        Args:
            data:
                Dictionary, Pydantic model, or supported object.

            workflow_id:
                Safe workflow correlation identifier.

        Returns:
            PrivacyGuardrailResult.
        """

        workflow_id = (
            workflow_id
            or self._get_workflow_id(data)
            or "unknown"
        )

        logger.info(
            "PrivacyGuardrail validation started | workflow_id={}",
            workflow_id,
        )

        try:

            normalized = self._to_dict(data)

            findings: List[PrivacyFinding] = []

            # -------------------------------------------------------------
            # Sensitive field detection
            # -------------------------------------------------------------

            findings.extend(
                self._detect_sensitive_fields(
                    normalized
                )
            )

            # -------------------------------------------------------------
            # Free-text PHI/PII detection
            # -------------------------------------------------------------

            findings.extend(
                self._detect_sensitive_patterns(
                    normalized
                )
            )

            # -------------------------------------------------------------
            # Credential/secret detection
            # -------------------------------------------------------------

            findings.extend(
                self._detect_credentials(
                    normalized
                )
            )

            # -------------------------------------------------------------
            # Data minimization
            # -------------------------------------------------------------

            findings.extend(
                self._check_data_minimization(
                    normalized
                )
            )

            # -------------------------------------------------------------
            # Determine action
            # -------------------------------------------------------------

            action = self._determine_action(
                findings
            )

            allowed = (
                action != PrivacyAction.BLOCK
            )

            requires_review = (
                action == PrivacyAction.REVIEW
            )

            sanitized_data = None

            if allowed:

                sanitized_data = self.sanitize(
                    normalized
                )

            redaction_count = (
                self._count_redactions(
                    normalized
                )
            )

            result = PrivacyGuardrailResult(
                allowed=allowed,
                action=action,
                findings=findings,
                sanitized_data=sanitized_data,
                redaction_count=redaction_count,
                requires_privacy_review=requires_review,
            )

            logger.info(
                "PrivacyGuardrail validation completed | "
                "workflow_id={} | action={} | findings={} | "
                "redactions={} | review={}",
                workflow_id,
                action.value,
                len(findings),
                redaction_count,
                requires_review,
            )

            return result

        except Exception:

            logger.exception(
                "PrivacyGuardrail validation failed | workflow_id={}",
                workflow_id,
            )

            raise

    # =========================================================================
    # Sanitization API
    # =========================================================================

    def sanitize(
        self,
        data: Any,
    ) -> Dict[str, Any]:
        """
        Return a privacy-sanitized copy of the input.

        The original object is never modified.

        Important:
            Clinical free-text is not automatically deleted.
            Identifiers embedded in clinical text are redacted where
            deterministic patterns allow reliable detection.
        """

        normalized = self._to_dict(data)

        return self._sanitize_value(
            normalized,
            path="root",
        )

    # =========================================================================
    # Sensitive Fields
    # =========================================================================

    def _detect_sensitive_fields(
        self,
        data: Dict[str, Any],
        path: str = "",
    ) -> List[PrivacyFinding]:
        """
        Detect fields whose names indicate PHI/PII.
        """

        findings = []

        for key, value in data.items():

            current_path = (
                f"{path}.{key}"
                if path
                else key
            )

            normalized_key = (
                str(key)
                .lower()
                .strip()
                .replace("-", "_")
                .replace(" ", "_")
            )

            if normalized_key in SENSITIVE_FIELD_NAMES:

                findings.append(
                    PrivacyFinding(
                        rule_id="PRIVACY-001",
                        finding_type=(
                            PrivacyFindingType.DIRECT_IDENTIFIER
                        ),
                        field=current_path,
                        severity=PrivacySeverity.HIGH,
                        action=PrivacyAction.REDACT,
                        match_count=1,
                        reason=(
                            "Field is classified as a direct "
                            "personal or healthcare identifier."
                        ),
                    )
                )

            elif isinstance(value, dict):

                findings.extend(
                    self._detect_sensitive_fields(
                        value,
                        current_path,
                    )
                )

            elif isinstance(value, list):

                for index, item in enumerate(value):

                    if isinstance(item, dict):

                        findings.extend(
                            self._detect_sensitive_fields(
                                item,
                                f"{current_path}[{index}]",
                            )
                        )

        return findings

    # =========================================================================
    # Pattern Detection
    # =========================================================================

    def _detect_sensitive_patterns(
        self,
        data: Dict[str, Any],
        path: str = "",
    ) -> List[PrivacyFinding]:
        """
        Detect common PHI/PII patterns in free text.
        """

        findings = []

        for key, value in data.items():

            current_path = (
                f"{path}.{key}"
                if path
                else key
            )

            if isinstance(value, str):

                findings.extend(
                    self._scan_text(
                        value,
                        current_path,
                    )
                )

            elif isinstance(value, dict):

                findings.extend(
                    self._detect_sensitive_patterns(
                        value,
                        current_path,
                    )
                )

            elif isinstance(value, list):

                for index, item in enumerate(value):

                    if isinstance(item, str):

                        findings.extend(
                            self._scan_text(
                                item,
                                f"{current_path}[{index}]",
                            )
                        )

                    elif isinstance(item, dict):

                        findings.extend(
                            self._detect_sensitive_patterns(
                                item,
                                f"{current_path}[{index}]",
                            )
                        )

        return findings

    def _scan_text(
        self,
        text: str,
        field_path: str,
    ) -> List[PrivacyFinding]:
        """
        Scan text for privacy-sensitive patterns.

        Sensitive values are never returned or logged.
        """

        if not text:
            return []

        text_to_scan = text[
            : self.max_text_length
        ]

        findings = []

        for (
            rule_id,
            finding_type,
            pattern,
            severity,
            action,
            reason,
        ) in self._patterns:

            matches = re.findall(
                pattern,
                text_to_scan,
                flags=re.IGNORECASE,
            )

            if not matches:
                continue

            findings.append(
                PrivacyFinding(
                    rule_id=rule_id,
                    finding_type=finding_type,
                    field=field_path,
                    severity=severity,
                    action=action,
                    match_count=len(matches),
                    reason=reason,
                )
            )

        return findings

    # =========================================================================
    # Credentials / Secrets
    # =========================================================================

    def _detect_credentials(
        self,
        data: Dict[str, Any],
    ) -> List[PrivacyFinding]:
        """
        Detect common credentials and secrets.

        These are security-sensitive and should normally be blocked
        from entering LLM workflows.
        """

        findings = []

        credential_fields = {
            "password",
            "passwd",
            "secret",
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "authorization",
            "private_key",
            "client_secret",
        }

        def walk(
            value: Any,
            path: str,
        ) -> None:

            if isinstance(value, dict):

                for key, child in value.items():

                    current_path = (
                        f"{path}.{key}"
                        if path
                        else str(key)
                    )

                    normalized_key = (
                        str(key)
                        .lower()
                        .strip()
                        .replace("-", "_")
                    )

                    if normalized_key in credential_fields:

                        findings.append(
                            PrivacyFinding(
                                rule_id="PRIVACY-002",
                                finding_type=(
                                    PrivacyFindingType.CREDENTIAL
                                ),
                                field=current_path,
                                severity=PrivacySeverity.CRITICAL,
                                action=PrivacyAction.BLOCK,
                                match_count=1,
                                reason=(
                                    "Credential or secret detected. "
                                    "Sensitive credentials must not "
                                    "enter the clinical AI workflow."
                                ),
                            )
                        )

                    else:

                        walk(
                            child,
                            current_path,
                        )

            elif isinstance(value, list):

                for index, child in enumerate(value):

                    walk(
                        child,
                        f"{path}[{index}]",
                    )

        walk(data, "root")

        return findings

    # =========================================================================
    # Data Minimization
    # =========================================================================

    def _check_data_minimization(
        self,
        data: Dict[str, Any],
    ) -> List[PrivacyFinding]:
        """
        Identify potentially unnecessary direct identifiers.

        This is intentionally conservative because the guardrail
        cannot know the purpose of every application field.
        """

        findings = []

        unnecessary_fields = {
            "passport_number",
            "driver_license",
            "social_security_number",
            "ssn",
            "bank_account",
            "credit_card_number",
        }

        def walk(
            value: Any,
            path: str,
        ) -> None:

            if not isinstance(value, dict):
                return

            for key, child in value.items():

                current_path = (
                    f"{path}.{key}"
                    if path
                    else str(key)
                )

                normalized_key = (
                    str(key)
                    .lower()
                    .strip()
                    .replace("-", "_")
                )

                if normalized_key in unnecessary_fields:

                    findings.append(
                        PrivacyFinding(
                            rule_id="PRIVACY-003",
                            finding_type=(
                                PrivacyFindingType.DATA_MINIMIZATION
                            ),
                            field=current_path,
                            severity=PrivacySeverity.HIGH,
                            action=PrivacyAction.REVIEW,
                            match_count=1,
                            reason=(
                                "The field contains a highly sensitive "
                                "identifier that may not be necessary "
                                "for clinical reasoning."
                            ),
                        )
                    )

                elif isinstance(child, dict):

                    walk(
                        child,
                        current_path,
                    )

                elif isinstance(child, list):

                    for index, item in enumerate(child):

                        if isinstance(item, dict):

                            walk(
                                item,
                                f"{current_path}[{index}]",
                            )

        walk(data, "root")

        return findings

    # =========================================================================
    # Sanitization
    # =========================================================================

    def _sanitize_value(
        self,
        value: Any,
        path: str,
    ) -> Any:
        """
        Recursively sanitize dictionaries/lists/strings.
        """

        if isinstance(value, dict):

            sanitized = {}

            for key, child in value.items():

                normalized_key = (
                    str(key)
                    .lower()
                    .strip()
                    .replace("-", "_")
                    .replace(" ", "_")
                )

                current_path = (
                    f"{path}.{key}"
                    if path
                    else str(key)
                )

                # ---------------------------------------------------------
                # Direct identifiers
                # ---------------------------------------------------------

                if normalized_key in SENSITIVE_FIELD_NAMES:

                    sanitized[key] = (
                        self.redaction_token
                    )

                    continue

                # ---------------------------------------------------------
                # Credential / secret fields
                # ---------------------------------------------------------

                if normalized_key in {
                    "password",
                    "passwd",
                    "secret",
                    "api_key",
                    "apikey",
                    "access_token",
                    "refresh_token",
                    "authorization",
                    "private_key",
                    "client_secret",
                }:

                    sanitized[key] = (
                        self.redaction_token
                    )

                    continue

                sanitized[key] = (
                    self._sanitize_value(
                        child,
                        current_path,
                    )
                )

            return sanitized

        if isinstance(value, list):

            return [
                self._sanitize_value(
                    item,
                    f"{path}[{index}]",
                )
                for index, item in enumerate(value)
            ]

        if isinstance(value, str):

            return self._redact_text(
                value
            )

        return value

    def _redact_text(
        self,
        text: str,
    ) -> str:
        """
        Redact detectable identifiers in free text.

        Only deterministic patterns are used here.
        """

        sanitized = text

        for (
            _rule_id,
            _finding_type,
            pattern,
            _severity,
            action,
            _reason,
        ) in self._patterns:

            if action not in {
                PrivacyAction.REDACT,
                PrivacyAction.BLOCK,
            }:
                continue

            sanitized = re.sub(
                pattern,
                self.redaction_token,
                sanitized,
                flags=re.IGNORECASE,
            )

        return sanitized

    # =========================================================================
    # Action Resolution
    # =========================================================================

    def _determine_action(
        self,
        findings: List[PrivacyFinding],
    ) -> PrivacyAction:
        """
        Determine overall privacy action.

        Priority:

            BLOCK > REVIEW > REDACT > ALLOW
        """

        if any(
            finding.action == PrivacyAction.BLOCK
            for finding in findings
        ):
            return PrivacyAction.BLOCK

        if any(
            finding.action == PrivacyAction.REVIEW
            for finding in findings
        ):
            return PrivacyAction.REVIEW

        if any(
            finding.action == PrivacyAction.REDACT
            for finding in findings
        ):
            return PrivacyAction.REDACT

        return PrivacyAction.ALLOW

    # =========================================================================
    # Pattern Definitions
    # =========================================================================

    @staticmethod
    def _build_patterns() -> List[Tuple]:
        """
        Build deterministic PHI/PII detection patterns.

        These patterns are intentionally conservative and should not
        be considered a complete HIPAA/PII classifier.
        """

        return [

            # -------------------------------------------------------------
            # Email address
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-001",
                PrivacyFindingType.PII,
                r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
                PrivacySeverity.HIGH,
                PrivacyAction.REDACT,
                "Email address detected.",
            ),

            # -------------------------------------------------------------
            # US-style phone number
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-002",
                PrivacyFindingType.PII,
                r"\b(?:\+?1[-.\s]?)?"
                r"(?:\(?\d{3}\)?[-.\s]?)"
                r"\d{3}[-.\s]?\d{4}\b",
                PrivacySeverity.HIGH,
                PrivacyAction.REDACT,
                "Phone number pattern detected.",
            ),

            # -------------------------------------------------------------
            # SSN
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-003",
                PrivacyFindingType.PHI,
                r"\b\d{3}-\d{2}-\d{4}\b",
                PrivacySeverity.CRITICAL,
                PrivacyAction.REDACT,
                "Social Security Number pattern detected.",
            ),

            # -------------------------------------------------------------
            # Date of birth
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-004",
                PrivacyFindingType.PHI,
                r"\b(?:"
                r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
                r"|"
                r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
                r")\b",
                PrivacySeverity.MEDIUM,
                PrivacyAction.REDACT,
                "Date-like identifier detected.",
            ),

            # -------------------------------------------------------------
            # IPv4 address
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-005",
                PrivacyFindingType.PII,
                r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
                PrivacySeverity.MEDIUM,
                PrivacyAction.REDACT,
                "IP address pattern detected.",
            ),

            # -------------------------------------------------------------
            # Credit card-like number
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-006",
                PrivacyFindingType.PII,
                r"\b(?:\d[ -]*?){13,19}\b",
                PrivacySeverity.CRITICAL,
                PrivacyAction.BLOCK,
                "Potential payment-card identifier detected.",
            ),

            # -------------------------------------------------------------
            # Generic MRN-like identifier.
            #
            # This is intentionally conservative because MRN formats
            # vary by organization.
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-007",
                PrivacyFindingType.IDENTIFIER,
                r"\b(?:MRN|Medical\s+Record\s+Number)"
                r"\s*[:#-]?\s*[A-Z0-9-]{4,}\b",
                PrivacySeverity.HIGH,
                PrivacyAction.REDACT,
                "Medical record identifier pattern detected.",
            ),

            # -------------------------------------------------------------
            # API key / bearer token
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-008",
                PrivacyFindingType.SECRET,
                r"\bBearer\s+[A-Za-z0-9._~+/=-]{20,}\b",
                PrivacySeverity.CRITICAL,
                PrivacyAction.BLOCK,
                "Bearer token pattern detected.",
            ),

            # -------------------------------------------------------------
            # Generic secret assignment
            # -------------------------------------------------------------

            (
                "PRIVACY-PATTERN-009",
                PrivacyFindingType.SECRET,
                r"\b(?:api[_-]?key|access[_-]?token|secret)"
                r"\s*[:=]\s*[A-Za-z0-9._~+/=-]{12,}\b",
                PrivacySeverity.CRITICAL,
                PrivacyAction.BLOCK,
                "Potential secret/token pattern detected.",
            ),
        ]

    # =========================================================================
    # Utility
    # =========================================================================

    @staticmethod
    def _to_dict(
        data: Any,
    ) -> Dict[str, Any]:
        """
        Convert supported object types to a dictionary.
        """

        if isinstance(data, dict):
            return dict(data)

        if hasattr(data, "model_dump"):
            return data.model_dump()

        if hasattr(data, "dict"):
            return data.dict()

        raise TypeError(
            "PrivacyGuardrail expects a dictionary or Pydantic model."
        )

    @staticmethod
    def _get_workflow_id(
        data: Any,
    ) -> Optional[str]:
        """
        Extract workflow ID for safe correlation logging.
        """

        try:

            if isinstance(data, dict):
                return data.get("workflow_id")

            return getattr(
                data,
                "workflow_id",
                None,
            )

        except Exception:

            return None

    def _count_redactions(
        self,
        data: Dict[str, Any],
    ) -> int:
        """
        Estimate number of values that will be redacted.

        Only counts are returned; actual values are never logged.
        """

        findings = []

        findings.extend(
            self._detect_sensitive_fields(data)
        )

        findings.extend(
            self._detect_sensitive_patterns(data)
        )

        findings.extend(
            self._detect_credentials(data)
        )

        return sum(
            finding.match_count
            for finding in findings
            if finding.action
            in {
                PrivacyAction.REDACT,
                PrivacyAction.BLOCK,
            }
        )