"""
app/security/audit.py

Security Audit Logging for the Clinical Decision Support System.

Responsibilities
----------------
1. Record security-relevant user/system activities.
2. Provide structured audit events.
3. Record authentication and authorization events.
4. Record access to protected clinical resources.
5. Record clinical workflow events.
6. Record human-review events.
7. Record security/policy violations.
8. Prevent secrets, tokens, passwords, and raw PHI from being logged.
9. Support correlation through request_id / workflow_id.
10. Use Loguru as the application logging backend.

IMPORTANT
---------
Audit logging is different from normal application logging.

Application logging:
    "What happened inside the application?"

Audit logging:
    "Who did what, to which resource, when, from where,
     and what was the outcome?"

For production healthcare systems, audit events should normally also
be persisted into a tamper-resistant audit store / SIEM / database.

This module intentionally does NOT:
    - authenticate users
    - authorize users
    - make clinical decisions
    - store clinical payloads
    - store access tokens
    - store passwords
    - store LLM prompts/responses
    - store embeddings

```

However, **do not pass the raw `patient_id` into ordinary Loguru output**. The implementation above hashes sensitive identifiers before emitting them.

### One production improvement I strongly recommend

For your healthcare system, treat Loguru as the **audit event generation/transport layer**, not the ultimate audit database.

```text
Application
     │
     ▼
AuditLogger
     │
     ▼
Loguru
     │
     ├──────────────► stdout / application logs
     │
     └──────────────► SIEM / centralized logging
                              │
                              ▼
                     Durable Audit Store
                              │
                     ┌────────┴────────┐
                     ▼                 ▼
                 Compliance        Security
                   Audit             SOC
```

For a production environment, audit records should ideally be **append-only, access-controlled, encrypted, retained according to policy, time-synchronized, and protected against tampering**. The audit record should capture *who, what, when, outcome, resource, and correlation IDs* without putting clinical content into the audit log.

One further improvement for your existing architecture is to add a **request/workflow correlation context** so `request_id` and `workflow_id` automatically propagate through Router → Service → Implementation → LangGraph → Agents → Audit, without manually passing them through every function.

"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional

from app.config.settings import settings
logger = settings.get_logger()

from pydantic import BaseModel, ConfigDict, Field


# ============================================================================
# ENUMS
# ============================================================================


class AuditEventType(str, Enum):
    """
    Categories of security and compliance events.
    """

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    AUTHENTICATION_SUCCESS = "authentication.success"
    AUTHENTICATION_FAILURE = "authentication.failure"
    LOGOUT = "authentication.logout"
    TOKEN_CREATED = "authentication.token_created"
    TOKEN_VALIDATION_FAILURE = "authentication.token_validation_failure"

    # ------------------------------------------------------------------
    # Authorization
    # ------------------------------------------------------------------

    AUTHORIZATION_ALLOWED = "authorization.allowed"
    AUTHORIZATION_DENIED = "authorization.denied"
    PRIVILEGE_ESCALATION_ATTEMPT = "authorization.privilege_escalation_attempt"

    # ------------------------------------------------------------------
    # Patient / Clinical Data Access
    # ------------------------------------------------------------------

    PATIENT_RECORD_READ = "patient.record_read"
    PATIENT_RECORD_CREATE = "patient.record_create"
    PATIENT_RECORD_UPDATE = "patient.record_update"
    PATIENT_RECORD_DELETE = "patient.record_delete"

    CLINICAL_RECORD_READ = "clinical_record.read"
    CLINICAL_RECORD_CREATE = "clinical_record.create"
    CLINICAL_RECORD_UPDATE = "clinical_record.update"

    DIAGNOSTIC_RESULT_READ = "diagnostic_result.read"
    DIAGNOSTIC_RESULT_CREATE = "diagnostic_result.create"
    DIAGNOSTIC_RESULT_UPDATE = "diagnostic_result.update"

    MEDICATION_READ = "medication.read"
    MEDICATION_CREATE = "medication.create"
    MEDICATION_UPDATE = "medication.update"

    # ------------------------------------------------------------------
    # Clinical Decision Support
    # ------------------------------------------------------------------

    CLINICAL_ANALYSIS_STARTED = "clinical_analysis.started"
    CLINICAL_ANALYSIS_COMPLETED = "clinical_analysis.completed"
    CLINICAL_ANALYSIS_FAILED = "clinical_analysis.failed"

    CLINICAL_RECOMMENDATION_GENERATED = (
        "clinical_recommendation.generated"
    )

    CLINICAL_RECOMMENDATION_REVIEWED = (
        "clinical_recommendation.reviewed"
    )

    # ------------------------------------------------------------------
    # Human Review
    # ------------------------------------------------------------------

    HUMAN_REVIEW_REQUIRED = "human_review.required"
    HUMAN_REVIEW_STARTED = "human_review.started"
    HUMAN_REVIEW_APPROVED = "human_review.approved"
    HUMAN_REVIEW_REJECTED = "human_review.rejected"
    HUMAN_REVIEW_COMPLETED = "human_review.completed"

    # ------------------------------------------------------------------
    # RAG / Evidence
    # ------------------------------------------------------------------

    EVIDENCE_RETRIEVED = "evidence.retrieved"
    EVIDENCE_ACCESS_DENIED = "evidence.access_denied"

    # ------------------------------------------------------------------
    # Agentic Workflow
    # ------------------------------------------------------------------

    AGENT_EXECUTION_STARTED = "agent.execution_started"
    AGENT_EXECUTION_COMPLETED = "agent.execution_completed"
    AGENT_EXECUTION_FAILED = "agent.execution_failed"

    TOOL_INVOCATION_STARTED = "tool.invocation_started"
    TOOL_INVOCATION_COMPLETED = "tool.invocation_completed"
    TOOL_INVOCATION_FAILED = "tool.invocation_failed"

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------

    GUARDRAIL_WARNING = "guardrail.warning"
    GUARDRAIL_BLOCKED = "guardrail.blocked"
    GUARDRAIL_REVIEW_REQUIRED = "guardrail.review_required"

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------

    PHI_ACCESS = "privacy.phi_access"
    PHI_ACCESS_DENIED = "privacy.phi_access_denied"
    PRIVACY_VIOLATION = "privacy.violation"
    DATA_REDACTED = "privacy.data_redacted"

    # ------------------------------------------------------------------
    # Security
    # ------------------------------------------------------------------

    SECURITY_ALERT = "security.alert"
    SECURITY_POLICY_VIOLATION = "security.policy_violation"
    SUSPICIOUS_ACTIVITY = "security.suspicious_activity"

    # ------------------------------------------------------------------
    # Administration
    # ------------------------------------------------------------------

    USER_CREATED = "admin.user_created"
    USER_UPDATED = "admin.user_updated"
    USER_DISABLED = "admin.user_disabled"

    ROLE_ASSIGNED = "admin.role_assigned"
    ROLE_REMOVED = "admin.role_removed"

    POLICY_CREATED = "admin.policy_created"
    POLICY_UPDATED = "admin.policy_updated"

    SYSTEM_CONFIGURATION_CHANGED = (
        "admin.system_configuration_changed"
    )


class AuditOutcome(str, Enum):
    """
    Result of the audited operation.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    DENIED = "denied"
    BLOCKED = "blocked"
    WARNING = "warning"
    PENDING_REVIEW = "pending_review"


class AuditSeverity(str, Enum):
    """
    Audit event severity.
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class AuditActorType(str, Enum):
    """
    Entity that initiated the activity.
    """

    USER = "user"
    SYSTEM = "system"
    SERVICE = "service"
    AGENT = "agent"


# ============================================================================
# AUDIT EVENT MODEL
# ============================================================================


class AuditEvent(BaseModel):
    """
    Structured security audit event.

    IMPORTANT:
        metadata must contain only safe, non-sensitive metadata.

    Do NOT put:
        - patient name
        - DOB
        - address
        - phone
        - email
        - MRN
        - diagnosis
        - symptoms
        - laboratory values
        - medications
        - clinical notes
        - prompts
        - model responses
        - access tokens
        - passwords
        - API keys
        - secrets
    """

    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(
        default_factory=lambda: str(uuid.uuid4())
    )

    event_type: AuditEventType

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    severity: AuditSeverity = AuditSeverity.INFO

    outcome: AuditOutcome = AuditOutcome.SUCCESS

    actor_type: AuditActorType = AuditActorType.USER

    actor_id: Optional[str] = None

    actor_role: Optional[str] = None

    request_id: Optional[str] = None

    workflow_id: Optional[str] = None

    session_id: Optional[str] = None

    resource_type: Optional[str] = None

    resource_id: Optional[str] = None

    action: Optional[str] = None

    service_name: Optional[str] = None

    endpoint: Optional[str] = None

    http_method: Optional[str] = None

    status_code: Optional[int] = None

    ip_address_hash: Optional[str] = None

    user_agent_hash: Optional[str] = None

    reason: Optional[str] = None

    duration_ms: Optional[float] = None

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


# ============================================================================
# AUDIT LOGGER
# ============================================================================


class AuditLogger:
    """
    Centralized security audit logger.

    The class provides a controlled interface over Loguru.

    Example
    -------

        audit_logger.log(
            event_type=AuditEventType.AUTHORIZATION_ALLOWED,
            actor_id=user.user_id,
            actor_role=user.role,
            outcome=AuditOutcome.SUCCESS,
            action="clinical_decision.analyze",
        )
    """

    # Fields that should never be accepted inside metadata.
    FORBIDDEN_METADATA_KEYS = frozenset(
        {
            "password",
            "passwd",
            "secret",
            "api_key",
            "apikey",
            "access_token",
            "refresh_token",
            "authorization",
            "cookie",
            "session_token",
            "private_key",
            "client_secret",
            "prompt",
            "system_prompt",
            "llm_prompt",
            "response",
            "llm_response",
            "completion",
            "embedding",
            "clinical_note",
            "clinical_notes",
            "medical_record",
            "diagnosis",
            "symptoms",
            "medications",
            "lab_results",
            "imaging",
            "patient_name",
            "date_of_birth",
            "dob",
            "address",
            "phone",
            "email",
            "ssn",
            "mrn",
        }
    )

    # Sensitive keys that should be hashed instead of logged.
    HASHED_METADATA_KEYS = frozenset(
        {
            "patient_id",
            "member_id",
            "user_id",
            "actor_id",
            "resource_id",
            "session_id",
            "ip_address",
            "user_agent",
        }
    )

    def __init__(
        self,
        service_name: str = "clinical-decision-support",
    ) -> None:

        self.service_name = service_name

        logger.info(
            "AuditLogger initialized | service={}",
            service_name,
        )

    # ==================================================================
    # MAIN AUDIT METHOD
    # ==================================================================

    def log(
        self,
        *,
        event_type: AuditEventType,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        severity: AuditSeverity = AuditSeverity.INFO,
        actor_type: AuditActorType = AuditActorType.USER,
        actor_id: Optional[str] = None,
        actor_role: Optional[str] = None,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        session_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        action: Optional[str] = None,
        endpoint: Optional[str] = None,
        http_method: Optional[str] = None,
        status_code: Optional[int] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        reason: Optional[str] = None,
        duration_ms: Optional[float] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditEvent:
        """
        Create and emit an audit event.

        All sensitive identifiers are hashed before being written
        to the application log.
        """

        safe_metadata = self._sanitize_metadata(
            metadata or {}
        )

        event = AuditEvent(
            event_type=event_type,
            severity=severity,
            outcome=outcome,
            actor_type=actor_type,
            actor_id=self._hash_identifier(actor_id),
            actor_role=actor_role,
            request_id=self._safe_identifier(request_id),
            workflow_id=self._safe_identifier(workflow_id),
            session_id=self._hash_identifier(session_id),
            resource_type=resource_type,
            resource_id=self._hash_identifier(resource_id),
            action=action,
            service_name=self.service_name,
            endpoint=endpoint,
            http_method=http_method,
            status_code=status_code,
            ip_address_hash=self._hash_identifier(
                ip_address
            ),
            user_agent_hash=self._hash_identifier(
                user_agent
            ),
            reason=self._sanitize_reason(reason),
            duration_ms=duration_ms,
            metadata=safe_metadata,
        )

        self._emit(event)

        return event

    # ==================================================================
    # AUTHENTICATION
    # ==================================================================

    def authentication_success(
        self,
        *,
        actor_id: str,
        request_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AUTHENTICATION_SUCCESS,
            actor_id=actor_id,
            request_id=request_id,
            ip_address=ip_address,
            user_agent=user_agent,
            action="authentication",
            outcome=AuditOutcome.SUCCESS,
        )

    def authentication_failure(
        self,
        *,
        actor_id: Optional[str] = None,
        request_id: Optional[str] = None,
        reason: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AUTHENTICATION_FAILURE,
            actor_id=actor_id,
            request_id=request_id,
            ip_address=ip_address,
            reason=reason,
            action="authentication",
            outcome=AuditOutcome.FAILURE,
            severity=AuditSeverity.WARNING,
        )

    # ==================================================================
    # AUTHORIZATION
    # ==================================================================

    def authorization_allowed(
        self,
        *,
        actor_id: str,
        actor_role: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AUTHORIZATION_ALLOWED,
            actor_id=actor_id,
            actor_role=actor_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            outcome=AuditOutcome.SUCCESS,
        )

    def authorization_denied(
        self,
        *,
        actor_id: Optional[str],
        actor_role: Optional[str],
        action: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        request_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AUTHORIZATION_DENIED,
            actor_id=actor_id,
            actor_role=actor_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            reason=reason,
            outcome=AuditOutcome.DENIED,
            severity=AuditSeverity.WARNING,
        )

    # ==================================================================
    # PHI / CLINICAL DATA ACCESS
    # ==================================================================

    def phi_access(
        self,
        *,
        actor_id: str,
        actor_role: Optional[str],
        action: str,
        resource_type: str,
        resource_id: str,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.PHI_ACCESS,
            actor_id=actor_id,
            actor_role=actor_role,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            request_id=request_id,
            workflow_id=workflow_id,
            reason=purpose,
            outcome=AuditOutcome.SUCCESS,
        )

    def clinical_record_read(
        self,
        *,
        actor_id: str,
        actor_role: Optional[str],
        resource_id: str,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.CLINICAL_RECORD_READ,
            actor_id=actor_id,
            actor_role=actor_role,
            action="clinical_record.read",
            resource_type="clinical_record",
            resource_id=resource_id,
            request_id=request_id,
            workflow_id=workflow_id,
        )

    # ==================================================================
    # CLINICAL DECISION WORKFLOW
    # ==================================================================

    def clinical_analysis_started(
        self,
        *,
        actor_id: Optional[str],
        workflow_id: str,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.CLINICAL_ANALYSIS_STARTED,
            actor_type=AuditActorType.USER,
            actor_id=actor_id,
            workflow_id=workflow_id,
            request_id=request_id,
            action="clinical_decision.analyze",
            outcome=AuditOutcome.SUCCESS,
        )

    def clinical_analysis_completed(
        self,
        *,
        actor_id: Optional[str],
        workflow_id: str,
        request_id: Optional[str] = None,
        duration_ms: Optional[float] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.CLINICAL_ANALYSIS_COMPLETED,
            actor_id=actor_id,
            workflow_id=workflow_id,
            request_id=request_id,
            action="clinical_decision.analyze",
            duration_ms=duration_ms,
            outcome=AuditOutcome.SUCCESS,
        )

    def clinical_analysis_failed(
        self,
        *,
        actor_id: Optional[str],
        workflow_id: str,
        request_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.CLINICAL_ANALYSIS_FAILED,
            actor_id=actor_id,
            workflow_id=workflow_id,
            request_id=request_id,
            action="clinical_decision.analyze",
            reason=reason,
            outcome=AuditOutcome.FAILURE,
            severity=AuditSeverity.ERROR,
        )

    # ==================================================================
    # HUMAN REVIEW
    # ==================================================================

    def human_review_required(
        self,
        *,
        workflow_id: str,
        reason: str,
        actor_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.HUMAN_REVIEW_REQUIRED,
            actor_id=actor_id,
            workflow_id=workflow_id,
            request_id=request_id,
            action="human_review.required",
            reason=reason,
            outcome=AuditOutcome.PENDING_REVIEW,
            severity=AuditSeverity.WARNING,
        )

    def human_review_completed(
        self,
        *,
        workflow_id: str,
        reviewer_id: str,
        outcome: AuditOutcome,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.HUMAN_REVIEW_COMPLETED,
            actor_id=reviewer_id,
            workflow_id=workflow_id,
            request_id=request_id,
            action="human_review.completed",
            outcome=outcome,
        )

    # ==================================================================
    # AGENT EXECUTION
    # ==================================================================

    def agent_started(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AGENT_EXECUTION_STARTED,
            actor_type=AuditActorType.AGENT,
            workflow_id=workflow_id,
            request_id=request_id,
            action="agent.execute",
            metadata={
                "agent_name": agent_name,
            },
        )

    def agent_completed(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        duration_ms: Optional[float] = None,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AGENT_EXECUTION_COMPLETED,
            actor_type=AuditActorType.AGENT,
            workflow_id=workflow_id,
            request_id=request_id,
            action="agent.execute",
            duration_ms=duration_ms,
            metadata={
                "agent_name": agent_name,
            },
        )

    def agent_failed(
        self,
        *,
        workflow_id: str,
        agent_name: str,
        reason: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.AGENT_EXECUTION_FAILED,
            actor_type=AuditActorType.AGENT,
            workflow_id=workflow_id,
            request_id=request_id,
            action="agent.execute",
            reason=reason,
            outcome=AuditOutcome.FAILURE,
            severity=AuditSeverity.ERROR,
            metadata={
                "agent_name": agent_name,
            },
        )

    # ==================================================================
    # GUARDRAILS
    # ==================================================================

    def guardrail_blocked(
        self,
        *,
        workflow_id: Optional[str],
        guardrail_name: str,
        reason: str,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.GUARDRAIL_BLOCKED,
            actor_type=AuditActorType.SYSTEM,
            workflow_id=workflow_id,
            request_id=request_id,
            action="guardrail.block",
            reason=reason,
            outcome=AuditOutcome.BLOCKED,
            severity=AuditSeverity.CRITICAL,
            metadata={
                "guardrail_name": guardrail_name,
            },
        )

    def guardrail_review_required(
        self,
        *,
        workflow_id: Optional[str],
        guardrail_name: str,
        reason: str,
        request_id: Optional[str] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.GUARDRAIL_REVIEW_REQUIRED,
            actor_type=AuditActorType.SYSTEM,
            workflow_id=workflow_id,
            request_id=request_id,
            action="guardrail.review",
            reason=reason,
            outcome=AuditOutcome.PENDING_REVIEW,
            severity=AuditSeverity.WARNING,
            metadata={
                "guardrail_name": guardrail_name,
            },
        )

    # ==================================================================
    # SECURITY ALERT
    # ==================================================================

    def security_alert(
        self,
        *,
        reason: str,
        severity: AuditSeverity = AuditSeverity.CRITICAL,
        actor_id: Optional[str] = None,
        request_id: Optional[str] = None,
        workflow_id: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> AuditEvent:

        return self.log(
            event_type=AuditEventType.SECURITY_ALERT,
            actor_type=AuditActorType.SYSTEM,
            actor_id=actor_id,
            request_id=request_id,
            workflow_id=workflow_id,
            reason=reason,
            severity=severity,
            outcome=AuditOutcome.FAILURE,
            metadata=metadata,
        )

    # ==================================================================
    # METADATA SANITIZATION
    # ==================================================================

    def _sanitize_metadata(
        self,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Sanitize metadata before writing it to the audit log.

        Nested dictionaries are recursively processed.

        Unknown values are converted to safe strings where possible.
        """

        sanitized: dict[str, Any] = {}

        for raw_key, value in metadata.items():

            key = str(raw_key).lower()

            # ----------------------------------------------------------
            # Explicitly forbidden fields
            # ----------------------------------------------------------

            if key in self.FORBIDDEN_METADATA_KEYS:
                sanitized[key] = "[REDACTED]"
                continue

            # ----------------------------------------------------------
            # Identifier fields
            # ----------------------------------------------------------

            if key in self.HASHED_METADATA_KEYS:
                sanitized[key] = self._hash_identifier(
                    str(value)
                )
                continue

            # ----------------------------------------------------------
            # Nested dictionaries
            # ----------------------------------------------------------

            if isinstance(value, Mapping):
                sanitized[key] = self._sanitize_metadata(
                    value
                )
                continue

            # ----------------------------------------------------------
            # Lists / tuples
            # ----------------------------------------------------------

            if isinstance(value, (list, tuple)):

                sanitized[key] = [
                    self._sanitize_value(item)
                    for item in value
                ]

                continue

            # ----------------------------------------------------------
            # Primitive / safe values
            # ----------------------------------------------------------

            sanitized[key] = self._sanitize_value(
                value
            )

        return sanitized

    def _sanitize_value(
        self,
        value: Any,
    ) -> Any:
        """
        Sanitize an individual metadata value.
        """

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, Mapping):
            return self._sanitize_metadata(value)

        if isinstance(value, (list, tuple)):
            return [
                self._sanitize_value(item)
                for item in value
            ]

        # Avoid serializing arbitrary objects because they may
        # contain sensitive fields.
        return f"<{type(value).__name__}>"

    # ==================================================================
    # REASON SANITIZATION
    # ==================================================================

    def _sanitize_reason(
        self,
        reason: Optional[str],
    ) -> Optional[str]:
        """
        Keep audit reasons short and prevent accidental PHI logging.

        Reasons should be categorical rather than clinical narratives.
        """

        if not reason:
            return None

        normalized = str(reason).strip()

        if len(normalized) > 500:
            normalized = normalized[:500]

        lowered = normalized.lower()

        sensitive_terms = (
            "password",
            "token",
            "api_key",
            "secret",
            "clinical note",
            "medical record",
            "diagnosis:",
            "symptoms:",
            "medications:",
            "lab result",
        )

        if any(
            term in lowered
            for term in sensitive_terms
        ):
            return "[REDACTED_SENSITIVE_REASON]"

        return normalized

    # ==================================================================
    # HASHING
    # ==================================================================

    @staticmethod
    def _hash_identifier(
        value: Optional[str],
    ) -> Optional[str]:
        """
        Hash an identifier before writing it to logs.

        SHA-256 is used here for correlation-safe logging.

        For highly sensitive identifiers, a keyed HMAC is preferable
        in a production environment because plain hashing can be
        vulnerable to dictionary attacks.
        """

        if not value:
            return None

        return hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()[:16]

    @staticmethod
    def _safe_identifier(
        value: Optional[str],
    ) -> Optional[str]:
        """
        Preserve application-generated correlation IDs.

        request_id/workflow_id should be UUIDs or opaque random IDs,
        not patient identifiers.
        """

        if not value:
            return None

        value = str(value)

        if len(value) > 128:
            return value[:128]

        return value

    # ==================================================================
    # EMIT
    # ==================================================================

    def _emit(
        self,
        event: AuditEvent,
    ) -> None:
        """
        Emit structured audit event through Loguru.

        In production, the Loguru sink should forward these records
        to a durable audit store / SIEM.
        """

        payload = event.model_dump(
            mode="json"
        )

        serialized = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

        if event.severity == AuditSeverity.DEBUG:
            logger.debug(
                "AUDIT {}",
                serialized,
            )

        elif event.severity == AuditSeverity.INFO:
            logger.info(
                "AUDIT {}",
                serialized,
            )

        elif event.severity == AuditSeverity.WARNING:
            logger.warning(
                "AUDIT {}",
                serialized,
            )

        elif event.severity == AuditSeverity.ERROR:
            logger.error(
                "AUDIT {}",
                serialized,
            )

        elif event.severity == AuditSeverity.CRITICAL:
            logger.critical(
                "AUDIT {}",
                serialized,
            )


# ============================================================================
# GLOBAL AUDIT LOGGER
# ============================================================================


audit_logger = AuditLogger(
    service_name="clinical-decision-support"
)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def audit_event(
    *,
    event_type: AuditEventType,
    outcome: AuditOutcome = AuditOutcome.SUCCESS,
    severity: AuditSeverity = AuditSeverity.INFO,
    actor_type: AuditActorType = AuditActorType.USER,
    actor_id: Optional[str] = None,
    actor_role: Optional[str] = None,
    request_id: Optional[str] = None,
    workflow_id: Optional[str] = None,
    session_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    action: Optional[str] = None,
    endpoint: Optional[str] = None,
    http_method: Optional[str] = None,
    status_code: Optional[int] = None,
    ip_address: Optional[str] = None,
    user_agent: Optional[str] = None,
    reason: Optional[str] = None,
    duration_ms: Optional[float] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> AuditEvent:

    return audit_logger.log(
        event_type=event_type,
        outcome=outcome,
        severity=severity,
        actor_type=actor_type,
        actor_id=actor_id,
        actor_role=actor_role,
        request_id=request_id,
        workflow_id=workflow_id,
        session_id=session_id,
        resource_type=resource_type,
        resource_id=resource_id,
        action=action,
        endpoint=endpoint,
        http_method=http_method,
        status_code=status_code,
        ip_address=ip_address,
        user_agent=user_agent,
        reason=reason,
        duration_ms=duration_ms,
        metadata=metadata,
    )


__all__ = [
    "AuditEventType",
    "AuditOutcome",
    "AuditSeverity",
    "AuditActorType",
    "AuditEvent",
    "AuditLogger",
    "audit_logger",
    "audit_event",
]
```

### How `audit.py` fits your security architecture

Your security package now has a clear separation:

```text
app/security/
│
├── authentication.py
│      │
│      └── WHO ARE YOU?
│
├── authorization.py
│      │
│      └── WHAT ARE YOU ALLOWED TO DO?
│
├── rbac.py
│      │
│      └── ROLE → PERMISSIONS
│
├── abac.py
│      │
│      └── CONTEXTUAL ACCESS
│
├── policies.py
│      │
│      └── SECURITY / ACCESS POLICIES
│
├── audit.py
│      │
│      └── WHAT HAPPENED?
│
├── dlp.py
│      │
│      └── DATA LOSS / PHI PROTECTION
│
├── secrets.py
│      │
│      └── SECRET MANAGEMENT
│
└── incident_response.py
       │
       └── SECURITY INCIDENT HANDLING
```

The most important distinction is:

```text
Authentication
       ↓
Authorization
       ↓
Audit
```

For example, when a doctor invokes your clinical decision endpoint:

```text
Doctor
  │
  ▼
Authentication
  │
  │ authenticated
  ▼
Authorization
  │
  │ CLINICAL_DECISION_ANALYZE = ALLOW
  ▼
Audit
  │
  │ clinical_analysis.started
  ▼
ClinicalDecisionService
  ▼
ClinicalDecisionImplementation
  ▼
LangGraph
```

And when the workflow completes:

```text
LangGraph
   │
   ├── Intake Agent
   ├── Context Agent
   ├── Evidence Agent
   ├── Differential Agent
   ├── Missing Information Agent
   ├── Diagnostic Test Agent
   ├── Safety Agent
   ├── Critic Agent
   └── Summary Agent
             │
             ▼
       Human Review
             │
             ▼
       Audit Event
             │
             └── clinical_analysis.completed
```

### Example: audit from `authorization.py`

Your authorization layer can record both successful and denied access:

```python
result = authorization_manager.authorize_permission(
    current_user,
    Permission.CLINICAL_DECISION_ANALYZE,
)

if result.authorized:
    audit_logger.authorization_allowed(
        actor_id=current_user.user_id,
        actor_role=current_user.role,
        action="clinical_decision.analyze",
        resource_type="clinical_decision",
        resource_id=request.patient_id,
        request_id=request_id,
    )
else:
    audit_logger.authorization_denied(
        actor_id=current_user.user_id,
        actor_role=current_user.role,
        action="clinical_decision.analyze",
        resource_type="clinical_decision",
        resource_id=request.patient_id,
        request_id=request_id,
        reason=result.reason.value,
    )