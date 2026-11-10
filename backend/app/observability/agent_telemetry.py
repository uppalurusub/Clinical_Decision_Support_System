"""
app/observability/agent_telemetry.py

Agent-level telemetry for the Clinical Decision Support &
Diagnosis Agentic AI system.

Responsibilities
----------------
1. Track Agent execution lifecycle.
2. Track Agent success/failure.
3. Track Agent latency.
4. Track Agent retries.
5. Track Agent tool invocations.
6. Track Agent workflow/graph iterations.
7. Track Agent reanalysis.
8. Track Agent human-review requirements.
9. Track Agent token usage.
10. Correlate Agent telemetry with:
       - request_id
       - workflow_id
       - trace_id
       - span_id
11. Use Loguru for operational logging.
12. Prevent PHI/PII/secrets from entering telemetry.

This module is specifically for AGENT OBSERVABILITY.

It is NOT:
    - application logging
    - security auditing
    - clinical safety logic
    - diagnosis
    - LLM evaluation
    - authorization

Recommended architecture
------------------------

FastAPI
   |
   +-- logging.py
   +-- metrics.py
   +-- tracing.py
   |
   +-- agent_telemetry.py
           |
           +-- IntakeAgent
           +-- ContextAgent
           +-- EvidenceAgent
           +-- DifferentialAgent
           +-- MissingInformationAgent
           +-- DiagnosticTestAgent
           +-- SafetyAgent
           +-- CriticAgent
           +-- SummaryAgent

Sensitive data policy
---------------------
NEVER record:

    - patient name
    - patient_id in raw form
    - DOB
    - MRN
    - symptoms
    - clinical notes
    - diagnosis
    - medications
    - laboratory values
    - imaging
    - prompts
    - LLM responses
    - retrieved clinical text
    - vector embeddings
    - access tokens
    - passwords
    - API keys
    - secrets

Only operational metadata should be captured.

```

Notice that the **tool name** and **duration** are captured, but not the clinical query or retrieved clinical documents.

### LangGraph reanalysis telemetry

This is particularly important for your architecture because your `CriticAgent` can route back to another Agent:

```text id="a8p5s4"
Differential
     │
     ▼
Missing Information
     │
     ▼
Diagnostic Tests
     │
     ▼
Safety
     │
     ▼
Critic
     │
     ├── PASS ───────────────► Summary
     │
     └── FAIL
          │
          ▼
       Reanalysis
          │
          ├── Evidence
          ├── Differential
          ├── Missing Information
          └── Safety
```

The telemetry can capture:

```python
agent_telemetry.reanalysis_requested(
    execution,
    target="evidence",
)
```

without recording the actual clinical reasoning.

This allows you to monitor an important Agentic AI KPI:

```text
Reanalysis Rate =
number of workflows requiring reanalysis
-----------------------------------------
total workflows
```

A high reanalysis rate can indicate problems in prompts, evidence retrieval, data quality, agent outputs, or workflow design.

### Agent dashboard

This telemetry supports an operational dashboard such as:

```text id="3u8r8f"
CLINICAL AGENTIC AI OPERATIONS
─────────────────────────────────────────────

Total Agent Executions             12,481
Successful Executions              11,932
Failed Executions                     214
Timeouts                               37
Retries                               298
Reanalysis                           1,124
Human Review Required               4,831

Active Agents                           14

Average Agent Duration              1.82 sec

LLM Calls                           28,392
Input Tokens                    12,483,921
Output Tokens                    4,821,332

RAG Calls                           8,421
Tool Calls                          3,921

─────────────────────────────────────────────

AGENT                 EXEC     FAIL    AVG MS
─────────────────────────────────────────────
Intake                 2,100      8       420
Context                2,050     11       610
Evidence               2,010     29     1,820
Differential           1,980     34     2,210
Missing Information    1,970     17     1,240
Diagnostic Tests       1,950     22     1,530
Safety                 2,000     14       890
Critic                 1,960     41     1,920
Summary                1,900     38     1,110
```

This is much more useful than simply having:

```text
"Agent completed"
```

in your application logs.

### One important production recommendation

For production Kubernetes deployment, keep this API:

```text
Agent
  ↓
AgentTelemetry
  ↓
OpenTelemetry / Metrics Backend
  ↓
Prometheus / Grafana / Datadog / Azure Monitor
```

rather than making the in-memory `AgentTelemetryRegistry` your final telemetry store.

Also, **do not use `workflow_id`, `patient_id`, `request_id`, or `user_id` as metric labels**. Those are high-cardinality identifiers. Use them for log/trace correlation, while metrics should use bounded dimensions such as `agent_name`, `status`, `execution_type`, and `environment`.

Your observability architecture is therefore shaping up as:

```text id="q5x3d1"
                         Agentic Clinical AI
                                  │
                ┌─────────────────┼─────────────────┐
                │                 │                 │
                ▼                 ▼                 ▼
             Logging           Metrics           Tracing
                │                 │                 │
                └─────────────────┼─────────────────┘
                                  │
                                  ▼
                         Agent Telemetry
                                  │
               ┌──────────────────┼──────────────────┐
               ▼                  ▼                  ▼
             Agents              RAG                LLM
               │                  │                  │
               └──────────────────┼──────────────────┘
                                  ▼
                             Guardrails
                                  │
                                  ▼
                           Human Review
                                  │
                                  ▼
                              Audit Log
```

This separation is especially valuable in your healthcare system because **observability can describe how the system operated without becoming a secondary store of patient clinical data**.

"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional

from app.config.settings import settings
logger = settings.get_logger()


# ============================================================================
# ENUMS
# ============================================================================


class AgentStatus(str, Enum):
    """
    Agent execution status.
    """

    STARTED = "started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"
    REQUIRES_REVIEW = "requires_review"


class AgentExecutionType(str, Enum):
    """
    Type of Agent operation.
    """

    EXECUTION = "execution"
    RETRY = "retry"
    REANALYSIS = "reanalysis"
    VALIDATION = "validation"
    REVIEW = "review"


class AgentEventType(str, Enum):
    """
    Agent telemetry events.
    """

    EXECUTION_STARTED = "agent.execution.started"
    EXECUTION_COMPLETED = "agent.execution.completed"
    EXECUTION_FAILED = "agent.execution.failed"
    EXECUTION_TIMEOUT = "agent.execution.timeout"

    RETRY_STARTED = "agent.retry.started"
    RETRY_COMPLETED = "agent.retry.completed"

    TOOL_STARTED = "agent.tool.started"
    TOOL_COMPLETED = "agent.tool.completed"
    TOOL_FAILED = "agent.tool.failed"

    RAG_STARTED = "agent.rag.started"
    RAG_COMPLETED = "agent.rag.completed"

    LLM_STARTED = "agent.llm.started"
    LLM_COMPLETED = "agent.llm.completed"
    LLM_FAILED = "agent.llm.failed"

    REANALYSIS_STARTED = "agent.reanalysis.started"
    REANALYSIS_COMPLETED = "agent.reanalysis.completed"

    HUMAN_REVIEW_REQUIRED = "agent.human_review.required"

    GUARDRAIL_TRIGGERED = "agent.guardrail.triggered"

    STATE_UPDATED = "agent.state.updated"


class AgentOutcome(str, Enum):
    """
    Agent execution outcome.
    """

    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    BLOCKED = "blocked"
    REVIEW_REQUIRED = "review_required"


# ============================================================================
# SAFE METADATA
# ============================================================================


class AgentTelemetrySanitizer:
    """
    Sanitizes telemetry metadata.

    Telemetry should contain operational metadata only.
    """

    FORBIDDEN_KEYS = frozenset(
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
            "client_secret",
            "private_key",
            "jwt",
            "token",
            "prompt",
            "system_prompt",
            "llm_prompt",
            "messages",
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

    HASH_KEYS = frozenset(
        {
            "patient_id",
            "user_id",
            "member_id",
            "resource_id",
            "session_id",
        }
    )

    @classmethod
    def sanitize(
        cls,
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Sanitize telemetry metadata.
        """

        result: dict[str, Any] = {}

        for raw_key, raw_value in metadata.items():

            key = str(raw_key)
            normalized_key = key.lower()

            if normalized_key in cls.FORBIDDEN_KEYS:

                result[key] = "[REDACTED]"
                continue

            if normalized_key in cls.HASH_KEYS:

                result[key] = cls.hash_identifier(
                    raw_value
                )
                continue

            result[key] = cls.safe_value(
                raw_value
            )

        return result

    @classmethod
    def safe_value(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, Mapping):
            return cls.sanitize(value)

        if isinstance(value, (list, tuple)):

            return [
                cls.safe_value(item)
                for item in value
            ]

        if isinstance(value, datetime):
            return value.isoformat()

        # Do not serialize arbitrary application objects.
        return f"<{type(value).__name__}>"

    @staticmethod
    def hash_identifier(
        value: Any,
    ) -> Optional[str]:

        if value is None:
            return None

        return hashlib.sha256(
            str(value).encode("utf-8")
        ).hexdigest()[:16]


# ============================================================================
# AGENT EXECUTION RECORD
# ============================================================================


@dataclass
class AgentExecution:
    """
    Represents one Agent execution.

    This object intentionally contains operational metadata only.
    """

    execution_id: str

    agent_name: str

    workflow_id: Optional[str]

    trace_id: Optional[str]

    span_id: Optional[str]

    execution_type: AgentExecutionType

    started_at: datetime

    status: AgentStatus = AgentStatus.STARTED

    completed_at: Optional[datetime] = None

    duration_ms: Optional[float] = None

    retry_count: int = 0

    graph_iteration: Optional[int] = None

    tool_calls: int = 0

    rag_calls: int = 0

    llm_calls: int = 0

    input_tokens: int = 0

    output_tokens: int = 0

    requires_human_review: bool = False

    reanalysis_requested: bool = False

    error_type: Optional[str] = None

    outcome: Optional[AgentOutcome] = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )


# ============================================================================
# AGENT TELEMETRY REGISTRY
# ============================================================================


class AgentTelemetryRegistry:
    """
    Thread-safe registry containing aggregate Agent telemetry.

    This is a process-local implementation.

    Production recommendation:
        Export telemetry to OpenTelemetry / Prometheus / a centralized
        observability backend.

    Do not use workflow_id or patient_id as metric labels.
    """

    def __init__(self) -> None:

        self._lock = threading.RLock()

        self._executions_total = 0

        self._completed_total = 0

        self._failed_total = 0

        self._timeout_total = 0

        self._blocked_total = 0

        self._review_required_total = 0

        self._reanalysis_total = 0

        self._retry_total = 0

        self._tool_calls_total = 0

        self._rag_calls_total = 0

        self._llm_calls_total = 0

        self._input_tokens_total = 0

        self._output_tokens_total = 0

        self._active_agents = 0

        self._duration_total_ms = 0.0

        self._duration_count = 0

        self._agent_statistics: dict[
            str,
            dict[str, Any],
        ] = defaultdict(
            lambda: {
                "executions": 0,
                "completed": 0,
                "failed": 0,
                "timeouts": 0,
                "duration_total_ms": 0.0,
                "duration_count": 0,
                "tool_calls": 0,
                "rag_calls": 0,
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "retries": 0,
                "reanalysis": 0,
                "review_required": 0,
            }
        )

        logger.info(
            "AgentTelemetryRegistry initialized"
        )

    # ==================================================================
    # START
    # ==================================================================

    def record_start(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._executions_total += 1

            self._active_agents += 1

            stats = self._agent_statistics[
                agent_name
            ]

            stats["executions"] += 1

    # ==================================================================
    # COMPLETE
    # ==================================================================

    def record_complete(
        self,
        agent_name: str,
        duration_ms: float,
    ) -> None:

        with self._lock:

            self._completed_total += 1

            self._active_agents = max(
                0,
                self._active_agents - 1,
            )

            self._duration_total_ms += (
                duration_ms
            )

            self._duration_count += 1

            stats = self._agent_statistics[
                agent_name
            ]

            stats["completed"] += 1

            stats["duration_total_ms"] += (
                duration_ms
            )

            stats["duration_count"] += 1

    # ==================================================================
    # FAILED
    # ==================================================================

    def record_failure(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._failed_total += 1

            self._active_agents = max(
                0,
                self._active_agents - 1,
            )

            self._agent_statistics[
                agent_name
            ]["failed"] += 1

    # ==================================================================
    # TIMEOUT
    # ==================================================================

    def record_timeout(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._timeout_total += 1

            self._active_agents = max(
                0,
                self._active_agents - 1,
            )

            self._agent_statistics[
                agent_name
            ]["timeouts"] += 1

    # ==================================================================
    # BLOCKED
    # ==================================================================

    def record_blocked(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._blocked_total += 1

            self._active_agents = max(
                0,
                self._active_agents - 1,
            )

    # ==================================================================
    # RETRY
    # ==================================================================

    def record_retry(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._retry_total += 1

            self._agent_statistics[
                agent_name
            ]["retries"] += 1

    # ==================================================================
    # REANALYSIS
    # ==================================================================

    def record_reanalysis(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._reanalysis_total += 1

            self._agent_statistics[
                agent_name
            ]["reanalysis"] += 1

    # ==================================================================
    # REVIEW
    # ==================================================================

    def record_review_required(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._review_required_total += 1

            self._agent_statistics[
                agent_name
            ]["review_required"] += 1

    # ==================================================================
    # TOOL
    # ==================================================================

    def record_tool_call(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._tool_calls_total += 1

            self._agent_statistics[
                agent_name
            ]["tool_calls"] += 1

    # ==================================================================
    # RAG
    # ==================================================================

    def record_rag_call(
        self,
        agent_name: str,
    ) -> None:

        with self._lock:

            self._rag_calls_total += 1

            self._agent_statistics[
                agent_name
            ]["rag_calls"] += 1

    # ==================================================================
    # LLM
    # ==================================================================

    def record_llm_call(
        self,
        agent_name: str,
        *,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:

        with self._lock:

            self._llm_calls_total += 1

            self._input_tokens_total += (
                input_tokens
            )

            self._output_tokens_total += (
                output_tokens
            )

            stats = self._agent_statistics[
                agent_name
            ]

            stats["llm_calls"] += 1

            stats["input_tokens"] += (
                input_tokens
            )

            stats["output_tokens"] += (
                output_tokens
            )

    # ==================================================================
    # SNAPSHOT
    # ==================================================================

    def snapshot(self) -> dict[str, Any]:

        with self._lock:

            average_duration = (
                self._duration_total_ms
                / self._duration_count
                if self._duration_count
                else 0.0
            )

            agent_statistics = {}

            for agent_name, stats in (
                self._agent_statistics.items()
            ):

                avg_duration = (
                    stats["duration_total_ms"]
                    / stats["duration_count"]
                    if stats["duration_count"]
                    else 0.0
                )

                agent_statistics[
                    agent_name
                ] = {
                    **stats,
                    "average_duration_ms": (
                        avg_duration
                    ),
                }

            return {
                "executions_total": (
                    self._executions_total
                ),
                "completed_total": (
                    self._completed_total
                ),
                "failed_total": (
                    self._failed_total
                ),
                "timeout_total": (
                    self._timeout_total
                ),
                "blocked_total": (
                    self._blocked_total
                ),
                "review_required_total": (
                    self._review_required_total
                ),
                "reanalysis_total": (
                    self._reanalysis_total
                ),
                "retry_total": (
                    self._retry_total
                ),
                "tool_calls_total": (
                    self._tool_calls_total
                ),
                "rag_calls_total": (
                    self._rag_calls_total
                ),
                "llm_calls_total": (
                    self._llm_calls_total
                ),
                "input_tokens_total": (
                    self._input_tokens_total
                ),
                "output_tokens_total": (
                    self._output_tokens_total
                ),
                "active_agents": (
                    self._active_agents
                ),
                "average_duration_ms": (
                    average_duration
                ),
                "agents": agent_statistics,
            }

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self) -> None:

        with self._lock:

            self._executions_total = 0
            self._completed_total = 0
            self._failed_total = 0
            self._timeout_total = 0
            self._blocked_total = 0
            self._review_required_total = 0
            self._reanalysis_total = 0
            self._retry_total = 0
            self._tool_calls_total = 0
            self._rag_calls_total = 0
            self._llm_calls_total = 0
            self._input_tokens_total = 0
            self._output_tokens_total = 0
            self._active_agents = 0
            self._duration_total_ms = 0.0
            self._duration_count = 0
            self._agent_statistics.clear()

        logger.info(
            "Agent telemetry registry reset"
        )


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================


agent_telemetry_registry = (
    AgentTelemetryRegistry()
)


# ============================================================================
# AGENT TELEMETRY MANAGER
# ============================================================================


class AgentTelemetry:
    """
    High-level Agent telemetry API.

    Use this class from Agents and LangGraph nodes.

    Example
    -------

        telemetry = AgentTelemetry()

        execution = telemetry.start(
            agent_name="evidence",
            workflow_id=workflow_id,
            trace_id=trace_id,
        )

        try:
            ...
            telemetry.complete(execution)
        except Exception:
            telemetry.fail(execution)
            raise
    """

    # ==================================================================
    # START
    # ==================================================================

    def start(
        self,
        *,
        agent_name: str,
        workflow_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        execution_type: AgentExecutionType = (
            AgentExecutionType.EXECUTION
        ),
        graph_iteration: Optional[int] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> AgentExecution:

        execution = AgentExecution(
            execution_id=str(
                uuid.uuid4()
            ),
            agent_name=agent_name,
            workflow_id=workflow_id,
            trace_id=trace_id,
            span_id=span_id,
            execution_type=execution_type,
            started_at=datetime.now(
                timezone.utc
            ),
            graph_iteration=graph_iteration,
            metadata=(
                AgentTelemetrySanitizer.sanitize(
                    metadata or {}
                )
            ),
        )

        agent_telemetry_registry.record_start(
            agent_name
        )

        logger.info(
            "Agent execution started | "
            "agent={} | execution_type={} | iteration={}",
            agent_name,
            execution_type.value,
            graph_iteration,
        )

        return execution

    # ==================================================================
    # COMPLETE
    # ==================================================================

    def complete(
        self,
        execution: AgentExecution,
    ) -> AgentExecution:

        execution.completed_at = (
            datetime.now(timezone.utc)
        )

        execution.duration_ms = (
            execution.completed_at
            - execution.started_at
        ).total_seconds() * 1000.0

        execution.status = (
            AgentStatus.COMPLETED
        )

        execution.outcome = (
            AgentOutcome.SUCCESS
        )

        agent_telemetry_registry.record_complete(
            execution.agent_name,
            execution.duration_ms,
        )

        logger.info(
            "Agent execution completed | "
            "agent={} | duration_ms={:.2f} | "
            "llm_calls={} | tool_calls={} | rag_calls={}",
            execution.agent_name,
            execution.duration_ms,
            execution.llm_calls,
            execution.tool_calls,
            execution.rag_calls,
        )

        return execution

    # ==================================================================
    # FAIL
    # ==================================================================

    def fail(
        self,
        execution: AgentExecution,
        exception: Optional[
            BaseException
        ] = None,
    ) -> AgentExecution:

        execution.completed_at = (
            datetime.now(timezone.utc)
        )

        execution.duration_ms = (
            execution.completed_at
            - execution.started_at
        ).total_seconds() * 1000.0

        execution.status = (
            AgentStatus.FAILED
        )

        execution.outcome = (
            AgentOutcome.FAILURE
        )

        if exception:

            execution.error_type = (
                type(exception).__name__
            )

        agent_telemetry_registry.record_failure(
            execution.agent_name
        )

        logger.error(
            "Agent execution failed | "
            "agent={} | duration_ms={:.2f} | error_type={}",
            execution.agent_name,
            execution.duration_ms,
            execution.error_type,
        )

        return execution

    # ==================================================================
    # TIMEOUT
    # ==================================================================

    def timeout(
        self,
        execution: AgentExecution,
    ) -> AgentExecution:

        execution.completed_at = (
            datetime.now(timezone.utc)
        )

        execution.duration_ms = (
            execution.completed_at
            - execution.started_at
        ).total_seconds() * 1000.0

        execution.status = (
            AgentStatus.TIMEOUT
        )

        execution.outcome = (
            AgentOutcome.TIMEOUT
        )

        agent_telemetry_registry.record_timeout(
            execution.agent_name
        )

        logger.warning(
            "Agent execution timeout | "
            "agent={} | duration_ms={:.2f}",
            execution.agent_name,
            execution.duration_ms,
        )

        return execution

    # ==================================================================
    # BLOCK
    # ==================================================================

    def blocked(
        self,
        execution: AgentExecution,
        reason: str,
    ) -> AgentExecution:

        execution.completed_at = (
            datetime.now(timezone.utc)
        )

        execution.duration_ms = (
            execution.completed_at
            - execution.started_at
        ).total_seconds() * 1000.0

        execution.status = (
            AgentStatus.BLOCKED
        )

        execution.outcome = (
            AgentOutcome.BLOCKED
        )

        execution.metadata[
            "blocked_reason"
        ] = self._safe_reason(
            reason
        )

        agent_telemetry_registry.record_blocked(
            execution.agent_name
        )

        logger.warning(
            "Agent execution blocked | agent={}",
            execution.agent_name,
        )

        return execution

    # ==================================================================
    # RETRY
    # ==================================================================

    def retry(
        self,
        execution: AgentExecution,
        *,
        reason: Optional[str] = None,
    ) -> None:

        execution.retry_count += 1

        agent_telemetry_registry.record_retry(
            execution.agent_name
        )

        logger.warning(
            "Agent retry | "
            "agent={} | retry_count={}",
            execution.agent_name,
            execution.retry_count,
        )

    # ==================================================================
    # REANALYSIS
    # ==================================================================

    def reanalysis_requested(
        self,
        execution: AgentExecution,
        *,
        target: str,
    ) -> None:

        execution.reanalysis_requested = True

        execution.metadata[
            "reanalysis_target"
        ] = self._safe_reason(
            target
        )

        agent_telemetry_registry.record_reanalysis(
            execution.agent_name
        )

        logger.warning(
            "Agent reanalysis requested | "
            "agent={} | target={}",
            execution.agent_name,
            target,
        )

    # ==================================================================
    # HUMAN REVIEW
    # ==================================================================

    def human_review_required(
        self,
        execution: AgentExecution,
        *,
        reason: str,
    ) -> None:

        execution.requires_human_review = True

        execution.status = (
            AgentStatus.REQUIRES_REVIEW
        )

        execution.outcome = (
            AgentOutcome.REVIEW_REQUIRED
        )

        execution.metadata[
            "review_reason"
        ] = self._safe_reason(
            reason
        )

        agent_telemetry_registry.record_review_required(
            execution.agent_name
        )

        logger.warning(
            "Agent requires human review | "
            "agent={}",
            execution.agent_name,
        )

    # ==================================================================
    # TOOL
    # ==================================================================

    def tool_started(
        self,
        execution: AgentExecution,
        tool_name: str,
    ) -> None:

        execution.tool_calls += 1

        agent_telemetry_registry.record_tool_call(
            execution.agent_name
        )

        logger.debug(
            "Agent tool started | "
            "agent={} | tool={}",
            execution.agent_name,
            tool_name,
        )

    def tool_completed(
        self,
        execution: AgentExecution,
        tool_name: str,
        duration_ms: float,
    ) -> None:

        logger.debug(
            "Agent tool completed | "
            "agent={} | tool={} | duration_ms={:.2f}",
            execution.agent_name,
            tool_name,
            duration_ms,
        )

    def tool_failed(
        self,
        execution: AgentExecution,
        tool_name: str,
    ) -> None:

        logger.warning(
            "Agent tool failed | "
            "agent={} | tool={}",
            execution.agent_name,
            tool_name,
        )

    # ==================================================================
    # RAG
    # ==================================================================

    def rag_started(
        self,
        execution: AgentExecution,
    ) -> None:

        execution.rag_calls += 1

        agent_telemetry_registry.record_rag_call(
            execution.agent_name
        )

        logger.debug(
            "Agent RAG operation started | agent={}",
            execution.agent_name,
        )

    def rag_completed(
        self,
        execution: AgentExecution,
        *,
        result_count: int,
        duration_ms: float,
    ) -> None:

        logger.debug(
            "Agent RAG operation completed | "
            "agent={} | result_count={} | duration_ms={:.2f}",
            execution.agent_name,
            result_count,
            duration_ms,
        )

    # ==================================================================
    # LLM
    # ==================================================================

    def llm_started(
        self,
        execution: AgentExecution,
        model: str,
    ) -> None:

        execution.llm_calls += 1

        logger.debug(
            "Agent LLM call started | "
            "agent={} | model={}",
            execution.agent_name,
            model,
        )

    def llm_completed(
        self,
        execution: AgentExecution,
        *,
        duration_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:

        execution.input_tokens += (
            input_tokens
        )

        execution.output_tokens += (
            output_tokens
        )

        agent_telemetry_registry.record_llm_call(
            execution.agent_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        logger.debug(
            "Agent LLM call completed | "
            "agent={} | duration_ms={:.2f} | "
            "input_tokens={} | output_tokens={}",
            execution.agent_name,
            duration_ms,
            input_tokens,
            output_tokens,
        )

    def llm_failed(
        self,
        execution: AgentExecution,
        model: str,
    ) -> None:

        logger.warning(
            "Agent LLM call failed | "
            "agent={} | model={}",
            execution.agent_name,
            model,
        )

    # ==================================================================
    # STATE
    # ==================================================================

    def state_updated(
        self,
        execution: AgentExecution,
        *,
        fields_updated: int,
    ) -> None:

        logger.debug(
            "Agent state updated | "
            "agent={} | fields_updated={}",
            execution.agent_name,
            fields_updated,
        )

    # ==================================================================
    # SAFE REASON
    # ==================================================================

    @staticmethod
    def _safe_reason(
        reason: str,
    ) -> str:

        value = str(reason).strip()

        if len(value) > 200:
            value = value[:200]

        sensitive_terms = (
            "password",
            "token",
            "secret",
            "api_key",
            "clinical note",
            "diagnosis:",
            "symptoms:",
            "medications:",
            "lab result",
        )

        if any(
            term in value.lower()
            for term in sensitive_terms
        ):
            return "[REDACTED]"

        return value


# ============================================================================
# GLOBAL TELEMETRY MANAGER
# ============================================================================


agent_telemetry = AgentTelemetry()


# ============================================================================
# AGENT EXECUTION CONTEXT MANAGER
# ============================================================================


@contextmanager
def track_agent(
    *,
    agent_name: str,
    workflow_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    execution_type: AgentExecutionType = (
        AgentExecutionType.EXECUTION
    ),
    graph_iteration: Optional[int] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> Iterator[AgentExecution]:
    """
    Synchronous Agent execution tracker.

    Example
    -------

        with track_agent(
            agent_name="intake",
            workflow_id=workflow_id,
        ) as telemetry:

            result = agent.run(state)
    """

    execution = agent_telemetry.start(
        agent_name=agent_name,
        workflow_id=workflow_id,
        trace_id=trace_id,
        span_id=span_id,
        execution_type=execution_type,
        graph_iteration=graph_iteration,
        metadata=metadata,
    )

    try:

        yield execution

    except TimeoutError:

        agent_telemetry.timeout(
            execution
        )

        raise

    except Exception as exc:

        agent_telemetry.fail(
            execution,
            exc,
        )

        raise

    else:

        if (
            execution.status
            == AgentStatus.REQUIRES_REVIEW
        ):
            return

        if (
            execution.status
            == AgentStatus.BLOCKED
        ):
            return

        agent_telemetry.complete(
            execution
        )


# ============================================================================
# ASYNC AGENT EXECUTION CONTEXT
# ============================================================================


@asynccontextmanager
async def track_agent_async(
    *,
    agent_name: str,
    workflow_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    span_id: Optional[str] = None,
    execution_type: AgentExecutionType = (
        AgentExecutionType.EXECUTION
    ),
    graph_iteration: Optional[int] = None,
    metadata: Optional[
        Mapping[str, Any]
    ] = None,
) -> AsyncIterator[AgentExecution]:
    """
    Async Agent execution tracker.

    This is the recommended helper for your LangGraph Agents.
    """

    execution = agent_telemetry.start(
        agent_name=agent_name,
        workflow_id=workflow_id,
        trace_id=trace_id,
        span_id=span_id,
        execution_type=execution_type,
        graph_iteration=graph_iteration,
        metadata=metadata,
    )

    try:

        yield execution

    except TimeoutError:

        agent_telemetry.timeout(
            execution
        )

        raise

    except Exception as exc:

        agent_telemetry.fail(
            execution,
            exc,
        )

        raise

    else:

        if (
            execution.status
            in {
                AgentStatus.REQUIRES_REVIEW,
                AgentStatus.BLOCKED,
            }
        ):
            return

        agent_telemetry.complete(
            execution
        )


# ============================================================================
# SNAPSHOT
# ============================================================================


def get_agent_telemetry_snapshot() -> dict[str, Any]:
    """
    Return aggregate Agent telemetry.
    """

    return agent_telemetry_registry.snapshot()


# ============================================================================
# HEALTH
# ============================================================================


def agent_telemetry_health() -> dict[str, Any]:
    """
    Return telemetry subsystem health.
    """

    snapshot = (
        agent_telemetry_registry.snapshot()
    )

    return {
        "status": "healthy",
        "active_agents": snapshot[
            "active_agents"
        ],
        "executions_total": snapshot[
            "executions_total"
        ],
        "failed_total": snapshot[
            "failed_total"
        ],
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "AgentStatus",
    "AgentExecutionType",
    "AgentEventType",
    "AgentOutcome",
    "AgentExecution",
    "AgentTelemetryRegistry",
    "agent_telemetry_registry",
    "AgentTelemetry",
    "agent_telemetry",
    "track_agent",
    "track_agent_async",
    "get_agent_telemetry_snapshot",
    "agent_telemetry_health",
    "AgentTelemetrySanitizer",
]
```

### How this fits your existing observability stack

You now have a clean four-layer observability architecture:

```text id="qv1g4s"
app/observability/
│
├── logging.py
│      │
│      └── Application events/errors
│
├── metrics.py
│      │
│      └── System measurements
│
├── tracing.py
│      │
│      └── Distributed execution path
│
└── agent_telemetry.py
       │
       └── Agent behavior/execution
```

The distinction is:

| Component            | Primary question                                   |
| -------------------- | -------------------------------------------------- |
| `logging.py`         | **What happened?**                                 |
| `metrics.py`         | **How much/how often/how fast?**                   |
| `tracing.py`         | **Where did the request/workflow spend time?**     |
| `agent_telemetry.py` | **How did each Agent execute?**                    |
| `security/audit.py`  | **Who performed what security/compliance action?** |

### Example: integrating with your `EvidenceAgent`

Your existing Evidence Agent can use:

```python
async def node(
    self,
    state: ClinicalState,
) -> dict:

    with track_agent_async(
        agent_name="evidence",
        workflow_id=state["workflow_id"],
        graph_iteration=state.get(
            "graph_iteration",
            0,
        ),
    ) as execution:

        result = await self.run(
            state
        )

        return result
```

Conceptually:

```text id="xk8c1h"
EvidenceAgent
      │
      ├── Agent started
      │
      ├── LLM call
      │
      ├── RAG retrieval
      │      │
      │      └── Vector DB
      │
      ├── LLM call
      │
      └── Agent completed
```

### Tool telemetry

For an Agent using a clinical knowledge-search tool:

```python
execution = agent_telemetry.start(
    agent_name="evidence",
    workflow_id=workflow_id,
)

try:

    agent_telemetry.tool_started(
        execution,
        "clinical_knowledge_search",
    )

    started = time.perf_counter()

    documents = await tool.ainvoke(
        query
    )

    duration_ms = (
        time.perf_counter() - started
    ) * 1000

    agent_telemetry.tool_completed(
        execution,
        "clinical_knowledge_search",
        duration_ms,
    )

finally:

    agent_telemetry.complete(
        execution
    )