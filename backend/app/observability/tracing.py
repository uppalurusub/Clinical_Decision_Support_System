
"""
app/observability/tracing.py

Distributed tracing layer for the Clinical Decision Support &
Diagnosis Agentic AI system.

Responsibilities
----------------
1. Create and manage distributed trace IDs.
2. Create spans for requests, workflows, agents, RAG, LLMs and tools.
3. Correlate traces with Loguru logs.
4. Propagate trace/span context through async execution.
5. Record timing and status.
6. Provide safe tracing metadata.
7. Support OpenTelemetry integration.
8. Keep PHI and secrets out of tracing attributes.

Architecture
------------

FastAPI
   |
   +-- HTTP Span
          |
          +-- Service Span
                 |
                 +-- Clinical Workflow Span
                        |
                        +-- Intake Agent Span
                        |
                        +-- Context Agent Span
                        |
                        +-- Evidence Agent Span
                        |       |
                        |       +-- RAG Retrieval Span
                        |
                        +-- Differential Agent Span
                        |
                        +-- Missing Information Span
                        |
                        +-- Diagnostic Test Span
                        |
                        +-- Safety Agent Span
                        |
                        +-- Critic Agent Span
                        |
                        +-- Summary Agent Span
                                |
                                +-- Human Review

Important
---------
This module intentionally does NOT trace or store:

    - patient names
    - DOB
    - MRN
    - addresses
    - phone numbers
    - email addresses
    - clinical notes
    - symptoms
    - diagnoses
    - medications
    - laboratory values
    - imaging content
    - raw prompts
    - raw LLM responses
    - access tokens
    - passwords
    - API keys
    - secrets
    - embeddings

Only operational metadata should be attached to spans.

Production recommendation
-------------------------
Use OpenTelemetry as the tracing API and export traces to:

    - Jaeger
    - Grafana Tempo
    - Azure Application Insights
    - Datadog
    - AWS X-Ray
    - OTLP-compatible tracing backend

Loguru remains the operational logging layer.

```

### 1. How tracing differs from your `logging.py` and `metrics.py`

You now have three distinct observability capabilities:

```text
┌──────────────────────────────────────────────────────────┐
│                    OBSERVABILITY                         │
├───────────────────┬───────────────────┬──────────────────┤
│ Logging           │ Metrics           │ Tracing          │
├───────────────────┼───────────────────┼──────────────────┤
│ What happened?    │ How much?         │ Where/why?       │
│                   │ How often?        │                  │
│                   │ How fast?         │                  │
├───────────────────┼───────────────────┼──────────────────┤
│ Log messages      │ Counters          │ Trace            │
│ Exceptions        │ Gauges            │ Spans            │
│ Events            │ Histograms        │ Parent/child     │
└───────────────────┴───────────────────┴──────────────────┘
```

For example:

```text
Request
  │
  │ Trace ID = T123
  ▼
HTTP Span
  │
  ▼
Service Span
  │
  ▼
Clinical Workflow Span
  │
  ├──────── Intake Span
  │
  ├──────── Context Span
  │
  ├──────── Evidence Span
  │             │
  │             └── RAG Retrieval Span
  │
  ├──────── Differential Span
  │
  ├──────── Missing Information Span
  │
  ├──────── Diagnostic Test Span
  │
  ├──────── Safety Span
  │
  ├──────── Critic Span
  │
  └──────── Summary Span
```

This is particularly valuable for your **LangGraph multi-agent workflow**, because you can see exactly where time was spent.

### 2. Recommended tracing attributes

Safe:

```python
{
    "agent.name": "evidence",
    "workflow.name": "clinical_decision",
    "llm.model": "gpt-4o",
    "llm.provider": "openai",
    "rag.operation": "retrieve",
    "retriever.name": "clinical_guideline_retriever",
    "tool.name": "clinical_knowledge_search",
}
```

Avoid:

```python
# ❌ Never

{
    "patient_name": "...",
    "patient_id": "...",
    "symptoms": "...",
    "diagnosis": "...",
    "clinical_notes": "...",
    "prompt": "...",
    "response": "...",
}
```

The implementation hashes identifiers such as `patient_id` rather than putting their raw values into tracing metadata.

### 3. LangGraph integration

Your `ClinicalGraph` can instrument each node:

```python
async def _intake_node(
    self,
    state: ClinicalState,
) -> dict:

    async with trace_agent("intake"):

        return await self.intake_agent.node(
            state
        )
```

The same pattern applies to the other nodes:

```text
ClinicalGraph
    │
    ├── trace_agent("intake")
    ├── trace_agent("context")
    ├── trace_agent("evidence")
    ├── trace_agent("differential")
    ├── trace_agent("missing_information")
    ├── trace_agent("diagnostic_tests")
    ├── trace_agent("safety")
    ├── trace_agent("critic")
    └── trace_agent("summary")
```

### 4. RAG tracing

Your Evidence Agent can trace the retrieval separately:

```python
async with trace_agent("evidence"):

    async with trace_rag("clinical_evidence"):

        async with trace_retrieval(
            "clinical_guideline_retriever"
        ):
            documents = await retriever.ainvoke(
                query
            )
```

The resulting trace becomes:

```text
Evidence Agent
      │
      ▼
RAG
      │
      ▼
Retriever
      │
      ▼
Vector Database
```

You can then determine whether latency came from the agent, RAG orchestration, or retrieval.

### 5. LLM tracing

For your LLM wrapper:

```python
async with trace_llm(
    model_name="gpt-4o",
    provider="openai",
):
    response = await llm.ainvoke(
        messages
    )
```

Do **not** attach:

```python
prompt
response
messages
clinical_context
```

to the span.

Instead record operational information such as:

```text
llm.model
llm.provider
duration
success/failure
token counts
```

Token counts are operational telemetry, although you should still apply your organization's retention/access policies.

### 6. The complete observability flow

Your current architecture can now become:

```text
                         FastAPI
                            │
                            ▼
                     ┌────────────┐
                     │   Router   │
                     └─────┬──────┘
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
              Logging   Metrics   Tracing
                 │         │         │
                 └─────────┼─────────┘
                           ▼
                        Service
                           │
                           ▼
                     Implementation
                           │
                           ▼
                       LangGraph
                           │
       ┌───────────────────┼───────────────────┐
       ▼                   ▼                   ▼
     Agents              RAG                  LLM
       │                   │                   │
       └───────────────────┼───────────────────┘
                           ▼
                      Guardrails
                           │
                           ▼
                     Human Review
                           │
                           ▼
                         Audit
```

The final distinction is important:

```text
logging.py
    → operational diagnostics

metrics.py
    → quantitative system health

tracing.py
    → distributed execution path

security/audit.py
    → security/compliance record
```

For production, I would use **OpenTelemetry as the actual distributed tracing implementation** and export OTLP traces to your selected backend. The `TracingManager` interface above lets you instrument the application now without coupling your LangGraph agents directly to a particular tracing vendor.

"""

from __future__ import annotations

import contextvars
import hashlib
import os
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator, Dict, Iterator, Mapping, Optional

from app.config.settings import settings
logger = settings.get_logger()


# ============================================================================
# OPTIONAL OPENTELEMETRY IMPORTS
# ============================================================================

try:
    from opentelemetry import trace
    from opentelemetry.trace import (
        Span,
        Status,
        StatusCode,
    )

    OPENTELEMETRY_AVAILABLE = True

except ImportError:
    trace = None
    Span = Any
    Status = None
    StatusCode = None

    OPENTELEMETRY_AVAILABLE = False


# ============================================================================
# ENUMS
# ============================================================================


class SpanType(str, Enum):
    """
    Logical span categories.
    """

    HTTP = "http"
    SERVICE = "service"
    WORKFLOW = "workflow"
    AGENT = "agent"
    LLM = "llm"
    RAG = "rag"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    DATABASE = "database"
    EXTERNAL_API = "external_api"
    GUARDRAIL = "guardrail"
    HUMAN_REVIEW = "human_review"


class TraceStatus(str, Enum):
    """
    Application-level trace status.
    """

    OK = "ok"
    ERROR = "error"
    UNSET = "unset"


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================


trace_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "trace_id",
    default=None,
)

span_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "span_id",
    default=None,
)

parent_span_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "parent_span_id",
    default=None,
)


# ============================================================================
# SAFE TRACE ATTRIBUTES
# ============================================================================


class TraceAttributeSanitizer:
    """
    Prevent sensitive data from being attached to spans.

    Trace backends often retain data for long periods, therefore
    clinical data should never be attached to span attributes.
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
        attributes: Mapping[str, Any],
    ) -> dict[str, Any]:
        """
        Sanitize trace attributes.
        """

        result: dict[str, Any] = {}

        for raw_key, raw_value in attributes.items():

            key = str(raw_key)
            normalized = key.lower()

            if normalized in cls.FORBIDDEN_KEYS:

                result[key] = "[REDACTED]"
                continue

            if normalized in cls.HASH_KEYS:

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

        if isinstance(value, (str, int, float, bool)):
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

        # Do not serialize arbitrary objects.
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
# TRACE CONFIGURATION
# ============================================================================


@dataclass
class TracingConfig:
    """
    Distributed tracing configuration.
    """

    enabled: bool = field(
        default_factory=lambda: (
            os.getenv(
                "TRACING_ENABLED",
                "true",
            ).lower()
            in {"true", "1", "yes", "on"}
        )
    )

    service_name: str = field(
        default_factory=lambda: os.getenv(
            "OTEL_SERVICE_NAME",
            "clinical-decision-support",
        )
    )

    environment: str = field(
        default_factory=lambda: os.getenv(
            "ENVIRONMENT",
            "development",
        )
    )

    otlp_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv(
            "OTEL_EXPORTER_OTLP_ENDPOINT"
        )
    )

    sample_rate: float = field(
        default_factory=lambda: float(
            os.getenv(
                "OTEL_TRACE_SAMPLE_RATE",
                "1.0",
            )
        )
    )


# ============================================================================
# SIMPLE SPAN
# ============================================================================


@dataclass
class LocalSpan:
    """
    Lightweight local span representation.

    Used when OpenTelemetry is not installed.

    This makes the tracing API usable during local development and
    testing while allowing production to use OpenTelemetry.
    """

    name: str

    span_type: SpanType

    trace_id: str

    span_id: str

    parent_span_id: Optional[str]

    started_at: float

    attributes: dict[str, Any] = field(
        default_factory=dict
    )

    status: TraceStatus = TraceStatus.UNSET

    error: Optional[str] = None

    ended_at: Optional[float] = None

    def set_attribute(
        self,
        key: str,
        value: Any,
    ) -> None:

        self.attributes[key] = (
            TraceAttributeSanitizer.safe_value(
                value
            )
        )

    def set_attributes(
        self,
        attributes: Mapping[str, Any],
    ) -> None:

        self.attributes.update(
            TraceAttributeSanitizer.sanitize(
                attributes
            )
        )

    def record_exception(
        self,
        exception: BaseException,
    ) -> None:

        self.error = (
            type(exception).__name__
        )

        self.status = TraceStatus.ERROR

    def set_status(
        self,
        status: TraceStatus,
    ) -> None:

        self.status = status

    def end(self) -> None:

        if self.ended_at is None:
            self.ended_at = time.perf_counter()

    @property
    def duration_ms(self) -> float:

        end = (
            self.ended_at
            if self.ended_at is not None
            else time.perf_counter()
        )

        return (
            end - self.started_at
        ) * 1000.0


# ============================================================================
# TRACING MANAGER
# ============================================================================


class TracingManager:
    """
    Central tracing manager.

    OpenTelemetry is preferred when installed.

    Otherwise a lightweight local span implementation is used.
    """

    def __init__(
        self,
        config: Optional[TracingConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else TracingConfig()
        )

        self._tracer = None

        if (
            self.config.enabled
            and OPENTELEMETRY_AVAILABLE
        ):

            self._tracer = trace.get_tracer(
                self.config.service_name
            )

        logger.info(
            "TracingManager initialized | "
            "enabled={} | opentelemetry={}",
            self.config.enabled,
            OPENTELEMETRY_AVAILABLE,
        )

    # ==================================================================
    # TRACE ID
    # ==================================================================

    def new_trace_id(self) -> str:
        """
        Generate a new application-level trace ID.
        """

        return uuid.uuid4().hex

    def new_span_id(self) -> str:
        """
        Generate a new application-level span ID.
        """

        return uuid.uuid4().hex[:16]

    # ==================================================================
    # START SPAN
    # ==================================================================

    def start_span(
        self,
        name: str,
        *,
        span_type: SpanType = SpanType.SERVICE,
        attributes: Optional[
            Mapping[str, Any]
        ] = None,
    ):
        """
        Start a span.

        Returns either:
            OpenTelemetry Span
        or:
            LocalSpan
        """

        safe_attributes = (
            TraceAttributeSanitizer.sanitize(
                attributes or {}
            )
        )

        if (
            self._tracer is not None
            and self.config.enabled
        ):

            span = self._tracer.start_span(
                name=name,
            )

            span.set_attribute(
                "span.type",
                span_type.value,
            )

            span.set_attribute(
                "service.name",
                self.config.service_name,
            )

            for key, value in safe_attributes.items():

                try:

                    span.set_attribute(
                        key,
                        value,
                    )

                except Exception:

                    logger.debug(
                        "Unable to set trace attribute | key={}",
                        key,
                    )

            context_trace_id = (
                self.current_trace_id()
            )

            if context_trace_id:

                span.set_attribute(
                    "app.trace_id",
                    context_trace_id,
                )

            return span

        # --------------------------------------------------------------
        # Local fallback
        # --------------------------------------------------------------

        current_trace_id = (
            self.current_trace_id()
            or self.new_trace_id()
        )

        current_span_id = (
            self.current_span_id()
        )

        span = LocalSpan(
            name=name,
            span_type=span_type,
            trace_id=current_trace_id,
            span_id=self.new_span_id(),
            parent_span_id=current_span_id,
            started_at=time.perf_counter(),
            attributes=safe_attributes,
        )

        return span

    # ==================================================================
    # CURRENT TRACE
    # ==================================================================

    @staticmethod
    def current_trace_id() -> Optional[str]:

        if (
            OPENTELEMETRY_AVAILABLE
            and trace is not None
        ):

            try:

                current_span = (
                    trace.get_current_span()
                )

                span_context = (
                    current_span.get_span_context()
                )

                if span_context.is_valid:

                    return format(
                        span_context.trace_id,
                        "032x",
                    )

            except Exception:

                pass

        return trace_id_context.get()

    @staticmethod
    def current_span_id() -> Optional[str]:

        if (
            OPENTELEMETRY_AVAILABLE
            and trace is not None
        ):

            try:

                current_span = (
                    trace.get_current_span()
                )

                span_context = (
                    current_span.get_span_context()
                )

                if span_context.is_valid:

                    return format(
                        span_context.span_id,
                        "016x",
                    )

            except Exception:

                pass

        return span_id_context.get()

    # ==================================================================
    # TRACE CONTEXT
    # ==================================================================

    def set_trace_context(
        self,
        trace_id: str,
    ) -> None:

        trace_id_context.set(
            trace_id
        )

    def set_span_context(
        self,
        span_id: str,
    ) -> None:

        span_id_context.set(
            span_id
        )

    # ==================================================================
    # LOG CONTEXT
    # ==================================================================

    def bind_log_context(
        self,
    ):
        """
        Bind trace information to Loguru.

        Example:

            tracing.bind_log_context()

            logger.info(
                "Processing clinical workflow"
            )
        """

        trace_id = (
            self.current_trace_id()
            or "-"
        )

        span_id = (
            self.current_span_id()
            or "-"
        )

        return logger.bind(
            trace_id=trace_id,
            span_id=span_id,
        )

    # ==================================================================
    # SPAN STATUS
    # ==================================================================

    @staticmethod
    def set_span_ok(
        span: Any,
    ) -> None:

        if isinstance(
            span,
            LocalSpan,
        ):

            span.set_status(
                TraceStatus.OK
            )

            return

        if (
            OPENTELEMETRY_AVAILABLE
            and Status is not None
        ):

            try:

                span.set_status(
                    Status(
                        StatusCode.OK
                    )
                )

            except Exception:

                pass

    @staticmethod
    def set_span_error(
        span: Any,
        exception: Optional[
            BaseException
        ] = None,
    ) -> None:

        if isinstance(
            span,
            LocalSpan,
        ):

            if exception:

                span.record_exception(
                    exception
                )

            span.set_status(
                TraceStatus.ERROR
            )

            return

        if (
            OPENTELEMETRY_AVAILABLE
            and Status is not None
        ):

            try:

                if exception:

                    span.record_exception(
                        exception
                    )

                span.set_status(
                    Status(
                        StatusCode.ERROR
                    )
                )

            except Exception:

                pass

    # ==================================================================
    # END SPAN
    # ==================================================================

    def end_span(
        self,
        span: Any,
    ) -> None:

        try:

            span.end()

        except Exception:

            logger.debug(
                "Unable to end tracing span"
            )


# ============================================================================
# GLOBAL MANAGER
# ============================================================================


tracing_manager = TracingManager()


# ============================================================================
# SPAN CONTEXT MANAGER
# ============================================================================


@contextmanager
def trace_span(
    name: str,
    *,
    span_type: SpanType = SpanType.SERVICE,
    attributes: Optional[
        Mapping[str, Any]
    ] = None,
) -> Iterator[Any]:
    """
    Synchronous tracing context.

    Example
    -------

        with trace_span(
            "clinical_decision.analyze",
            span_type=SpanType.WORKFLOW,
        ):
            result = implementation.analyze(request)
    """

    span = tracing_manager.start_span(
        name,
        span_type=span_type,
        attributes=attributes,
    )

    started = time.perf_counter()

    try:

        yield span

        tracing_manager.set_span_ok(
            span
        )

    except Exception as exc:

        tracing_manager.set_span_error(
            span,
            exc,
        )

        logger.exception(
            "Trace span failed | span={} | duration_ms={:.2f}",
            name,
            (
                time.perf_counter()
                - started
            )
            * 1000.0,
        )

        raise

    finally:

        duration_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        logger.debug(
            "Trace span completed | "
            "span={} | type={} | duration_ms={:.2f}",
            name,
            span_type.value,
            duration_ms,
        )

        tracing_manager.end_span(
            span
        )


# ============================================================================
# ASYNC SPAN CONTEXT MANAGER
# ============================================================================


@asynccontextmanager
async def async_trace_span(
    name: str,
    *,
    span_type: SpanType = SpanType.SERVICE,
    attributes: Optional[
        Mapping[str, Any]
    ] = None,
) -> AsyncIterator[Any]:
    """
    Async tracing context.

    This is the primary tracing helper for FastAPI,
    LangGraph and async Agent implementations.

    Example
    -------

        async with async_trace_span(
            "clinical_graph",
            span_type=SpanType.WORKFLOW,
        ):
            result = await graph.ainvoke(state)
    """

    span = tracing_manager.start_span(
        name,
        span_type=span_type,
        attributes=attributes,
    )

    started = time.perf_counter()

    try:

        yield span

        tracing_manager.set_span_ok(
            span
        )

    except Exception as exc:

        tracing_manager.set_span_error(
            span,
            exc,
        )

        logger.exception(
            "Async trace span failed | "
            "span={} | duration_ms={:.2f}",
            name,
            (
                time.perf_counter()
                - started
            )
            * 1000.0,
        )

        raise

    finally:

        duration_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        logger.debug(
            "Async trace span completed | "
            "span={} | type={} | duration_ms={:.2f}",
            name,
            span_type.value,
            duration_ms,
        )

        tracing_manager.end_span(
            span
        )


# ============================================================================
# SPECIALIZED TRACE HELPERS
# ============================================================================


@asynccontextmanager
async def trace_workflow(
    workflow_name: str,
    *,
    workflow_id: Optional[str] = None,
) -> AsyncIterator[Any]:
    """
    Trace a clinical workflow.
    """

    attributes = {
        "workflow.name": workflow_name,
    }

    if workflow_id:
        attributes["workflow.id"] = (
            TraceAttributeSanitizer.hash_identifier(
                workflow_id
            )
        )

    async with async_trace_span(
        f"workflow.{workflow_name}",
        span_type=SpanType.WORKFLOW,
        attributes=attributes,
    ) as span:

        yield span


@asynccontextmanager
async def trace_agent(
    agent_name: str,
) -> AsyncIterator[Any]:
    """
    Trace an Agentic AI agent.
    """

    async with async_trace_span(
        f"agent.{agent_name}",
        span_type=SpanType.AGENT,
        attributes={
            "agent.name": agent_name,
        },
    ) as span:

        yield span


@asynccontextmanager
async def trace_llm(
    model_name: str,
    *,
    provider: Optional[str] = None,
) -> AsyncIterator[Any]:
    """
    Trace an LLM invocation.

    Never attach the prompt or response to this span.
    """

    attributes = {
        "llm.model": model_name,
    }

    if provider:
        attributes["llm.provider"] = provider

    async with async_trace_span(
        "llm.invoke",
        span_type=SpanType.LLM,
        attributes=attributes,
    ) as span:

        yield span


@asynccontextmanager
async def trace_rag(
    operation: str,
) -> AsyncIterator[Any]:
    """
    Trace a RAG operation.
    """

    async with async_trace_span(
        f"rag.{operation}",
        span_type=SpanType.RAG,
        attributes={
            "rag.operation": operation,
        },
    ) as span:

        yield span


@asynccontextmanager
async def trace_retrieval(
    retriever_name: str,
) -> AsyncIterator[Any]:
    """
    Trace vector/database retrieval.
    """

    async with async_trace_span(
        "rag.retrieve",
        span_type=SpanType.RETRIEVAL,
        attributes={
            "retriever.name": retriever_name,
        },
    ) as span:

        yield span


@asynccontextmanager
async def trace_tool(
    tool_name: str,
) -> AsyncIterator[Any]:
    """
    Trace an agent tool invocation.

    Do not attach tool arguments or results when they may contain
    clinical information.
    """

    async with async_trace_span(
        f"tool.{tool_name}",
        span_type=SpanType.TOOL,
        attributes={
            "tool.name": tool_name,
        },
    ) as span:

        yield span


@asynccontextmanager
async def trace_guardrail(
    guardrail_name: str,
) -> AsyncIterator[Any]:
    """
    Trace guardrail execution.
    """

    async with async_trace_span(
        f"guardrail.{guardrail_name}",
        span_type=SpanType.GUARDRAIL,
        attributes={
            "guardrail.name": guardrail_name,
        },
    ) as span:

        yield span


@asynccontextmanager
async def trace_human_review(
    review_type: str,
) -> AsyncIterator[Any]:
    """
    Trace clinician/human review.
    """

    async with async_trace_span(
        "human_review",
        span_type=SpanType.HUMAN_REVIEW,
        attributes={
            "review.type": review_type,
        },
    ) as span:

        yield span


# ============================================================================
# TRACE ID HELPERS
# ============================================================================


def get_trace_id() -> Optional[str]:
    """
    Return the current trace ID.
    """

    return tracing_manager.current_trace_id()


def get_span_id() -> Optional[str]:
    """
    Return the current span ID.
    """

    return tracing_manager.current_span_id()


def set_trace_id(
    trace_id: str,
) -> None:
    """
    Explicitly set application trace ID.
    """

    trace_id_context.set(
        trace_id
    )


# ============================================================================
# TRACE HEALTH
# ============================================================================


def tracing_health() -> dict[str, Any]:
    """
    Return non-sensitive tracing health information.
    """

    return {
        "status": (
            "healthy"
            if tracing_manager.config.enabled
            else "disabled"
        ),
        "enabled": (
            tracing_manager.config.enabled
        ),
        "opentelemetry_available": (
            OPENTELEMETRY_AVAILABLE
        ),
        "service_name": (
            tracing_manager.config.service_name
        ),
        "environment": (
            tracing_manager.config.environment
        ),
        "otlp_configured": bool(
            tracing_manager.config.otlp_endpoint
        ),
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "SpanType",
    "TraceStatus",
    "TracingConfig",
    "LocalSpan",
    "TracingManager",
    "tracing_manager",
    "trace_span",
    "async_trace_span",
    "trace_workflow",
    "trace_agent",
    "trace_llm",
    "trace_rag",
    "trace_retrieval",
    "trace_tool",
    "trace_guardrail",
    "trace_human_review",
    "get_trace_id",
    "get_span_id",
    "set_trace_id",
    "tracing_health",
    "TraceAttributeSanitizer",
]