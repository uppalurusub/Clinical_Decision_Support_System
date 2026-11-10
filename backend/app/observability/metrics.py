"""
app/observability/metrics.py

Application metrics and measurement layer for the Clinical Decision
Support & Diagnosis system.

Responsibilities
----------------
1. Track application metrics.
2. Track HTTP request metrics.
3. Track clinical workflow metrics.
4. Track Agent execution metrics.
5. Track RAG retrieval metrics.
6. Track guardrail and safety metrics.
7. Track human-review metrics.
8. Track LLM operational metrics.
9. Provide metric snapshots for health/observability endpoints.
10. Use Loguru for operational metric logging.

Important
---------
This module measures SYSTEM BEHAVIOR.

It is NOT:
    - a security audit system
    - a clinical decision engine
    - a patient-risk scoring engine
    - an LLM evaluation framework

Never record:
    - patient names
    - DOB
    - MRN
    - clinical notes
    - symptoms
    - diagnoses
    - medication details
    - laboratory values
    - imaging content
    - prompts
    - LLM responses
    - access tokens
    - passwords
    - API keys
    - secrets

Metric labels must have LOW CARDINALITY.

Do NOT use:
    patient_id
    workflow_id
    request_id
    user_id

as Prometheus-style metric labels.

These identifiers belong in logs/traces/audit records, not metric labels.

```

Even better, because you already have the centralized logging/context architecture:

```text
Router
   │
   └── request_id
          │
          ▼
Service
   │
   └── workflow_id
          │
          ▼
ClinicalGraph
   │
   ├── agent_name=intake
   ├── agent_name=context
   ├── agent_name=evidence
   ├── agent_name=differential
   ├── agent_name=missing_information
   ├── agent_name=diagnostic_tests
   ├── agent_name=safety
   ├── agent_name=critic
   └── agent_name=summary
```

This gives you the three complementary observability dimensions:

```text
                    Clinical AI System
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
       Logging           Metrics          Tracing
          │                │                │
          │                │                │
       What             How much         Where/Why
       happened?        /How fast?       did it spend time?
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                    Operational Insight
```

One architectural recommendation: this in-process registry is excellent as a **starter/reference implementation**, but for your production Kubernetes deployment I would eventually replace the registry backend with **OpenTelemetry/Prometheus-compatible metrics** while keeping the `ApplicationMetrics` interface unchanged. That way your agents and LangGraph code do not need to change when the metrics backend changes.

"""

from __future__ import annotations

import asyncio
import threading
import time
from collections import defaultdict
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Any, Dict, Iterator, Optional

from app.config.settings import settings
logger = settings.get_logger()


# ============================================================================
# ENUMS
# ============================================================================


class MetricType(str, Enum):
    """
    Logical metric types.
    """

    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class MetricCategory(str, Enum):
    """
    Metric categories.
    """

    HTTP = "http"
    WORKFLOW = "workflow"
    AGENT = "agent"
    RAG = "rag"
    LLM = "llm"
    GUARDRAIL = "guardrail"
    SAFETY = "safety"
    HUMAN_REVIEW = "human_review"
    SYSTEM = "system"


# ============================================================================
# METRIC DATA STRUCTURES
# ============================================================================


@dataclass
class CounterMetric:
    """
    Monotonically increasing counter.
    """

    name: str
    description: str
    value: float = 0.0


@dataclass
class GaugeMetric:
    """
    Current-value metric.
    """

    name: str
    description: str
    value: float = 0.0


@dataclass
class HistogramMetric:
    """
    Simple in-process histogram representation.

    Production deployments should generally export these values to
    Prometheus/OpenTelemetry rather than relying on this in-memory
    implementation as the authoritative metrics backend.
    """

    name: str
    description: str
    count: int = 0
    total: float = 0.0
    minimum: Optional[float] = None
    maximum: Optional[float] = None

    def observe(
        self,
        value: float,
    ) -> None:

        self.count += 1
        self.total += value

        if (
            self.minimum is None
            or value < self.minimum
        ):
            self.minimum = value

        if (
            self.maximum is None
            or value > self.maximum
        ):
            self.maximum = value

    @property
    def average(self) -> float:

        if self.count == 0:
            return 0.0

        return self.total / self.count


# ============================================================================
# METRIC REGISTRY
# ============================================================================


class MetricsRegistry:
    """
    Thread-safe in-process metric registry.

    This registry is useful for:
        - local development
        - testing
        - basic operational visibility
        - exposing a snapshot endpoint

    For production distributed systems, export metrics to:
        - Prometheus
        - OpenTelemetry
        - Azure Monitor
        - CloudWatch
        - Datadog
        - Grafana ecosystem

    Important:
        Multiple Kubernetes replicas each have their own process-local
        registry. Therefore, process-local values are not a replacement
        for a centralized metrics backend.
    """

    def __init__(self) -> None:

        self._counters: dict[
            str,
            CounterMetric,
        ] = {}

        self._gauges: dict[
            str,
            GaugeMetric,
        ] = {}

        self._histograms: dict[
            str,
            HistogramMetric,
        ] = {}

        self._lock = threading.RLock()

        logger.info(
            "MetricsRegistry initialized"
        )

    # ==================================================================
    # COUNTERS
    # ==================================================================

    def increment_counter(
        self,
        name: str,
        *,
        description: str = "",
        value: float = 1.0,
    ) -> float:

        with self._lock:

            metric = self._counters.get(
                name
            )

            if metric is None:

                metric = CounterMetric(
                    name=name,
                    description=description,
                )

                self._counters[name] = metric

            metric.value += value

            return metric.value

    # ==================================================================
    # GAUGES
    # ==================================================================

    def set_gauge(
        self,
        name: str,
        value: float,
        *,
        description: str = "",
    ) -> float:

        with self._lock:

            metric = self._gauges.get(
                name
            )

            if metric is None:

                metric = GaugeMetric(
                    name=name,
                    description=description,
                )

                self._gauges[name] = metric

            metric.value = value

            return metric.value

    def increment_gauge(
        self,
        name: str,
        *,
        value: float = 1.0,
        description: str = "",
    ) -> float:

        with self._lock:

            metric = self._gauges.get(
                name
            )

            if metric is None:

                metric = GaugeMetric(
                    name=name,
                    description=description,
                )

                self._gauges[name] = metric

            metric.value += value

            return metric.value

    def decrement_gauge(
        self,
        name: str,
        *,
        value: float = 1.0,
        description: str = "",
    ) -> float:

        return self.increment_gauge(
            name,
            value=-value,
            description=description,
        )

    # ==================================================================
    # HISTOGRAMS
    # ==================================================================

    def observe_histogram(
        self,
        name: str,
        value: float,
        *,
        description: str = "",
    ) -> HistogramMetric:

        with self._lock:

            metric = self._histograms.get(
                name
            )

            if metric is None:

                metric = HistogramMetric(
                    name=name,
                    description=description,
                )

                self._histograms[name] = metric

            metric.observe(value)

            return metric

    # ==================================================================
    # GET
    # ==================================================================

    def get_counter(
        self,
        name: str,
    ) -> Optional[CounterMetric]:

        with self._lock:
            return self._counters.get(name)

    def get_gauge(
        self,
        name: str,
    ) -> Optional[GaugeMetric]:

        with self._lock:
            return self._gauges.get(name)

    def get_histogram(
        self,
        name: str,
    ) -> Optional[HistogramMetric]:

        with self._lock:
            return self._histograms.get(name)

    # ==================================================================
    # SNAPSHOT
    # ==================================================================

    def snapshot(self) -> dict[str, Any]:
        """
        Return a safe metrics snapshot.
        """

        with self._lock:

            return {
                "counters": {
                    name: {
                        "description": metric.description,
                        "value": metric.value,
                    }
                    for name, metric
                    in self._counters.items()
                },
                "gauges": {
                    name: {
                        "description": metric.description,
                        "value": metric.value,
                    }
                    for name, metric
                    in self._gauges.items()
                },
                "histograms": {
                    name: {
                        "description": metric.description,
                        "count": metric.count,
                        "total": metric.total,
                        "minimum": metric.minimum,
                        "maximum": metric.maximum,
                        "average": metric.average,
                    }
                    for name, metric
                    in self._histograms.items()
                },
            }

    # ==================================================================
    # RESET
    # ==================================================================

    def reset(self) -> None:
        """
        Reset all in-process metrics.

        Primarily useful for testing.
        """

        with self._lock:

            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()

        logger.info(
            "MetricsRegistry reset"
        )


# ============================================================================
# GLOBAL REGISTRY
# ============================================================================


metrics_registry = MetricsRegistry()


# ============================================================================
# APPLICATION METRICS
# ============================================================================


class ApplicationMetrics:
    """
    High-level metrics API for the clinical AI application.

    Application code should preferably use this class rather than
    directly manipulating the registry.
    """

    # ==================================================================
    # HTTP
    # ==================================================================

    def http_request(
        self,
        *,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
    ) -> None:
        """
        Record HTTP request metrics.

        Note:
            Path must be a route template where possible.

        Good:
            /api/v1/clinical-decision/analyze

        Avoid:
            /api/v1/patients/123456/analyze
        """

        self.increment(
            "http_requests_total",
            description="Total HTTP requests",
        )

        self.increment(
            f"http_requests_status_{status_code}_total",
            description="HTTP requests by status code",
        )

        self.observe(
            "http_request_duration_ms",
            duration_ms,
            description="HTTP request duration in milliseconds",
        )

        if status_code >= 500:

            self.increment(
                "http_server_errors_total",
                description="HTTP 5xx responses",
            )

        elif status_code >= 400:

            self.increment(
                "http_client_errors_total",
                description="HTTP 4xx responses",
            )

        logger.debug(
            "HTTP metrics recorded | "
            "method={} | path={} | status={} | duration_ms={:.2f}",
            method,
            path,
            status_code,
            duration_ms,
        )

    # ==================================================================
    # WORKFLOW
    # ==================================================================

    def workflow_started(
        self,
        workflow_name: str,
    ) -> None:

        self.increment(
            "clinical_workflows_started_total",
            description="Clinical workflows started",
        )

        self.increment(
            f"workflow_{self._safe_name(workflow_name)}_started_total",
            description="Workflow starts by workflow type",
        )

    def workflow_completed(
        self,
        workflow_name: str,
        duration_ms: float,
    ) -> None:

        self.increment(
            "clinical_workflows_completed_total",
            description="Clinical workflows completed",
        )

        self.observe(
            "clinical_workflow_duration_ms",
            duration_ms,
            description="Clinical workflow duration",
        )

        logger.info(
            "Workflow metrics | "
            "workflow={} | duration_ms={:.2f}",
            workflow_name,
            duration_ms,
        )

    def workflow_failed(
        self,
        workflow_name: str,
    ) -> None:

        self.increment(
            "clinical_workflows_failed_total",
            description="Clinical workflows failed",
        )

        logger.error(
            "Workflow failure metric | workflow={}",
            workflow_name,
        )

    # ==================================================================
    # AGENTS
    # ==================================================================

    def agent_started(
        self,
        agent_name: str,
    ) -> None:

        self.increment(
            "agent_executions_started_total",
            description="Agent executions started",
        )

        logger.debug(
            "Agent started metric | agent={}",
            agent_name,
        )

    def agent_completed(
        self,
        agent_name: str,
        duration_ms: float,
    ) -> None:

        self.increment(
            "agent_executions_completed_total",
            description="Agent executions completed",
        )

        self.observe(
            "agent_execution_duration_ms",
            duration_ms,
            description="Agent execution duration",
        )

        logger.debug(
            "Agent completed metric | "
            "agent={} | duration_ms={:.2f}",
            agent_name,
            duration_ms,
        )

    def agent_failed(
        self,
        agent_name: str,
    ) -> None:

        self.increment(
            "agent_executions_failed_total",
            description="Agent executions failed",
        )

        logger.warning(
            "Agent failure metric | agent={}",
            agent_name,
        )

    # ==================================================================
    # RAG
    # ==================================================================

    def rag_query(
        self,
        *,
        result_count: int,
        duration_ms: float,
    ) -> None:

        self.increment(
            "rag_queries_total",
            description="Total RAG queries",
        )

        self.observe(
            "rag_query_duration_ms",
            duration_ms,
            description="RAG query duration",
        )

        self.observe(
            "rag_results_per_query",
            float(result_count),
            description="Retrieved documents per query",
        )

        if result_count == 0:

            self.increment(
                "rag_empty_results_total",
                description="RAG queries with zero results",
            )

    def rag_failure(self) -> None:

        self.increment(
            "rag_query_failures_total",
            description="RAG query failures",
        )

    # ==================================================================
    # LLM
    # ==================================================================

    def llm_request(
        self,
        *,
        model: str,
        duration_ms: float,
        success: bool,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
    ) -> None:

        self.increment(
            "llm_requests_total",
            description="LLM requests",
        )

        self.observe(
            "llm_request_duration_ms",
            duration_ms,
            description="LLM request duration",
        )

        if success:

            self.increment(
                "llm_requests_success_total",
                description="Successful LLM requests",
            )

        else:

            self.increment(
                "llm_requests_failed_total",
                description="Failed LLM requests",
            )

        if input_tokens is not None:

            self.increment(
                "llm_input_tokens_total",
                description="LLM input tokens",
                value=input_tokens,
            )

        if output_tokens is not None:

            self.increment(
                "llm_output_tokens_total",
                description="LLM output tokens",
                value=output_tokens,
            )

        logger.debug(
            "LLM metrics | model={} | success={} | "
            "duration_ms={:.2f} | input_tokens={} | output_tokens={}",
            model,
            success,
            duration_ms,
            input_tokens,
            output_tokens,
        )

    # ==================================================================
    # GUARDRAILS
    # ==================================================================

    def guardrail_triggered(
        self,
        *,
        guardrail_name: str,
        action: str,
    ) -> None:

        self.increment(
            "guardrail_events_total",
            description="Guardrail events",
        )

        if action.lower() == "block":

            self.increment(
                "guardrail_blocks_total",
                description="Guardrail blocks",
            )

        elif action.lower() == "review":

            self.increment(
                "guardrail_review_required_total",
                description="Guardrail human-review events",
            )

        logger.warning(
            "Guardrail metric | "
            "guardrail={} | action={}",
            guardrail_name,
            action,
        )

    # ==================================================================
    # SAFETY
    # ==================================================================

    def safety_event(
        self,
        *,
        severity: str,
    ) -> None:

        self.increment(
            "clinical_safety_events_total",
            description="Clinical safety events",
        )

        normalized = severity.lower()

        if normalized in {
            "critical",
            "high",
        }:

            self.increment(
                "clinical_high_severity_safety_events_total",
                description="High/critical safety events",
            )

        logger.warning(
            "Clinical safety metric | severity={}",
            severity,
        )

    # ==================================================================
    # HUMAN REVIEW
    # ==================================================================

    def human_review_required(self) -> None:

        self.increment(
            "human_reviews_required_total",
            description="Clinical workflows requiring human review",
        )

    def human_review_completed(
        self,
        *,
        outcome: str,
    ) -> None:

        self.increment(
            "human_reviews_completed_total",
            description="Completed human reviews",
        )

        normalized = outcome.lower()

        self.increment(
            f"human_review_{self._safe_name(normalized)}_total",
            description="Human review outcomes",
        )

    # ==================================================================
    # SYSTEM
    # ==================================================================

    def active_workflows(
        self,
        value: int,
    ) -> None:

        metrics_registry.set_gauge(
            "active_clinical_workflows",
            value,
            description="Currently active clinical workflows",
        )

    def active_agents(
        self,
        value: int,
    ) -> None:

        metrics_registry.set_gauge(
            "active_agents",
            value,
            description="Currently executing agents",
        )

    # ==================================================================
    # GENERIC API
    # ==================================================================

    def increment(
        self,
        name: str,
        *,
        description: str = "",
        value: float = 1.0,
    ) -> float:

        return metrics_registry.increment_counter(
            name,
            description=description,
            value=value,
        )

    def observe(
        self,
        name: str,
        value: float,
        *,
        description: str = "",
    ) -> HistogramMetric:

        return metrics_registry.observe_histogram(
            name,
            value,
            description=description,
        )

    def gauge(
        self,
        name: str,
        value: float,
        *,
        description: str = "",
    ) -> float:

        return metrics_registry.set_gauge(
            name,
            value,
            description=description,
        )

    # ==================================================================
    # NAME SAFETY
    # ==================================================================

    @staticmethod
    def _safe_name(
        value: str,
    ) -> str:

        result = "".join(
            character
            if character.isalnum()
            else "_"
            for character in value.lower()
        )

        return result[:64]


# ============================================================================
# GLOBAL APPLICATION METRICS
# ============================================================================


application_metrics = ApplicationMetrics()


# ============================================================================
# TIMER
# ============================================================================


class MetricsTimer:
    """
    Timer that automatically records a histogram observation.

    Example
    -------

        with metrics_timer(
            "agent_execution_duration_ms",
            "Agent execution duration",
        ):
            result = agent.run()
    """

    def __init__(
        self,
        metric_name: str,
        description: str = "",
    ) -> None:

        self.metric_name = metric_name
        self.description = description
        self.started_at = time.perf_counter()

    def elapsed_ms(self) -> float:

        return (
            time.perf_counter()
            - self.started_at
        ) * 1000.0


@contextmanager
def metrics_timer(
    metric_name: str,
    description: str = "",
) -> Iterator[MetricsTimer]:
    """
    Synchronous metric timer.
    """

    timer = MetricsTimer(
        metric_name,
        description,
    )

    try:
        yield timer

    finally:

        duration_ms = timer.elapsed_ms()

        application_metrics.observe(
            metric_name,
            duration_ms,
            description=description,
        )


@asynccontextmanager
async def async_metrics_timer(
    metric_name: str,
    description: str = "",
) -> Iterator[MetricsTimer]:
    """
    Async metric timer.

    Example
    -------

        async with async_metrics_timer(
            "clinical_workflow_duration_ms"
        ):
            result = await graph.ainvoke(state)
    """

    timer = MetricsTimer(
        metric_name,
        description,
    )

    try:
        yield timer

    finally:

        duration_ms = timer.elapsed_ms()

        application_metrics.observe(
            metric_name,
            duration_ms,
            description=description,
        )


# ============================================================================
# METRIC SNAPSHOT
# ============================================================================


def get_metrics_snapshot() -> dict[str, Any]:
    """
    Return the current in-process metric snapshot.
    """

    return metrics_registry.snapshot()


# ============================================================================
# METRIC HEALTH
# ============================================================================


def metrics_health() -> dict[str, Any]:
    """
    Return a safe metrics health response.
    """

    snapshot = metrics_registry.snapshot()

    return {
        "status": "healthy",
        "counter_count": len(
            snapshot["counters"]
        ),
        "gauge_count": len(
            snapshot["gauges"]
        ),
        "histogram_count": len(
            snapshot["histograms"]
        ),
    }


# ============================================================================
# ASYNC CONCURRENCY HELPER
# ============================================================================


class AsyncGauge:
    """
    Convenience helper for tracking active asynchronous operations.
    """

    def __init__(
        self,
        metric_name: str,
        description: str = "",
    ) -> None:

        self.metric_name = metric_name
        self.description = description
        self._lock = asyncio.Lock()
        self._value = 0

    async def acquire(self) -> int:

        async with self._lock:

            self._value += 1

            metrics_registry.set_gauge(
                self.metric_name,
                self._value,
                description=self.description,
            )

            return self._value

    async def release(self) -> int:

        async with self._lock:

            self._value = max(
                0,
                self._value - 1,
            )

            metrics_registry.set_gauge(
                self.metric_name,
                self._value,
                description=self.description,
            )

            return self._value


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "MetricType",
    "MetricCategory",
    "CounterMetric",
    "GaugeMetric",
    "HistogramMetric",
    "MetricsRegistry",
    "metrics_registry",
    "ApplicationMetrics",
    "application_metrics",
    "MetricsTimer",
    "metrics_timer",
    "async_metrics_timer",
    "AsyncGauge",
    "get_metrics_snapshot",
    "metrics_health",
]
```

### How `metrics.py` fits your observability architecture

For your Clinical Decision Support Agentic AI system:

```text
                         Observability
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
     logging.py           metrics.py           tracing.py
        │                     │                     │
        ▼                     ▼                     ▼
    Events/messages       Numbers/time        Request spans
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                    Central Observability
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
           Grafana          SIEM          Trace Backend
```

### Key metrics for your Agentic Clinical workflow

I would track these categories:

| Category     | Example metrics                                |
| ------------ | ---------------------------------------------- |
| HTTP         | request count, 4xx, 5xx, latency               |
| Workflow     | started, completed, failed, duration           |
| Agents       | execution count, failures, duration            |
| RAG          | queries, empty results, latency, results/query |
| LLM          | requests, failures, latency, token usage       |
| Guardrails   | triggered, blocked, review-required            |
| Safety       | high/critical safety events                    |
| Human Review | required, completed, outcome                   |
| System       | active workflows, active agents                |

The resulting operational picture becomes:

```text
Clinical Decision Request
        │
        ▼
HTTP Metrics
        │
        ├── latency
        ├── status
        └── error rate
        │
        ▼
Clinical Workflow
        │
        ├── workflow duration
        ├── success/failure
        └── active workflows
        │
        ▼
LangGraph
        │
        ├── Intake
        ├── Context
        ├── Evidence
        ├── Differential
        ├── Missing Information
        ├── Diagnostic Tests
        ├── Safety
        ├── Critic
        └── Summary
              │
              ▼
        Agent Metrics
              │
        ┌─────┼──────────┐
        ▼     ▼          ▼
       RAG   LLM      Guardrails
        │     │          │
        ▼     ▼          ▼
     latency tokens   blocks
     results failures reviews
                    │
                    ▼
              Human Review
```

### Very important: metric cardinality

For your healthcare application, **do not create metrics such as**:

```python
# ❌ Never do this

metrics.increment(
    f"patient_{patient_id}_requests_total"
)

metrics.increment(
    f"workflow_{workflow_id}_duration"
)

metrics.increment(
    f"user_{user_id}_requests"
)
```

This creates extremely high-cardinality metrics.

Instead:

```python
# ✅ Good

application_metrics.increment(
    "clinical_workflows_started_total"
)

application_metrics.observe(
    "clinical_workflow_duration_ms",
    duration_ms,
)
```

Use `request_id`, `workflow_id`, `patient_id`, and `user_id` in **logs/traces/audit**, not metric names or high-cardinality labels.

### Example in your `ClinicalGraph`

Your graph can measure every agent:

```python
async def _run_agent(
    self,
    agent,
    state,
):
    agent_name = agent.__class__.__name__

    application_metrics.agent_started(
        agent_name
    )

    started = time.perf_counter()

    try:
        result = await agent.node(state)

        duration_ms = (
            time.perf_counter() - started
        ) * 1000

        application_metrics.agent_completed(
            agent_name,
            duration_ms,
        )

        return result

    except Exception:

        application_metrics.agent_failed(
            agent_name
        )

        raise