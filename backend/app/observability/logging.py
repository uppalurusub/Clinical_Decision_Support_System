"""
app/observability/logging.py

Centralized application logging using Loguru.

Responsibilities
----------------
1. Configure Loguru globally.
2. Provide console and optional file logging.
3. Provide structured application logging helpers.
4. Support request/workflow/agent correlation.
5. Provide safe logging for healthcare applications.
6. Prevent accidental PHI/PII/secret logging.
7. Support development and production environments.
8. Provide exception logging helpers.
9. Provide agent/workflow lifecycle logging.
10. Provide performance/duration logging.

Important
---------
This module is APPLICATION LOGGING.

It is NOT the security audit layer.

Use:
    app/security/audit.py

for:
    - authentication events
    - authorization events
    - PHI access
    - security violations
    - human-review decisions
    - compliance events

Never log:
    - passwords
    - JWT/access tokens
    - API keys
    - secrets
    - raw prompts
    - raw LLM responses
    - patient names
    - DOB
    - MRN
    - addresses
    - phone numbers
    - clinical notes
    - laboratory values
    - medication details
    - diagnostic narratives
    - imaging content

```

### How this fits your observability architecture

I would keep **application logging, audit logging, metrics, tracing, and agent telemetry separate**:

```text
app/observability/
│
├── logging.py
│     └── Application Logs
│
├── metrics.py
│     └── Prometheus / OpenTelemetry Metrics
│
├── tracing.py
│     └── Distributed Tracing
│
├── events.py
│     └── Application Events
│
├── errors.py
│     └── Error Classification
│
├── performance.py
│     └── Latency / Throughput
│
├── agent_telemetry.py
│     └── Agent Execution
│
└── llm_telemetry.py
      └── LLM Calls / Tokens / Latency
```

The important distinction is:

```text
logging.py
    │
    ├── DEBUG
    ├── INFO
    ├── WARNING
    ├── ERROR
    └── CRITICAL
         │
         └── Application Operations


security/audit.py
    │
    ├── Authentication
    ├── Authorization
    ├── PHI Access
    ├── Human Review
    ├── Security Events
    └── Compliance Events
         │
         └── Audit Trail
```

### Request → Agent correlation

The `contextvars` support is particularly useful for your LangGraph workflow:

```text
HTTP Request
    │
    │ request_id = R123
    ▼
Router
    │
    │ workflow_id = W456
    ▼
Service
    │
    ▼
Implementation
    │
    ▼
LangGraph
    │
    ├── Intake Agent
    │      agent=intake
    │
    ├── Context Agent
    │      agent=context
    │
    ├── Evidence Agent
    │      agent=evidence
    │
    ├── Differential Agent
    │      agent=differential
    │
    ├── Missing Information Agent
    │      agent=missing_information
    │
    ├── Diagnostic Test Agent
    │      agent=diagnostic_tests
    │
    ├── Safety Agent
    │      agent=safety
    │
    ├── Critic Agent
    │      agent=critic
    │
    └── Summary Agent
           agent=summary
```

This gives you a searchable correlation chain:

```text
request_id
    ↓
workflow_id
    ↓
agent_name
    ↓
operation
    ↓
duration
    ↓
status
```

For example, an operational log can look conceptually like:

```text
2026-09-24 06:30:10.125 | INFO | clinical-decision-support
| req=8c... | wf=2f... | agent=evidence
| Agent started
```

without putting the patient's clinical information into the log.

### Recommended `.env`

```text
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_TO_CONSOLE=true
LOG_TO_FILE=true
LOG_FILE=logs/application.log
LOG_ROTATION=100 MB
LOG_RETENTION=30 days
LOG_COMPRESSION=zip
LOG_ENQUEUE=true
LOG_SERIALIZE=false

SERVICE_NAME=clinical-decision-support
ENVIRONMENT=development
```

For Kubernetes/container deployment, I would generally prefer:

```text
Application
    ↓
Loguru
    ↓
stdout
    ↓
Fluent Bit / OpenTelemetry Collector
    ↓
Centralized Logging
    ↓
ELK / OpenSearch / Splunk / Azure Monitor / CloudWatch
```

rather than having every pod maintain its own long-lived log files.

**One important healthcare design rule:** `logging.py` should never be the place where you intentionally log clinical payloads for debugging. If detailed clinical/LLM observability is needed, use a dedicated, access-controlled telemetry design with explicit PHI handling, redaction, retention, and access policies.

"""

from __future__ import annotations

import contextvars
import json
import os
import re
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import Token
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from app.config.settings import settings
logger = settings.get_logger()


# ============================================================================
# CONTEXT VARIABLES
# ============================================================================

request_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "request_id",
    default=None,
)

workflow_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "workflow_id",
    default=None,
)

user_id_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "user_id",
    default=None,
)

agent_name_context: contextvars.ContextVar[
    Optional[str]
] = contextvars.ContextVar(
    "agent_name",
    default=None,
)


# ============================================================================
# CONFIGURATION
# ============================================================================


class LogLevel(str, Enum):
    """
    Supported logging levels.
    """

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig:
    """
    Application logging configuration.

    Configuration is read from environment variables.

    Environment variables
    ---------------------
    LOG_LEVEL
        INFO

    LOG_FORMAT
        text | json

    LOG_TO_CONSOLE
        true

    LOG_TO_FILE
        false

    LOG_FILE
        logs/application.log

    LOG_ROTATION
        100 MB

    LOG_RETENTION
        30 days

    LOG_COMPRESSION
        zip

    LOG_ENQUEUE
        true

    LOG_SERIALIZE
        false

    LOG_BACKTRACE
        false

    LOG_DIAGNOSE
        false
    """

    def __init__(self) -> None:

        self.level = os.getenv(
            "LOG_LEVEL",
            "INFO",
        ).upper()

        self.format = os.getenv(
            "LOG_FORMAT",
            "text",
        ).lower()

        self.to_console = self._get_bool(
            "LOG_TO_CONSOLE",
            True,
        )

        self.to_file = self._get_bool(
            "LOG_TO_FILE",
            False,
        )

        self.file_path = os.getenv(
            "LOG_FILE",
            "logs/application.log",
        )

        self.rotation = os.getenv(
            "LOG_ROTATION",
            "100 MB",
        )

        self.retention = os.getenv(
            "LOG_RETENTION",
            "30 days",
        )

        self.compression = os.getenv(
            "LOG_COMPRESSION",
            "zip",
        )

        self.enqueue = self._get_bool(
            "LOG_ENQUEUE",
            True,
        )

        self.serialize = self._get_bool(
            "LOG_SERIALIZE",
            False,
        )

        self.backtrace = self._get_bool(
            "LOG_BACKTRACE",
            False,
        )

        self.diagnose = self._get_bool(
            "LOG_DIAGNOSE",
            False,
        )

        self.service_name = os.getenv(
            "SERVICE_NAME",
            "clinical-decision-support",
        )

        self.environment = os.getenv(
            "ENVIRONMENT",
            "development",
        )

    @staticmethod
    def _get_bool(
        name: str,
        default: bool,
    ) -> bool:

        value = os.getenv(name)

        if value is None:
            return default

        return value.lower() in {
            "true",
            "1",
            "yes",
            "on",
        }


# ============================================================================
# SENSITIVE DATA PROTECTION
# ============================================================================


class SensitiveDataFilter:
    """
    Protect application logs from accidental sensitive data.

    This is a defense-in-depth mechanism.

    It should NOT be considered a replacement for:
        - DLP
        - Privacy Guardrails
        - PHI classification
        - access control
        - secure coding practices
    """

    SENSITIVE_KEYS = frozenset(
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

    SECRET_PATTERNS = (
        re.compile(
            r"(?i)bearer\s+[A-Za-z0-9._\-]+"
        ),
        re.compile(
            r"(?i)(api[_-]?key|secret|password)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        re.compile(
            r"(?i)sk-[A-Za-z0-9_-]{10,}"
        ),
    )

    @classmethod
    def sanitize_mapping(
        cls,
        data: Mapping[str, Any],
    ) -> dict[str, Any]:

        sanitized: dict[str, Any] = {}

        for raw_key, value in data.items():

            key = str(raw_key)
            normalized_key = key.lower()

            if normalized_key in cls.SENSITIVE_KEYS:
                sanitized[key] = "[REDACTED]"
                continue

            sanitized[key] = cls.sanitize_value(
                value
            )

        return sanitized

    @classmethod
    def sanitize_value(
        cls,
        value: Any,
    ) -> Any:

        if value is None:
            return None

        if isinstance(value, Mapping):
            return cls.sanitize_mapping(value)

        if isinstance(value, (list, tuple)):
            return [
                cls.sanitize_value(item)
                for item in value
            ]

        if isinstance(value, Enum):
            return value.value

        if isinstance(value, datetime):
            return value.isoformat()

        if isinstance(value, str):

            sanitized = value

            for pattern in cls.SECRET_PATTERNS:
                sanitized = pattern.sub(
                    "[REDACTED]",
                    sanitized,
                )

            # Do not attempt broad PHI detection here.
            # Application code should never send PHI to logger.
            return sanitized

        if isinstance(
            value,
            (int, float, bool),
        ):
            return value

        return f"<{type(value).__name__}>"

    @classmethod
    def sanitize_message(
        cls,
        message: str,
    ) -> str:

        if not message:
            return message

        sanitized = message

        for pattern in cls.SECRET_PATTERNS:
            sanitized = pattern.sub(
                "[REDACTED]",
                sanitized,
            )

        return sanitized


# ============================================================================
# LOG FORMAT
# ============================================================================


TEXT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[service]}</cyan> | "
    "<cyan>req={extra[request_id]}</cyan> | "
    "<cyan>wf={extra[workflow_id]}</cyan> | "
    "<cyan>agent={extra[agent_name]}</cyan> | "
    "<level>{message}</level>"
)


def _json_serializer(record: Mapping[str, Any]) -> str:
    """
    Serialize a Loguru record as JSON.

    Used for centralized logging platforms such as:
        - ELK
        - OpenSearch
        - Splunk
        - Datadog
        - Azure Monitor
        - CloudWatch
    """

    extra = record.get("extra", {})

    payload = {
        "timestamp": record["time"].astimezone(
            timezone.utc
        ).isoformat(),
        "level": record["level"].name,
        "message": SensitiveDataFilter.sanitize_message(
            record["message"]
        ),
        "service": extra.get(
            "service",
            "unknown",
        ),
        "environment": extra.get(
            "environment",
            "unknown",
        ),
        "request_id": extra.get(
            "request_id"
        ),
        "workflow_id": extra.get(
            "workflow_id"
        ),
        "user_id": extra.get(
            "user_id"
        ),
        "agent_name": extra.get(
            "agent_name"
        ),
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def _text_serializer(
    record: Mapping[str, Any],
) -> str:
    """
    Plain text serializer.
    """

    extra = record.get(
        "extra",
        {},
    )

    request_id = extra.get(
        "request_id",
        "-",
    )

    workflow_id = extra.get(
        "workflow_id",
        "-",
    )

    agent_name = extra.get(
        "agent_name",
        "-",
    )

    return (
        f"{record['time'].strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} "
        f"| {record['level'].name:<8} "
        f"| {extra.get('service', 'unknown')} "
        f"| req={request_id} "
        f"| wf={workflow_id} "
        f"| agent={agent_name} "
        f"| {SensitiveDataFilter.sanitize_message(record['message'])}"
    )


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================


class LoggingManager:
    """
    Central Loguru configuration manager.

    Initialize once during application startup.
    """

    def __init__(
        self,
        config: Optional[LoggingConfig] = None,
    ) -> None:

        self.config = (
            config
            if config is not None
            else LoggingConfig()
        )

        self._configured = False

    # ==================================================================
    # CONFIGURE
    # ==================================================================

    def configure(self) -> None:
        """
        Configure global Loguru sinks.
        """

        if self._configured:
            return

        logger.remove()

        base_logger = logger.bind(
            service=self.config.service_name,
            environment=self.config.environment,
            request_id="-",
            workflow_id="-",
            user_id="-",
            agent_name="-",
        )

        # --------------------------------------------------------------
        # Console
        # --------------------------------------------------------------

        if self.config.to_console:

            if self.config.format == "json":

                base_logger.add(
                    sys.stdout,
                    level=self.config.level,
                    serialize=False,
                    enqueue=self.config.enqueue,
                    backtrace=self.config.backtrace,
                    diagnose=self.config.diagnose,
                    filter=self._filter_record,
                    format="{message}",
                )

            else:

                base_logger.add(
                    sys.stdout,
                    level=self.config.level,
                    colorize=True,
                    enqueue=self.config.enqueue,
                    backtrace=self.config.backtrace,
                    diagnose=self.config.diagnose,
                    filter=self._filter_record,
                    format=TEXT_LOG_FORMAT,
                )

        # --------------------------------------------------------------
        # File
        # --------------------------------------------------------------

        if self.config.to_file:

            self._ensure_log_directory()

            base_logger.add(
                self.config.file_path,
                level=self.config.level,
                rotation=self.config.rotation,
                retention=self.config.retention,
                compression=self.config.compression,
                enqueue=self.config.enqueue,
                backtrace=self.config.backtrace,
                diagnose=self.config.diagnose,
                filter=self._filter_record,
                serialize=(
                    self.config.format == "json"
                ),
                encoding="utf-8",
            )

        self._configured = True

        logger.info(
            "Application logging configured | "
            "level={} | format={} | console={} | file={} | environment={}",
            self.config.level,
            self.config.format,
            self.config.to_console,
            self.config.to_file,
            self.config.environment,
        )

    # ==================================================================
    # DIRECTORY
    # ==================================================================

    def _ensure_log_directory(self) -> None:

        path = Path(
            self.config.file_path
        )

        if path.parent != Path("."):
            path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ==================================================================
    # FILTER
    # ==================================================================

    @staticmethod
    def _filter_record(
        record: dict[str, Any],
    ) -> bool:
        """
        Add correlation context and sanitize extra values.

        Loguru's filter executes before the sink writes the record.
        """

        extra = record.setdefault(
            "extra",
            {},
        )

        extra.setdefault(
            "service",
            "clinical-decision-support",
        )

        extra.setdefault(
            "environment",
            "development",
        )

        extra.setdefault(
            "request_id",
            request_id_context.get() or "-",
        )

        extra.setdefault(
            "workflow_id",
            workflow_id_context.get() or "-",
        )

        extra.setdefault(
            "user_id",
            user_id_context.get() or "-",
        )

        extra.setdefault(
            "agent_name",
            agent_name_context.get() or "-",
        )

        # Never allow arbitrary structured data to pass through.
        safe_extra = {}

        for key, value in extra.items():

            if key in {
                "service",
                "environment",
                "request_id",
                "workflow_id",
                "user_id",
                "agent_name",
            }:

                safe_extra[key] = (
                    SensitiveDataFilter.sanitize_value(
                        value
                    )
                )

        record["extra"] = safe_extra

        return True

    # ==================================================================
    # STATUS
    # ==================================================================

    @property
    def configured(self) -> bool:
        return self._configured


# ============================================================================
# GLOBAL MANAGER
# ============================================================================


logging_manager = LoggingManager()


def configure_logging() -> None:
    """
    Configure application logging.

    Call once during FastAPI startup.
    """

    logging_manager.configure()


# ============================================================================
# CORRELATION CONTEXT
# ============================================================================


def set_request_context(
    request_id: Optional[str] = None,
) -> str:
    """
    Set request correlation ID.

    Returns:
        Request ID.
    """

    value = (
        request_id
        or str(uuid.uuid4())
    )

    request_id_context.set(value)

    return value


def set_workflow_context(
    workflow_id: Optional[str],
) -> Optional[str]:
    """
    Set LangGraph workflow correlation ID.
    """

    workflow_id_context.set(
        workflow_id
    )

    return workflow_id


def set_user_context(
    user_id: Optional[str],
) -> Optional[str]:
    """
    Set authenticated user context.

    The user ID should be an opaque application identifier.
    """

    user_id_context.set(
        user_id
    )

    return user_id


def set_agent_context(
    agent_name: Optional[str],
) -> Optional[str]:
    """
    Set current Agentic AI agent context.
    """

    agent_name_context.set(
        agent_name
    )

    return agent_name


# ============================================================================
# CONTEXT MANAGERS
# ============================================================================


@contextmanager
def request_context(
    request_id: Optional[str] = None,
) -> Iterator[str]:
    """
    Temporarily set request context.

    Example
    -------

        with request_context(request_id):
            logger.info("Processing request")
    """

    value = (
        request_id
        or str(uuid.uuid4())
    )

    token: Token = request_id_context.set(
        value
    )

    try:
        yield value

    finally:
        request_id_context.reset(
            token
        )


@contextmanager
def workflow_context(
    workflow_id: str,
) -> Iterator[str]:
    """
    Temporarily set workflow context.
    """

    token: Token = workflow_id_context.set(
        workflow_id
    )

    try:
        yield workflow_id

    finally:
        workflow_id_context.reset(
            token
        )


@contextmanager
def agent_context(
    agent_name: str,
) -> Iterator[str]:
    """
    Temporarily set agent context.
    """

    token: Token = agent_name_context.set(
        agent_name
    )

    try:
        yield agent_name

    finally:
        agent_name_context.reset(
            token
        )


# ============================================================================
# STRUCTURED APPLICATION LOGGING
# ============================================================================


def log_event(
    event: str,
    *,
    level: str = "INFO",
    **metadata: Any,
) -> None:
    """
    Write a structured application event.

    Only operational metadata should be supplied.

    Example:

        log_event(
            "clinical_workflow_started",
            workflow_id=workflow_id,
            agent_count=9,
        )
    """

    safe_metadata = (
        SensitiveDataFilter.sanitize_mapping(
            metadata
        )
    )

    message = (
        f"{event} | "
        f"{json.dumps(safe_metadata, default=str)}"
    )

    logger.log(
        level.upper(),
        message,
    )


# ============================================================================
# COMMON LOGGING HELPERS
# ============================================================================


def log_request_started(
    *,
    method: str,
    path: str,
) -> None:

    logger.info(
        "HTTP request started | method={} | path={}",
        method,
        path,
    )


def log_request_completed(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:

    logger.info(
        "HTTP request completed | "
        "method={} | path={} | status={} | duration_ms={:.2f}",
        method,
        path,
        status_code,
        duration_ms,
    )


def log_request_failed(
    *,
    method: str,
    path: str,
    status_code: int,
    duration_ms: float,
) -> None:

    logger.error(
        "HTTP request failed | "
        "method={} | path={} | status={} | duration_ms={:.2f}",
        method,
        path,
        status_code,
        duration_ms,
    )


# ============================================================================
# WORKFLOW LOGGING
# ============================================================================


def log_workflow_started(
    workflow_id: str,
    workflow_name: str,
) -> None:

    with workflow_context(workflow_id):

        logger.info(
            "Workflow started | workflow={} ",
            workflow_name,
        )


def log_workflow_completed(
    workflow_id: str,
    workflow_name: str,
    duration_ms: Optional[float] = None,
) -> None:

    with workflow_context(workflow_id):

        logger.info(
            "Workflow completed | "
            "workflow={} | duration_ms={}",
            workflow_name,
            (
                f"{duration_ms:.2f}"
                if duration_ms is not None
                else "-"
            ),
        )


def log_workflow_failed(
    workflow_id: str,
    workflow_name: str,
    duration_ms: Optional[float] = None,
) -> None:

    with workflow_context(workflow_id):

        logger.error(
            "Workflow failed | "
            "workflow={} | duration_ms={}",
            workflow_name,
            (
                f"{duration_ms:.2f}"
                if duration_ms is not None
                else "-"
            ),
        )


# ============================================================================
# AGENT LOGGING
# ============================================================================


def log_agent_started(
    agent_name: str,
) -> None:

    logger.info(
        "Agent started | agent={}",
        agent_name,
    )


def log_agent_completed(
    agent_name: str,
    duration_ms: Optional[float] = None,
) -> None:

    logger.info(
        "Agent completed | "
        "agent={} | duration_ms={}",
        agent_name,
        (
            f"{duration_ms:.2f}"
            if duration_ms is not None
            else "-"
        ),
    )


def log_agent_failed(
    agent_name: str,
) -> None:

    logger.error(
        "Agent failed | agent={}",
        agent_name,
    )


# ============================================================================
# PERFORMANCE MEASUREMENT
# ============================================================================


class Timer:
    """
    Simple performance timer.

    Example
    -------

        timer = Timer()

        ...

        logger.info(
            "Operation completed | duration_ms={:.2f}",
            timer.elapsed_ms(),
        )
    """

    def __init__(self) -> None:
        self.started_at = time.perf_counter()

    def elapsed_ms(self) -> float:
        return (
            time.perf_counter()
            - self.started_at
        ) * 1000.0


@contextmanager
def measure(
    operation: str,
) -> Iterator[Timer]:
    """
    Measure execution duration.

    Example
    -------

        with measure("clinical_graph"):
            await graph.ainvoke(state)

    """

    timer = Timer()

    logger.debug(
        "Operation started | operation={}",
        operation,
    )

    try:
        yield timer

    except Exception:

        logger.exception(
            "Operation failed | "
            "operation={} | duration_ms={:.2f}",
            operation,
            timer.elapsed_ms(),
        )

        raise

    else:

        logger.debug(
            "Operation completed | "
            "operation={} | duration_ms={:.2f}",
            operation,
            timer.elapsed_ms(),
        )


# ============================================================================
# EXCEPTION LOGGING
# ============================================================================


def log_exception(
    message: str,
    *,
    exc: Optional[BaseException] = None,
    **metadata: Any,
) -> None:
    """
    Safely log an application exception.

    Avoid passing the exception itself if its message could contain PHI.
    """

    safe_metadata = (
        SensitiveDataFilter.sanitize_mapping(
            metadata
        )
    )

    if safe_metadata:

        logger.exception(
            "{} | metadata={}",
            SensitiveDataFilter.sanitize_message(
                message
            ),
            json.dumps(
                safe_metadata,
                default=str,
            ),
        )

    else:

        logger.exception(
            "{}",
            SensitiveDataFilter.sanitize_message(
                message
            ),
        )


# ============================================================================
# HEALTH CHECK
# ============================================================================


def logging_health() -> dict[str, Any]:
    """
    Return non-sensitive logging configuration status.
    """

    config = logging_manager.config

    return {
        "status": "healthy"
        if logging_manager.configured
        else "not_configured",
        "level": config.level,
        "format": config.format,
        "console_enabled": config.to_console,
        "file_enabled": config.to_file,
        "service": config.service_name,
        "environment": config.environment,
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "LogLevel",
    "LoggingConfig",
    "LoggingManager",
    "logging_manager",
    "configure_logging",
    "request_id_context",
    "workflow_id_context",
    "user_id_context",
    "agent_name_context",
    "set_request_context",
    "set_workflow_context",
    "set_user_context",
    "set_agent_context",
    "request_context",
    "workflow_context",
    "agent_context",
    "log_event",
    "log_request_started",
    "log_request_completed",
    "log_request_failed",
    "log_workflow_started",
    "log_workflow_completed",
    "log_workflow_failed",
    "log_agent_started",
    "log_agent_completed",
    "log_agent_failed",
    "Timer",
    "measure",
    "log_exception",
    "logging_health",
]
