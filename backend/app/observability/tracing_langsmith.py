"""Shared tracing helpers for LangChain, LangGraph, and FastAPI."""
from __future__ import annotations


import os
import uuid
from typing import Any

from langsmith import traceable

from app.config.settings import settings
logger = settings.get_logger()


def new_request_id() -> str:
    return str(uuid.uuid4())


def langsmith_metadata(request_id: str, **extra: Any) -> dict[str, Any]:
    """Metadata is safe operational context; do not include PHI or secrets."""
    return {
        "request_id": request_id,
        "service": os.getenv("SERVICE_NAME", "clinical-ai-api"),
        "environment": os.getenv("APP_ENV", "development"),
        **extra,
    }


def tracing_config(request_id: str, **extra: Any) -> dict[str, Any]:
    return {
        "run_name": extra.pop("run_name", "clinical_ai_request"),
        "tags": ["production", "healthcare-ai"],
        "metadata": langsmith_metadata(request_id, **extra),
        "configurable": {"thread_id": request_id},
        "recursion_limit": int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "25")),
    }


@traceable(name="clinical_ai_operation", run_type="chain")
def traced_operation(operation: str, request_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Example wrapper for a non-graph operation. Avoid sending sensitive payloads."""
    logger.info("operation=%s request_id=%s", operation, request_id)
    return {"operation": operation, "request_id": request_id, "received": bool(payload)}
