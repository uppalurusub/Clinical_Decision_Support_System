"""
clinical_decision_service.py

Service layer for Clinical Decision Support & Diagnosis.

Architecture:
    FastAPI Router
          ↓
    ClinicalDecisionService
          ↓
    ClinicalDecisionImplementation
          ↓
    LangGraph Workflow

```

### Layer responsibility

With this service, your architecture is now cleanly separated:

```text
┌─────────────────────────────────────────────┐
│ FastAPI Router                              │
│                                             │
│ - HTTP                                      │
│ - Request/Response                          │
│ - HTTP Exceptions                           │
│ - Dependency Injection                      │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ ClinicalDecisionService                     │
│                                             │
│ - Application validation                    │
│ - Workflow lifecycle logging                │
│ - Delegation                                │
│ - Service-level exception propagation       │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ ClinicalDecisionImplementation              │
│                                             │
│ - Build LangGraph state                     │
│ - Invoke graph                              │
│ - Transform graph result                    │
└─────────────────────┬───────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────┐
│ LangGraph                                   │
│                                             │
│ Intake → Context → RAG → Differential       │
│                  ↓                          │
│       Missing Information → Tests            │
│                  ↓                          │
│              Safety → Critic                 │
│                  ↓                          │
│             Human Review                     │
└─────────────────────────────────────────────┘
```

### Why this separation is important

The **Router should not call LangGraph directly**:

```python
# ❌ Avoid

@router.post("/analyze")
async def analyze(request):
    result = await clinical_graph.ainvoke(...)
```

The Router should call the Service:

```python
# ✅ Recommended

@router.post("/analyze")
async def analyze(request):

    return await service.analyze(request)
```

The Service should delegate to the Implementation:

```python
# ✅

response = await self.implementation.analyze(request)
```

And the Implementation should invoke LangGraph:

```python
# ✅

result = await self.clinical_graph.ainvoke(
    initial_state
)
```

So the final responsibility chain is:

```text
Router
  │
  │ HTTP concerns
  ▼
Service
  │
  │ application concerns
  ▼
Implementation
  │
  │ AI workflow concerns
  ▼
LangGraph
  │
  ├── Agents
  ├── Tools
  ├── RAG
  ├── LLM
  └── Guardrails
```

One additional production consideration: **don't put detailed clinical payloads into Loguru logs**. Log workflow IDs, operation names, status, latency, and non-sensitive identifiers; keep PHI out of ordinary application logs.

"""

from app.config.settings import settings
logger = settings.get_logger()

from app.implementations.clinical_decision_impl import (
    ClinicalDecisionImplementation,
)
from app.schemas.request_model import ClinicalDecisionRequest
from app.schemas.response_model import ClinicalDecisionResponse


class ClinicalDecisionService:
    """
    Service layer responsible for coordinating the
    Clinical Decision Support workflow.

    Responsibilities:
        - Validate service-level requirements
        - Log workflow lifecycle events
        - Delegate execution to implementation layer
        - Handle service-level exceptions
        - Return structured response

    The service should NOT contain:
        - FastAPI route logic
        - LangGraph node definitions
        - LLM prompts
        - Direct database implementation
        - Agent implementation
    """

    def __init__(
        self,
        implementation: ClinicalDecisionImplementation,
    ) -> None:
        """
        Initialize ClinicalDecisionService.

        Args:
            implementation:
                Clinical decision implementation responsible
                for executing the LangGraph workflow.
        """

        self.implementation = implementation

        logger.debug(
            "ClinicalDecisionService initialized | "
            "implementation={}",
            implementation.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Analyze Clinical Decision
    # ------------------------------------------------------------------

    async def analyze(
        self,
        request: ClinicalDecisionRequest,
    ) -> ClinicalDecisionResponse:
        """
        Execute the clinical decision-support workflow.

        Flow:

            Router
               ↓
            Service
               ↓
            Implementation
               ↓
            LangGraph
               ↓
            Agents
               ↓
            Response

        Args:
            request:
                Clinical decision-support request.

        Returns:
            ClinicalDecisionResponse:
                Structured clinical decision-support result.

        Raises:
            ValueError:
                When request validation fails.

            Exception:
                When implementation/workflow execution fails.
        """

        patient_id = request.patient_id

        logger.info(
            "Clinical decision service started | "
            "patient_id={}",
            patient_id,
        )

        try:

            # ----------------------------------------------------------
            # Service-level validation
            # ----------------------------------------------------------

            self._validate_request(request)

            logger.debug(
                "Clinical decision request validated | "
                "patient_id={}",
                patient_id,
            )

            # ----------------------------------------------------------
            # Delegate to implementation layer
            # ----------------------------------------------------------

            response = await self.implementation.analyze(
                request
            )

            logger.info(
                "Clinical decision service completed | "
                "patient_id={}",
                patient_id,
            )

            return response

        except ValueError:
            logger.warning(
                "Clinical decision service validation error | "
                "patient_id={}",
                patient_id,
            )

            raise

        except Exception as exc:

            logger.exception(
                "Clinical decision service execution failed | "
                "patient_id={} | error={}",
                patient_id,
                str(exc),
            )

            raise

    # ------------------------------------------------------------------
    # Request Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_request(
        request: ClinicalDecisionRequest,
    ) -> None:
        """
        Perform service-level validation.

        Pydantic handles structural validation.
        This method handles application-specific validation.

        Args:
            request:
                Clinical decision-support request.

        Raises:
            ValueError:
                If required business conditions are not satisfied.
        """

        if not request.patient_id:
            logger.warning(
                "Clinical decision request missing patient_id"
            )

            raise ValueError(
                "patient_id is required."
            )

        if not request.symptoms:
            logger.warning(
                "Clinical decision request missing symptoms | "
                "patient_id={}",
                request.patient_id,
            )

            raise ValueError(
                "At least one clinical symptom or presenting "
                "complaint is required."
            )

        logger.debug(
            "Clinical decision request passed "
            "application validation | patient_id={}",
            request.patient_id,
        )
