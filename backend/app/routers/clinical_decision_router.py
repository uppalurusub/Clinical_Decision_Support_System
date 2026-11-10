
"""
clinical_decision_router.py

FastAPI router for Clinical Decision Support & Diagnosis.

Architecture:
    Router
       ↓
    Service
       ↓
    Implementation
       ↓
    LangGraph Workflow



### Recommended dependency

Because the router should **not instantiate the service directly**, use a dependency such as:

```python
from app.implementations.clinical_decision_impl import (
    ClinicalDecisionImplementation,
)
from app.services.clinical_decision_service import (
    ClinicalDecisionService,
)


def get_clinical_decision_service() -> ClinicalDecisionService:
    implementation = ClinicalDecisionImplementation()

    return ClinicalDecisionService(
        implementation=implementation
    )
```

### Resulting architecture

```text
POST /api/v1/clinical-decision/analyze
                  │
                  ▼
      clinical_decision_router.py
                  │
             Loguru logging
                  │
                  ▼
      ClinicalDecisionService
                  │
                  ▼
   ClinicalDecisionImplementation
                  │
                  ▼
          LangGraph Workflow
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
    Context    Evidence    Safety
     Agent       Agent      Agent
       │          │          │
       └──────────┼──────────┘
                  ▼
          Differential Agent
                  │
                  ▼
          Clinical Critic
                  │
                  ▼
           Human Review


### Important Loguru/Healthcare consideration

Notice that the router logs only:

```text
patient_id
operation
status
error
```

It should **not log raw symptoms, diagnoses, laboratory values, medications, clinical notes, or other PHI**. For production healthcare systems, keep detailed clinical data in appropriately controlled audit/telemetry systems rather than ordinary application logs.

"""

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from app.config.settings import settings
logger = settings.get_logger()

from app.schemas.request_model import ClinicalDecisionRequest
from app.schemas.response_model import ClinicalDecisionResponse
from app.services.clinical_decision_service import (
    ClinicalDecisionService,
)

#from app.dependencies import get_clinical_service
from app.dependencies import get_clinical_decision_service


# ---------------------------------------------------------------------------
# Router Configuration
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/api/v1/clinical-decision",
    tags=["Clinical Decision Support"],
)


# ---------------------------------------------------------------------------
# POST /analyze
# ---------------------------------------------------------------------------

@router.post(
    "/analyze",
    response_model=ClinicalDecisionResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze clinical information",
    description=(
        "Executes the agentic clinical decision-support workflow "
        "using LangGraph and LangChain under clinician supervision."
    ),
)
async def analyze_clinical_decision(
    request: ClinicalDecisionRequest,
    service: ClinicalDecisionService = Depends(
        get_clinical_decision_service
    ),
) -> ClinicalDecisionResponse:
    """
    Analyze patient clinical information.

    Flow:
        HTTP Request
            ↓
        FastAPI Router
            ↓
        ClinicalDecisionService
            ↓
        ClinicalDecisionImplementation
            ↓
        LangGraph
            ↓
        Clinical Agents
            ↓
        ClinicalDecisionResponse

    Args:
        request:
            Clinical decision-support request.

        service:
            Injected ClinicalDecisionService.

    Returns:
        ClinicalDecisionResponse:
            Structured clinical decision-support result.

    Raises:
        HTTPException:
            If the workflow fails.
    """

    patient_id = request.patient_id

    logger.info(
        "Clinical decision analysis request received | "
        "patient_id={}",
        patient_id,
    )

    try:
        # ---------------------------------------------------------------
        # Delegate business/application logic to Service layer
        # ---------------------------------------------------------------

        response = await service.analyze(request)

        logger.info(
            "Clinical decision analysis completed successfully | "
            "patient_id={}",
            patient_id,
        )

        return response

    except ValueError as exc:
        logger.warning(
            "Clinical decision validation failed | "
            "patient_id={} | error={}",
            patient_id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Clinical decision analysis failed | "
            "patient_id={} | error={}",
            patient_id,
            str(exc),
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to complete clinical decision-support analysis."
            ),
        ) from exc