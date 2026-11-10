"""
clinical_decision_impl.py

Implementation layer for Clinical Decision Support & Diagnosis.

Architecture:

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
    Clinical Decision Response

```

### One important correction to the earlier dependency design

I recommend that `ClinicalDecisionImplementation` receive the **compiled LangGraph** through dependency injection rather than creating it internally.

```text
FastAPI
   │
   ▼
Router
   │
   ▼
Service
   │
   ▼
Implementation
   │
   ▼
Compiled ClinicalGraph
   │
   ▼
LangGraph StateGraph
```

For example:

```python
clinical_graph = build_clinical_graph()

implementation = ClinicalDecisionImplementation(
    clinical_graph=clinical_graph
)
```

This is better than:

```python
# ❌ Avoid creating a new graph for every request

implementation = ClinicalDecisionImplementation()

graph = build_clinical_graph()
```

because the graph is application infrastructure and should normally be constructed once and reused.

### Final three-layer architecture

You now have a clean separation:

```text
┌────────────────────────────────────────────┐
│ clinical_decision_router.py                │
│                                            │
│ FastAPI                                    │
│ HTTP request/response                      │
│ Dependency injection                       │
│ HTTP exceptions                            │
│ Loguru request logging                     │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│ clinical_decision_service.py               │
│                                            │
│ Application/business coordination          │
│ Request validation                          │
│ Service logging                             │
│ Delegation                                 │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│ clinical_decision_impl.py                  │
│                                            │
│ Build LangGraph state                      │
│ Invoke graph                               │
│ Validate graph result                      │
│ Transform graph → API response             │
│ Workflow logging                           │
└──────────────────────┬─────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────┐
│ clinical_graph.py                          │
│                                            │
│ LangGraph StateGraph                       │
│                                            │
│ Intake                                     │
│    ↓                                       │
│ Clinical Context                           │
│    ↓                                       │
│ Evidence / RAG                             │
│    ↓                                       │
│ Differential Diagnosis                    │
│    ↓                                       │
│ Missing Information                        │
│    ↓                                       │
│ Diagnostic Tests                           │
│    ↓                                       │
│ Safety                                     │
│    ↓                                       │
│ Clinical Critic                            │
│    ↓                                       │
│ Human Review                               │
└────────────────────────────────────────────┘
```

**The key principle:** `clinical_decision_impl.py` should know **how to execute the clinical AI workflow**, while `clinical_decision_service.py` knows **how the application coordinates the operation**, and `clinical_decision_router.py` knows **how HTTP exposes it**.

"""

from typing import Any, Dict

from app.config.settings import settings
logger = settings.get_logger()

from app.graph.clinical_graph import ClinicalGraph
from app.schemas.request_model import ClinicalDecisionRequest
from app.schemas.response_model import ClinicalDecisionResponse
import os
import uuid
import time

class ClinicalDecisionImplementation:
    """
    Implementation layer for the Clinical Decision Support system.

    Responsibilities:
        - Build LangGraph input state
        - Invoke LangGraph workflow
        - Handle workflow execution
        - Transform graph state into API response
        - Log workflow lifecycle events

    Does NOT handle:
        - HTTP routing
        - HTTP exceptions
        - FastAPI request handling
        - Business endpoint definitions
    """

    def __init__(
        self,
        clinical_graph: ClinicalGraph,
    ) -> None:
        """
        Initialize ClinicalDecisionImplementation.

        Args:
            clinical_graph:
                Compiled LangGraph clinical workflow.
        """

        self.clinical_graph = clinical_graph

        logger.debug(
            "ClinicalDecisionImplementation initialized | "
            "graph={}",
            clinical_graph.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    async def analyze(
        self,
        request: ClinicalDecisionRequest,
    ) -> ClinicalDecisionResponse:
        """
        Execute the clinical decision-support LangGraph workflow.

        Flow:

            Request
               ↓
            Build State
               ↓
            LangGraph
               ↓
            Intake Agent
               ↓
            Clinical Context Agent
               ↓
            Evidence/RAG Agent
               ↓
            Differential Diagnosis Agent
               ↓
            Missing Information Agent
               ↓
            Diagnostic Test Agent
               ↓
            Safety Agent
               ↓
            Clinical Critic
               ↓
            Summary
               ↓
            Response

        Args:
            request:
                Validated clinical decision request.

        Returns:
            ClinicalDecisionResponse:
                Structured clinical decision-support response.
        """

        patient_id = request.patient_id

        logger.info(
            "Clinical decision implementation started | "
            "patient_id={}",
            patient_id,
        )

        try:

            # ----------------------------------------------------------
            # 1. Build initial LangGraph state
            # ----------------------------------------------------------

            initial_state = self._build_initial_state(
                request
            )

            logger.debug(
                "Initial clinical graph state created | "
                "patient_id={}",
                patient_id,
            )

            # ----------------------------------------------------------
            # 2. Invoke LangGraph
            # ----------------------------------------------------------

            logger.info(
                "Invoking clinical LangGraph workflow | "
                "patient_id={}",
                patient_id,
            )
            request_id = str(uuid.uuid4())
            started_at = time.perf_counter()

            result = await self.clinical_graph.ainvoke(
                initial_state,
                config={
                    "run_name": "clinical-decision-support",
                    "tags": [
                        "clinical-decision-support",
                        "langgraph",
                        "openrouter",
                    ],
                    "metadata": {
                        "request_id": request_id,
                        "workflow": "clinical_decision_support",
                        "environment": os.getenv(
                            "APP_ENV", "development"
                        ),
                    },
                    "recursion_limit": int(
                        os.getenv("LANGGRAPH_RECURSION_LIMIT", "25")
                    ),
                },
            )

            logger.info(
                "Clinical LangGraph workflow completed | "
                "patient_id={}",
                patient_id,
            )

            # ----------------------------------------------------------
            # 3. Validate graph result
            # ----------------------------------------------------------

            self._validate_graph_result(
                result=result,
                patient_id=patient_id,
            )

            # ----------------------------------------------------------
            # 4. Transform graph state to API response
            # ----------------------------------------------------------

            response = self._build_response(
                result=result,
                patient_id=patient_id,
            )

            logger.info(
                "Clinical decision implementation completed | "
                "patient_id={}",
                patient_id,
            )

            return response

        except Exception as exc:

            logger.exception(
                "Clinical decision implementation failed | "
                "patient_id={} | error={}",
                patient_id,
                str(exc),
            )

            raise

    # ------------------------------------------------------------------
    # Build Initial State
    # ------------------------------------------------------------------

    @staticmethod
    def _build_initial_state(
        request: ClinicalDecisionRequest,
    ) -> Dict[str, Any]:
        """
        Convert API request into LangGraph state.

        Important:
            Do not pass the Pydantic request object directly into
            LangGraph. Explicitly construct the state so the graph
            remains independent from the API layer.

        Args:
            request:
                Clinical decision request.

        Returns:
            Initial ClinicalState dictionary.
        """

        return {
            "patient_id": request.patient_id,

            "symptoms": request.symptoms or {},

            "history": request.history or {},

            "vitals": request.vitals or {},

            "medications": request.medications or [],

            "allergies": request.allergies or [],

            "labs": request.labs or [],

            "imaging": request.imaging or [],

            # Workflow-generated fields
            "clinical_context": {},

            "evidence": [],

            "differential_diagnosis": [],

            "missing_information": [],

            "suggested_tests": [],

            "safety_flags": [],

            "critique": {},

            "final_summary": {},

            "requires_human_review": True,
        }

    # ------------------------------------------------------------------
    # Validate Graph Result
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_graph_result(
        result: Dict[str, Any],
        patient_id: str,
    ) -> None:
        """
        Validate the result returned by LangGraph.

        Args:
            result:
                Final LangGraph state.

            patient_id:
                Patient identifier used for logging.

        Raises:
            ValueError:
                If the workflow result is invalid or incomplete.
        """

        if not result:

            logger.error(
                "LangGraph returned empty result | "
                "patient_id={}",
                patient_id,
            )

            raise ValueError(
                "Clinical workflow returned an empty result."
            )

        if "final_summary" not in result:

            logger.error(
                "LangGraph result missing final_summary | "
                "patient_id={}",
                patient_id,
            )

            raise ValueError(
                "Clinical workflow did not produce a final summary."
            )

        logger.debug(
            "LangGraph result validation successful | "
            "patient_id={}",
            patient_id,
        )

    # ------------------------------------------------------------------
    # Build API Response
    # ------------------------------------------------------------------

    @staticmethod
    def _build_response(
        result: Dict[str, Any],
        patient_id: str,
    ) -> ClinicalDecisionResponse:
        """
        Transform LangGraph state into API response.

        Args:
            result:
                Final LangGraph state.

            patient_id:
                Patient identifier.

        Returns:
            ClinicalDecisionResponse.
        """

        final_summary = result.get(
            "final_summary",
            {},
        )

        return ClinicalDecisionResponse(
            patient_id=patient_id,

            clinical_summary=(
                final_summary.get(
                    "clinical_summary",
                    "",
                )
            ),

            differential_diagnoses=(
                result.get(
                    "differential_diagnosis",
                    [],
                )
            ),

            missing_information=(
                result.get(
                    "missing_information",
                    [],
                )
            ),

            suggested_tests=(
                result.get(
                    "suggested_tests",
                    [],
                )
            ),

            safety_flags=(
                result.get(
                    "safety_flags",
                    [],
                )
            ),

            evidence=(
                result.get(
                    "evidence",
                    [],
                )
            ),

            requires_human_review=(
                result.get(
                    "requires_human_review",
                    True,
                )
            ),

            workflow_id=(
                result.get(
                    "workflow_id",
                    "",
                )
            ),
        )
