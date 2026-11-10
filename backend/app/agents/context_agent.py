"""
context_agent.py

Clinical Context Agent.

Responsibilities:
    - Consume structured intake information
    - Organize the patient's clinical context
    - Identify active clinical problems
    - Identify relevant risk factors
    - Identify clinically relevant history
    - Identify important abnormalities
    - Identify temporal relationships
    - Prepare context for downstream agents

Architecture:

    Intake Agent
         ↓
    Context Agent
         ↓
    Clinical Context
         ↓
    Evidence / RAG Agent
         ↓
    Differential Diagnosis Agent

Important:
    This agent does NOT make a final diagnosis or treatment decision.

```

## How `ContextAgent` fits into the workflow

Your first two agents now form:

```text
Raw Patient Data
       │
       ▼
┌──────────────────┐
│   IntakeAgent    │
│                  │
│ Extract /        │
│ Normalize        │
└────────┬─────────┘
         │
         ▼
   IntakeResult
         │
         ▼
┌──────────────────┐
│   ContextAgent   │
│                  │
│ Organize /       │
│ Contextualize    │
└────────┬─────────┘
         │
         ▼
 ClinicalContext
         │
         ▼
┌──────────────────┐
│ Evidence/RAG     │
│ Agent            │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Differential     │
│ Diagnosis Agent  │
└──────────────────┘
```

### Key distinction

The **Intake Agent** answers:

> **"What clinical information is present?"**

The **Context Agent** answers:

> **"How should the available clinical information be organized into a coherent patient context?"**

The **Evidence Agent** will then answer:

> **"What trusted clinical evidence is relevant to this context?"**

And the **Differential Diagnosis Agent** will answer:

> **"What diagnostic possibilities should the clinician consider based on the available evidence and patient context?"**

That separation prevents one giant LLM prompt from attempting extraction, contextualization, evidence retrieval, diagnosis, and decision-making simultaneously.

### LangGraph integration

Your graph can now be:

```python
graph.add_node(
    "intake",
    intake_agent.node,
)

graph.add_node(
    "context",
    context_agent.node,
)

graph.add_edge(
    START,
    "intake",
)

graph.add_edge(
    "intake",
    "context",
)

graph.add_edge(
    "context",
    "evidence",
)
```

So the workflow becomes:

```text
START
  │
  ▼
IntakeAgent
  │
  ▼
ContextAgent
  │
  ▼
Evidence/RAG Agent
  │
  ▼
Differential Diagnosis Agent
  │
  ▼
Missing Information Agent
  │
  ▼
Diagnostic Test Agent
  │
  ▼
Safety Agent
  │
  ▼
Clinical Critic
  │
  ├──── FAIL ────► Re-analysis
  │
  ▼
Summary
  │
  ▼
Human Clinical Review
  │
  ▼
END
```

**One schema change I recommend:** add `active_problems`, `risk_factors`, `abnormal_findings`, `temporal_events`, `information_gaps`, and `unresolved_issues` to your shared `ClinicalState`. This makes the Context Agent's output directly available to the Evidence, Differential, Safety, and Critic agents without forcing them to re-derive the same information.

"""

from typing import Any, Dict, List, Optional

from langchain_openai import ChatOpenAI
from app.config.settings import settings
logger = settings.get_logger()

llm = settings.get_model_client()

from pydantic import BaseModel, Field


from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Structured Output Models
# ============================================================================


class ActiveProblem(BaseModel):
    """A clinically relevant active problem."""

    problem: str = Field(
        description="Name or description of the active clinical problem."
    )

    evidence: List[str] = Field(
        default_factory=list,
        description="Clinical findings supporting the problem."
    )

    status: Optional[str] = Field(
        default=None,
        description=(
            "Current status such as active, worsening, "
            "stable, or unresolved when supported by the input."
        ),
    )


class RiskFactor(BaseModel):
    """A clinically relevant risk factor."""

    factor: str = Field(
        description="Name of the risk factor."
    )

    relevance: Optional[str] = Field(
        default=None,
        description="Why the risk factor may be clinically relevant."
    )


class ClinicalAbnormality(BaseModel):
    """Clinically relevant abnormal finding."""

    finding: str = Field(
        description="Abnormal clinical finding."
    )

    source: Optional[str] = Field(
        default=None,
        description="Source such as vital, laboratory, imaging, history."
    )

    significance: Optional[str] = Field(
        default=None,
        description=(
            "Clinical significance when explicitly supported "
            "by available information."
        ),
    )


class TemporalEvent(BaseModel):
    """Important clinical event or temporal relationship."""

    event: str = Field(
        description="Clinical event."
    )

    timing: Optional[str] = Field(
        default=None,
        description="Timing or duration if available."
    )

    relationship: Optional[str] = Field(
        default=None,
        description="Relationship to other clinical findings."
    )


class ClinicalContextResult(BaseModel):
    """Structured clinical context produced by the agent."""

    patient_overview: str = Field(
        default="",
        description=(
            "Concise summary of the patient's relevant "
            "clinical presentation."
        ),
    )

    active_problems: List[ActiveProblem] = Field(
        default_factory=list,
        description="Clinically relevant active problems."
    )

    relevant_history: Dict[str, Any] = Field(
        default_factory=dict,
        description="Relevant historical information."
    )

    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Relevant clinical risk factors."
    )

    medications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Medications relevant to the clinical context."
    )

    allergies: List[str] = Field(
        default_factory=list,
        description="Known allergies."
    )

    abnormal_findings: List[ClinicalAbnormality] = Field(
        default_factory=list,
        description="Clinically relevant abnormal findings."
    )

    temporal_events: List[TemporalEvent] = Field(
        default_factory=list,
        description="Important temporal relationships."
    )

    unresolved_issues: List[str] = Field(
        default_factory=list,
        description="Clinical issues that remain unresolved."
    )

    information_gaps: List[str] = Field(
        default_factory=list,
        description="Important information gaps."
    )


# ============================================================================
# Context Agent
# ============================================================================


class ContextAgent:
    """
    Clinical Context Agent.

    Converts normalized intake information into a coherent
    clinical context for downstream reasoning agents.

    The agent does NOT:

        - Make a definitive diagnosis
        - Prescribe medication
        - Order tests
        - Make autonomous treatment decisions
        - Replace clinician review
    """

    def __init__(
        self,
        llm: ChatOpenAI,
    ) -> None:
        """
        Initialize ContextAgent.

        Args:
            llm:
                LangChain-compatible ChatModel.
        """

        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Clinical Context Agent in a
clinical decision-support system.

Your task is to transform structured patient intake
information into a clinically coherent context that
can be consumed by downstream Evidence/RAG and
Differential Diagnosis agents.

Your responsibilities:

1. Summarize the patient's current clinical presentation.
2. Identify active clinical problems.
3. Identify relevant medical history.
4. Identify clinically relevant risk factors.
5. Identify relevant medications.
6. Identify allergies.
7. Identify clinically important abnormal findings.
8. Identify temporal relationships.
9. Identify unresolved clinical issues.
10. Identify important information gaps.

Rules:

- Do NOT make a definitive diagnosis.
- Do NOT prescribe treatment.
- Do NOT invent clinical information.
- Do NOT assume that an abnormal value has a specific
  diagnosis unless supported by the supplied information.
- Preserve the distinction between observed findings
  and clinical interpretation.
- Use only information supplied in the patient context.
- Do not hallucinate missing history.
- If information is missing, explicitly identify the gap.
- Prioritize clinically relevant information.
- Avoid unnecessary repetition.
- The output will be consumed by other clinical agents.

Return structured output only.
""",
                ),
                (
                    "human",
                    """
Structured Clinical Intake:

{intake_result}

Additional Clinical Context:

{clinical_context}

Patient Information:

Symptoms:
{symptoms}

History:
{history}

Vitals:
{vitals}

Medications:
{medications}

Allergies:
{allergies}

Laboratory Results:
{labs}

Imaging:
{imaging}

Create a structured clinical context from the
available information.
""",
                ),
            ]
        )

        self.structured_llm = self.llm.with_structured_output(
            ClinicalContextResult
        )

        self.chain = (
            self.prompt
            | self.structured_llm
        )

        logger.debug(
            "ContextAgent initialized | llm={}",
            llm.__class__.__name__,
        )

    # ------------------------------------------------------------------
    # Run Agent
    # ------------------------------------------------------------------

    async def run(
        self,
        state: Dict[str, Any],
    ) -> ClinicalContextResult:
        """
        Execute the Context Agent.

        Args:
            state:
                Current LangGraph clinical state.

        Returns:
            ClinicalContextResult:
                Structured clinical context.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.info(
            "ContextAgent started | patient_id={}",
            patient_id,
        )

        try:

            intake_result = state.get(
                "intake_result",
                {},
            )

            clinical_context = state.get(
                "clinical_context",
                {},
            )

            result = await self.chain.ainvoke(
                {
                    "intake_result": intake_result,

                    "clinical_context": clinical_context,

                    "symptoms": state.get(
                        "symptoms",
                        {},
                    ),

                    "history": state.get(
                        "history",
                        {},
                    ),

                    "vitals": state.get(
                        "vitals",
                        {},
                    ),

                    "medications": state.get(
                        "medications",
                        [],
                    ),

                    "allergies": state.get(
                        "allergies",
                        [],
                    ),

                    "labs": state.get(
                        "labs",
                        [],
                    ),

                    "imaging": state.get(
                        "imaging",
                        [],
                    ),
                }
            )

            logger.info(
                "ContextAgent completed | "
                "patient_id={} | active_problems={} | "
                "risk_factors={} | abnormalities={}",
                patient_id,
                len(result.active_problems),
                len(result.risk_factors),
                len(result.abnormal_findings),
            )

            return result

        except Exception as exc:

            logger.exception(
                "ContextAgent failed | "
                "patient_id={} | error={}",
                patient_id,
                str(exc),
            )

            raise

    # ------------------------------------------------------------------
    # LangGraph Node
    # ------------------------------------------------------------------

    async def node(
        self,
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        LangGraph node wrapper.

        Converts ContextAgent output into updates
        to the shared ClinicalState.

        Args:
            state:
                Current LangGraph state.

        Returns:
            State updates.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.debug(
            "ContextAgent LangGraph node executing | "
            "patient_id={}",
            patient_id,
        )

        result = await self.run(state)

        context_dict = result.model_dump()

        return {
            "clinical_context": context_dict,

            "active_problems": [
                problem.model_dump()
                for problem in result.active_problems
            ],

            "risk_factors": [
                factor.model_dump()
                for factor in result.risk_factors
            ],

            "abnormal_findings": [
                finding.model_dump()
                for finding in result.abnormal_findings
            ],

            "temporal_events": [
                event.model_dump()
                for event in result.temporal_events
            ],

            "information_gaps": (
                result.information_gaps
            ),

            "unresolved_issues": (
                result.unresolved_issues
            ),
        }
