"""
differential_agent.py

Clinical Differential Diagnosis Agent.

Responsibilities:
    - Analyze structured clinical context
    - Analyze retrieved clinical evidence
    - Generate a structured differential diagnosis
    - Identify supporting findings
    - Identify contradicting findings
    - Identify evidence supporting each possibility
    - Identify missing information
    - Flag diagnostic uncertainty

Architecture:

    ContextAgent
         ↓
    EvidenceAgent
         ↓
    DifferentialAgent
         ↓
    MissingInformationAgent
         ↓
    SafetyAgent
         ↓
    ClinicalCriticAgent

Important:
    This agent provides decision support.
    It does NOT make the final clinical diagnosis.
    Final clinical decisions remain with an appropriately
    qualified clinician.

```

## How it fits into your current workflow

You now have three major reasoning stages:

```text
                    Patient Data
                         │
                         ▼
                  ┌─────────────┐
                  │ IntakeAgent │
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ ContextAgent│
                  └──────┬──────┘
                         │
                         ▼
                  ┌─────────────┐
                  │ Evidence    │
                  │ Agent / RAG │
                  └──────┬──────┘
                         │
                         ▼
               ┌──────────────────┐
               │ DifferentialAgent│
               └────────┬─────────┘
                        │
                        ▼
              Candidate Diagnoses
                        │
                        ▼
             Missing Information
                        │
                        ▼
                 Safety Agent
                        │
                        ▼
               Clinical Critic
                        │
                        ▼
                Human Clinician
```

### LangGraph registration

```python
graph.add_node(
    "differential",
    differential_agent.node,
)

graph.add_edge(
    "evidence",
    "differential",
)

graph.add_edge(
    "differential",
    "missing_information",
)
```

## Why this agent should consume EvidenceAgent output

Avoid this architecture:

```text
Clinical Context
      ↓
LLM
      ↓
"Give me the diagnosis"
```

Instead:

```text
Clinical Context
       │
       ├──────────────┐
       │              │
       ▼              ▼
 EvidenceAgent    Patient Data
       │              │
       └───────┬──────┘
               ▼
       DifferentialAgent
               │
               ▼
       Structured Differential
```

This creates an important separation between:

**Evidence retrieval**

and

**Clinical reasoning**.

## Important point about `confidence`

I intentionally use:

```python
confidence: str
```

rather than:

```python
confidence: float
```

and the prompt explicitly states:

> "This is NOT a clinical probability."

An LLM saying:

```text
confidence = 0.91
```

does **not** mean that the patient has a 91% probability of that diagnosis.

If your system eventually needs actual risk/probability estimates, those should come from a **validated clinical prediction model or validated clinical scoring system**, with appropriate calibration and validation, rather than treating LLM output as a probability.

## Recommended next LangGraph stage

The next agent should be `missing_information_agent.py`.

Its job is different from `ContextAgent`:

```text
ContextAgent
    ↓
"What information do we currently have?"

DifferentialAgent
    ↓
"What diagnoses could explain it?"

MissingInformationAgent
    ↓
"What additional information could distinguish
 between these possibilities?"
```

That gives you a much stronger diagnostic workflow:

```text
Intake
  ↓
Context
  ↓
Evidence/RAG
  ↓
Differential
  ↓
Missing Information
  ↓
Diagnostic Test/Next-Step Support
  ↓
Safety
  ↓
Critic
  ↓
Human Review
```

And throughout the workflow, **Loguru should log workflow/agent lifecycle information—not raw clinical payloads or PHI**.

"""

from typing import Any, Dict, List, Optional
import time
from langchain_openai import ChatOpenAI
from app.config.settings import settings
logger = settings.get_logger()

llm = settings.get_model_client()

from pydantic import BaseModel, Field


from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Structured Output Models
# ============================================================================


class DifferentialItem(BaseModel):
    """One candidate diagnosis in the differential."""

    diagnosis: str = Field(
        description=(
            "Candidate diagnosis or clinical condition "
            "to consider."
        )
    )

    supporting_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Patient findings that support consideration "
            "of this diagnosis."
        )
    )

    contradicting_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Patient findings that may argue against "
            "this diagnosis."
        )
    )

    relevant_evidence: List[str] = Field(
        default_factory=list,
        description=(
            "Evidence statements from retrieved clinical "
            "sources relevant to this diagnosis."
        )
    )

    missing_information: List[str] = Field(
        default_factory=list,
        description=(
            "Additional information that could help "
            "differentiate this diagnosis."
        )
    )

    reasoning: str = Field(
        default="",
        description=(
            "Concise explanation of why this diagnosis "
            "should be considered."
        )
    )

    confidence: str = Field(
        default="uncertain",
        description=(
            "Qualitative reasoning confidence such as "
            "high, moderate, low, or uncertain. "
            "This is NOT a clinical probability."
        )
    )


class DifferentialResult(BaseModel):
    """Structured output from the Differential Diagnosis Agent."""

    differential_diagnoses: List[DifferentialItem] = Field(
        default_factory=list,
        description=(
            "Candidate diagnoses or clinical conditions "
            "that should be considered."
        )
    )

    key_discriminating_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Findings that may help distinguish among "
            "the differential possibilities."
        )
    )

    diagnostic_uncertainties: List[str] = Field(
        default_factory=list,
        description=(
            "Important uncertainties in the current "
            "clinical assessment."
        )
    )

    overall_reasoning: str = Field(
        default="",
        description=(
            "Concise synthesis of the differential "
            "diagnostic reasoning."
        )
    )

    requires_clinician_review: bool = Field(
        default=True,
        description=(
            "Whether clinician review is required."
        )
    )


# ============================================================================
# Differential Diagnosis Agent
# ============================================================================


class DifferentialAgent:
    """
    Clinical Differential Diagnosis Agent.

    Combines:

        Clinical Context
              +
        Retrieved Evidence
              +
        Patient Findings
              ↓
        Differential Diagnosis

    The agent does NOT:

        - Make a definitive diagnosis
        - Prescribe treatment
        - Order tests autonomously
        - Override clinician judgment
        - Convert LLM confidence into clinical probability
    """

    def __init__(
        self,
        llm: ChatOpenAI,
    ) -> None:
        """
        Initialize DifferentialAgent.

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
You are a Clinical Differential Diagnosis Agent
within a clinician-supervised clinical decision-support
system.

Your task is to generate a structured differential diagnosis
from:

1. Patient clinical context
2. Patient findings
3. Retrieved clinical evidence
4. Known information gaps

Your responsibilities:

- Identify plausible diagnoses or clinical conditions
  that should be considered.
- Explain why each possibility is being considered.
- Identify supporting findings.
- Identify contradicting findings.
- Link reasoning to retrieved evidence.
- Identify information that could discriminate between
  competing possibilities.
- Explicitly identify uncertainty.
- Avoid unsupported assumptions.

IMPORTANT SAFETY RULES:

- Do NOT claim that any candidate diagnosis is confirmed.
- Do NOT provide a definitive diagnosis.
- Do NOT prescribe treatment.
- Do NOT independently order tests.
- Do NOT fabricate clinical findings.
- Do NOT fabricate evidence or citations.
- Use only information provided in the context and evidence.
- Do not treat LLM confidence as a clinical probability.
- If information is insufficient, explicitly state that.
- Always require clinician review.

The purpose is to support clinician reasoning,
not replace clinical judgment.

Return structured output only.
""",
                ),
                (
                    "human",
                    """
PATIENT CLINICAL CONTEXT
------------------------

{clinical_context}


ACTIVE PROBLEMS
---------------

{active_problems}


RISK FACTORS
------------

{risk_factors}


ABNORMAL FINDINGS
-----------------

{abnormal_findings}


UNRESOLVED ISSUES
-----------------

{unresolved_issues}


INFORMATION GAPS
----------------

{information_gaps}


RETRIEVED CLINICAL EVIDENCE
---------------------------

{evidence}


EVIDENCE GAPS
-------------

{evidence_gaps}


Generate a structured differential diagnosis.
""",
                ),
            ]
        )

        # Request a larger completion budget for structured output.
        # Confirm the configured model/provider supports this parameter.
        self.completion_token_budget = 3000
        self.budgeted_llm = self.llm.bind(
            max_tokens=self.completion_token_budget
        )
        self.structured_llm = self.budgeted_llm.with_structured_output(
            DifferentialResult
        )
        self.chain = self.prompt | self.structured_llm

        logger.info(
            "DifferentialAgent configured | "
            "completion_token_budget={}",
            self.completion_token_budget,
        )

        logger.debug(
            "DifferentialAgent initialized | llm={}",
            llm.__class__.__name__,
        )

    @staticmethod
    def _compact_value(
        value: Any,
        max_items: int = 8,
        max_chars: int = 1500,
    ) -> str:
        """Render bounded retry context; does not validate clinical content."""
        try:
            if value is None:
                text_value = "Not provided."
            elif isinstance(value, str):
                text_value = value
            elif isinstance(value, (list, tuple)):
                selected = list(value)[:max_items]
                text_value = "\\n".join(str(item) for item in selected)
                if len(value) > max_items:
                    text_value += (
                        f"\\n[Truncated list: {len(value) - max_items} "
                        "additional item(s) omitted from retry context.]"
                    )
            elif isinstance(value, dict):
                pairs = list(value.items())[:max_items]
                text_value = "\\n".join(
                    f"{key}: {item}" for key, item in pairs
                )
                if len(value) > max_items:
                    text_value += (
                        f"\\n[Truncated mapping: {len(value) - max_items} "
                        "additional field(s) omitted from retry context.]"
                    )
            else:
                text_value = str(value)
        except Exception:
            text_value = "[Value could not be rendered for compact retry.]"

        if len(text_value) > max_chars:
            text_value = text_value[:max_chars] + " [truncated]"
        return text_value

    # ------------------------------------------------------------------
    # Run Agent
    # ------------------------------------------------------------------

    async def run(
        self,
        state: Dict[str, Any],
    ) -> DifferentialResult:
        """
        Execute the Differential Diagnosis Agent.

        Args:
            state:
                Current LangGraph clinical state.

        Returns:
            DifferentialResult.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.info(
            "DifferentialAgent started | patient_id={}",
            patient_id,
        )

        try:

            # ----------------------------------------------------------
            # Gather state
            # ----------------------------------------------------------

            clinical_context = state.get(
                "clinical_context",
                {},
            )

            active_problems = state.get(
                "active_problems",
                [],
            )

            risk_factors = state.get(
                "risk_factors",
                [],
            )

            abnormal_findings = state.get(
                "abnormal_findings",
                [],
            )

            unresolved_issues = state.get(
                "unresolved_issues",
                [],
            )

            information_gaps = state.get(
                "information_gaps",
                [],
            )

            evidence = state.get(
                "evidence",
                [],
            )

            evidence_gaps = state.get(
                "evidence_gaps",
                [],
            )

            logger.debug(
                "DifferentialAgent input prepared | "
                "patient_id={} | evidence_count={} | "
                "information_gaps={}",
                patient_id,
                len(evidence),
                len(information_gaps),
            )

            # ----------------------------------------------------------
            # Invoke LLM
            # ----------------------------------------------------------

            chain_input = {
                "clinical_context": clinical_context,
                "active_problems": active_problems,
                "risk_factors": risk_factors,
                "abnormal_findings": abnormal_findings,
                "unresolved_issues": unresolved_issues,
                "information_gaps": information_gaps,
                "evidence": evidence,
                "evidence_gaps": evidence_gaps,
            }

            try:
                result = await self.chain.ainvoke(chain_input)
                if not isinstance(result, DifferentialResult):
                    result = DifferentialResult.model_validate(result)

            except Exception as first_exc:
                logger.warning(
                    "Differential structured output failed; attempting "
                    "one compact retry | error_type={}",
                    type(first_exc).__name__,
                )

                compact_input = {
                    "clinical_context": self._compact_value(
                        clinical_context, max_chars=2500
                    ),
                    "active_problems": self._compact_value(
                        active_problems, max_items=8, max_chars=1200
                    ),
                    "risk_factors": self._compact_value(
                        risk_factors, max_items=8, max_chars=1000
                    ),
                    "abnormal_findings": self._compact_value(
                        abnormal_findings, max_items=10, max_chars=1600
                    ),
                    "unresolved_issues": self._compact_value(
                        unresolved_issues, max_items=8, max_chars=1000
                    ),
                    "information_gaps": self._compact_value(
                        information_gaps, max_items=8, max_chars=1000
                    ),
                    "evidence": self._compact_value(
                        evidence, max_items=5, max_chars=2200
                    ),
                    "evidence_gaps": self._compact_value(
                        evidence_gaps, max_items=6, max_chars=800
                    ),
                }

                try:
                    started = time.perf_counter()

                    try:
                        result = await self.chain.ainvoke(compact_input)
                        if not isinstance(result, DifferentialResult):
                            result = DifferentialResult.model_validate(result)
                    finally:
                        logger.info(
                            "Differential primary attempt duration_seconds={:.3f}",
                            time.perf_counter() - started,
                        )

                    if not isinstance(result, DifferentialResult):
                        result = DifferentialResult.model_validate(result)

                except Exception as retry_exc:
                    logger.exception(
                        "Differential compact retry failed; returning "
                        "explicitly incomplete result | error_type={}",
                        type(retry_exc).__name__,
                    )

                    existing = state.get("differential_diagnosis", [])
                    preserved_items = []
                    if isinstance(existing, list):
                        for item in existing[:8]:
                            try:
                                if isinstance(item, DifferentialItem):
                                    preserved_items.append(item)
                                elif isinstance(item, dict):
                                    preserved_items.append(
                                        DifferentialItem.model_validate(item)
                                    )
                            except Exception:
                                continue

                    return DifferentialResult(
                        differential_diagnoses=preserved_items,
                        key_discriminating_findings=[],
                        diagnostic_uncertainties=[
                            "Differential LLM analysis did not complete "
                            "successfully. The differential is incomplete "
                            "and requires clinician assessment."
                        ],
                        overall_reasoning=(
                            "INCOMPLETE DIFFERENTIAL ASSESSMENT: structured "
                            "LLM output failed, including the bounded retry. "
                            "Any preserved prior candidates are not a new "
                            "validated assessment. Do not interpret missing "
                            "or absent candidates as evidence against a "
                            "diagnosis."
                        ),
                        requires_clinician_review=True,
                    )

            # ----------------------------------------------------------
            # Enforce clinician review
            # ----------------------------------------------------------

            result.requires_clinician_review = True

            logger.info(
                "DifferentialAgent completed | "
                "patient_id={} | candidates={} | "
                "uncertainties={}",
                patient_id,
                len(result.differential_diagnoses),
                len(result.diagnostic_uncertainties),
            )

            return result

        except Exception as exc:

            logger.exception(
                "DifferentialAgent failed | "
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

        Args:
            state:
                Current LangGraph state.

        Returns:
            State updates containing differential diagnosis.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.debug(
            "DifferentialAgent LangGraph node executing | "
            "patient_id={}",
            patient_id,
        )

        result = await self.run(state)

        differential_items = [
            item.model_dump()
            for item in result.differential_diagnoses
        ]

        return {
            "differential_diagnosis": (
                differential_items
            ),

            "key_discriminating_findings": (
                result.key_discriminating_findings
            ),

            "diagnostic_uncertainties": (
                result.diagnostic_uncertainties
            ),

            "diagnostic_reasoning": (
                result.overall_reasoning
            ),

            # Always keep human review enabled.
            "requires_human_review": True,
        }
