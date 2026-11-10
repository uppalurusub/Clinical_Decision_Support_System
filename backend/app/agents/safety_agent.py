
"""
safety_agent.py

Clinical Safety / Red-Flag Agent.

Responsibilities:
    - Evaluate the current clinical state for safety concerns
    - Identify potential red flags
    - Identify urgent clinical concerns
    - Detect contradictions or unsafe workflow conditions
    - Combine deterministic safety rules with LLM analysis
    - Produce structured safety findings
    - Force clinician review when safety concerns exist

Architecture:

    ContextAgent
         ↓
    EvidenceAgent
         ↓
    DifferentialAgent
         ↓
    MissingInfoAgent
         ↓
    TestAgent
         ↓
    SafetyAgent
         ↓
    ClinicalCriticAgent
         ↓
    Human Clinical Review

Important:
    This agent is a clinical decision-support component.
    It does NOT independently diagnose, treat, or manage
    emergency patients.

```

### LangGraph integration

Register it after the `TestAgent`:

```python id="j6ez6d"
graph.add_node(
    "safety",
    safety_agent.node,
)

graph.add_edge(
    "diagnostic_tests",
    "safety",
)

graph.add_edge(
    "safety",
    "clinical_critic",
)
```

Your workflow is now:

```text id="xplh5q"
START
  │
  ▼
IntakeAgent
  │
  ▼
ContextAgent
  │
  ▼
EvidenceAgent
  │
  ▼
DifferentialAgent
  │
  ▼
MissingInfoAgent
  │
  ▼
TestAgent
  │
  ▼
┌─────────────────────┐
│     SafetyAgent     │
│                     │
│  ┌───────────────┐  │
│  │ Safety Rules  │  │
│  └───────┬───────┘  │
│          │          │
│          ▼          │
│    LLM Safety       │
│     Analysis        │
│          │          │
│          ▼          │
│    Merge Findings   │
└──────────┬──────────┘
           │
           ▼
    ClinicalCriticAgent
           │
      ┌────┴────┐
      │         │
     FAIL      PASS
      │         │
      ▼         ▼
 Differential  Summary
    Agent        │
                 ▼
          Human Review
                 │
                 ▼
                END
```

## Why I recommend hybrid safety

For this particular agent, I would **not** make the architecture:

```text
Patient
  ↓
LLM
  ↓
"Is this patient safe?"
```

Instead:

```text
                    Clinical State
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
      Deterministic Rules       LLM Analysis
             │                       │
             │                       │
             └───────────┬───────────┘
                         ▼
                  Safety Assessment
                         │
                         ▼
                   Human Review
```

This gives you two complementary mechanisms:

**Deterministic layer**

```text
Validated clinical rules
Validated clinical scores
Institutional safety policies
Data-quality checks
Medication/allergy rules
```

**LLM layer**

```text
Contextual reasoning
Cross-agent consistency
Explanation
Identification of potential overlooked concerns
```

The illustrative `default_safety_rules()` above is intentionally **not a production clinical rule set**. Replace those examples with rules that have been clinically validated and approved for your target population and workflow.

### Very important state-design point

The Safety Agent should be able to see the outputs of **all preceding agents**:

```text id="mqqk5m"
                  ClinicalState
                      │
       ┌──────────────┼───────────────┐
       ▼              ▼               ▼
    Context        Evidence       Differential
       │              │               │
       └──────────────┼───────────────┘
                      ▼
                Missing Info
                      │
                      ▼
                  Test Agent
                      │
                      ▼
                 SafetyAgent
```

That allows it to identify problems such as:

```text
Differential says: diagnosis A is plausible
        │
        ▼
MissingInfo says: critical information is unavailable
        │
        ▼
TestAgent says: additional diagnostic information
                may be useful
        │
        ▼
SafetyAgent:
    "Current information is insufficient for
     confident clinical assessment."
```

That is much more useful than simply having an LLM look at the original patient message.

### One additional recommendation

For your eventual `clinical_critic_agent.py`, have the critic explicitly challenge **both** the diagnostic reasoning and the safety assessment:

```text id="d6a1bf"
Clinical Critic
     │
     ├── Did we miss a differential?
     ├── Did we overstate evidence?
     ├── Did we treat missing data as negative?
     ├── Are citations relevant?
     ├── Did we overlook a safety flag?
     ├── Did an agent contradict another agent?
     └── Is clinician review mandatory?
```

This gives your overall system a **reason → challenge → revise → human-review** pattern, which is a much stronger Agentic AI architecture for clinical decision support than a simple linear sequence of LLM calls.

"""

from typing import Any, Callable, Dict, List, Optional

from langchain_openai import ChatOpenAI
from app.config.settings import settings
logger = settings.get_logger()

llm = settings.get_model_client()

from pydantic import BaseModel, Field


from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Structured Output Models
# ============================================================================


class SafetyFlag(BaseModel):
    """
    A potential clinical safety concern.
    """

    flag: str = Field(
        description="Name or description of the safety concern."
    )

    severity: str = Field(
        default="moderate",
        description=(
            "Qualitative severity such as critical, high, "
            "moderate, or low."
        ),
    )

    category: str = Field(
        default="clinical",
        description=(
            "Safety category such as emergency, vital_sign, "
            "laboratory, medication, allergy, diagnostic, "
            "data_quality, or workflow."
        ),
    )

    supporting_findings: List[str] = Field(
        default_factory=list,
        description=(
            "Available findings supporting the safety concern."
        ),
    )

    rationale: str = Field(
        description=(
            "Concise explanation of why the finding "
            "requires attention."
        ),
    )

    recommended_action: Optional[str] = Field(
        default=None,
        description=(
            "High-level safety-oriented action for clinician "
            "review. This is not an autonomous treatment order."
        ),
    )

    source: str = Field(
        default="llm",
        description=(
            "Source of the safety finding, such as "
            "deterministic_rule or llm."
        ),
    )


class SafetyAssessment(BaseModel):
    """
    Structured safety assessment.
    """

    safety_flags: List[SafetyFlag] = Field(
        default_factory=list,
        description="Identified safety concerns.",
    )

    immediate_attention_required: bool = Field(
        default=False,
        description=(
            "Whether the available information indicates "
            "a potentially urgent safety concern requiring "
            "immediate clinician attention."
        ),
    )

    data_quality_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "Data-quality or clinical-information concerns."
        ),
    )

    medication_safety_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "Potential medication-related safety concerns."
        ),
    )

    diagnostic_safety_concerns: List[str] = Field(
        default_factory=list,
        description=(
            "Potential diagnostic reasoning or workflow "
            "safety concerns."
        ),
    )

    overall_assessment: str = Field(
        default="",
        description="Concise overall safety assessment.",
    )

    requires_human_review: bool = Field(
        default=True,
        description="Whether clinician review is required.",
    )


# ============================================================================
# Deterministic Safety Rule
# ============================================================================


SafetyRule = Callable[
    [Dict[str, Any]],
    List[SafetyFlag],
]


def default_safety_rules(
    state: Dict[str, Any],
) -> List[SafetyFlag]:
    """
    Execute basic deterministic safety checks.

    IMPORTANT:
        These are illustrative framework rules, NOT validated
        clinical decision rules. They must be replaced or augmented
        with institutionally validated rules before production use.

    Args:
        state:
            Current LangGraph clinical state.

    Returns:
        List of safety flags.
    """

    flags: List[SafetyFlag] = []

    patient_id = state.get(
        "patient_id",
        "unknown",
    )

    vitals = state.get(
        "vitals",
        {},
    )

    allergies = state.get(
        "allergies",
        [],
    )

    medications = state.get(
        "medications",
        [],
    )

    # ------------------------------------------------------------------
    # Example: missing critical safety information
    # ------------------------------------------------------------------

    if not vitals:

        flags.append(
            SafetyFlag(
                flag="Vital signs unavailable",
                severity="high",
                category="data_quality",
                supporting_findings=[
                    "No vital-sign data available in current context."
                ],
                rationale=(
                    "Important clinical deterioration may not be "
                    "detectable without appropriate vital-sign information."
                ),
                recommended_action=(
                    "Clinician should verify current vital signs."
                ),
                source="deterministic_rule",
            )
        )

    # ------------------------------------------------------------------
    # Example: allergy information unavailable
    # ------------------------------------------------------------------

    if allergies is None:

        flags.append(
            SafetyFlag(
                flag="Allergy information unavailable",
                severity="high",
                category="allergy",
                supporting_findings=[
                    "Allergy information is unavailable."
                ],
                rationale=(
                    "Medication-related clinical decisions may be "
                    "unsafe when allergy information is unknown."
                ),
                recommended_action=(
                    "Clinician should verify allergy information."
                ),
                source="deterministic_rule",
            )
        )

    # ------------------------------------------------------------------
    # Example: medication information unavailable
    # ------------------------------------------------------------------

    if medications is None:

        flags.append(
            SafetyFlag(
                flag="Medication information unavailable",
                severity="moderate",
                category="medication",
                supporting_findings=[
                    "Current medication information is unavailable."
                ],
                rationale=(
                    "Medication reconciliation may be incomplete."
                ),
                recommended_action=(
                    "Clinician should verify current medications."
                ),
                source="deterministic_rule",
            )
        )

    logger.debug(
        "Deterministic safety rules evaluated | "
        "patient_id={} | flags={}",
        patient_id,
        len(flags),
    )

    return flags


# ============================================================================
# Safety Agent
# ============================================================================


class SafetyAgent:
    """
    Clinical Safety / Red-Flag Agent.

    Uses:

        Deterministic Safety Rules
                  +
             Clinical State
                  +
             LLM Analysis
                  ↓
             Safety Assessment

    The deterministic rules are intended to provide a
    predictable safety layer.

    The LLM provides contextual analysis but should not
    be considered the sole safety mechanism.
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        safety_rules: Optional[
            List[SafetyRule]
        ] = None,
    ) -> None:
        """
        Initialize SafetyAgent.

        Args:
            llm:
                LangChain-compatible ChatModel.

            safety_rules:
                Optional list of deterministic safety rules.
        """

        self.llm = llm

        self.safety_rules = (
            safety_rules
            if safety_rules is not None
            else [default_safety_rules]
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Clinical Safety Agent within a
clinician-supervised clinical decision-support system.

Your responsibility is to identify potential clinical
and workflow safety concerns from the available
patient information.

Review:

- Clinical context
- Vital signs
- Abnormal findings
- Differential diagnoses
- Missing information
- Diagnostic uncertainties
- Suggested diagnostic tests
- Medication information
- Allergy information
- Deterministic safety findings

Look for:

1. Potential red flags.
2. Potential urgent clinical concerns.
3. Dangerous diagnostic uncertainty.
4. Important missing safety information.
5. Medication-related concerns.
6. Allergy-related concerns.
7. Contradictory or inconsistent clinical information.
8. Unsafe assumptions.
9. Workflow or data-quality risks.

IMPORTANT:

- Do NOT make a definitive diagnosis.
- Do NOT prescribe treatment.
- Do NOT issue autonomous medical orders.
- Do NOT invent patient findings.
- Do NOT assume that missing information is normal.
- Do NOT fabricate clinical evidence.
- Do NOT override deterministic safety findings.
- Always require clinician review.
- Clearly distinguish potential safety concerns
  from confirmed clinical conditions.

If a potential urgent concern is identified, clearly
flag it for clinician attention.

Return structured output only.
""",
                ),
                (
                    "human",
                    """
PATIENT CLINICAL CONTEXT
------------------------

{clinical_context}


VITAL SIGNS
-----------

{vitals}


ABNORMAL FINDINGS
-----------------

{abnormal_findings}


RISK FACTORS
------------

{risk_factors}


DIFFERENTIAL DIAGNOSES
----------------------

{differential_diagnoses}


MISSING INFORMATION
-------------------

{missing_information}


CRITICAL INFORMATION GAPS
-------------------------

{critical_information_gaps}


DIAGNOSTIC UNCERTAINTIES
------------------------

{diagnostic_uncertainties}


SUGGESTED TESTS
---------------

{suggested_tests}


MEDICATIONS
-----------

{medications}


ALLERGIES
---------

{allergies}


DETERMINISTIC SAFETY FINDINGS
-----------------------------

{deterministic_flags}

Assess potential clinical safety concerns.
""",
                ),
            ]
        )

        # Explicitly raise the completion budget for structured output.
        # Confirm that the configured model/provider supports
        # max_completion_tokens. This does not override a provider-side
        # hard limit, deployment limit, or a conflicting model configuration.
        self.completion_token_budget = 3000
        self.budgeted_llm = self.llm.bind(
            max_completion_tokens=self.completion_token_budget
        )

        self.structured_llm = self.budgeted_llm.with_structured_output(
            SafetyAssessment
        )

        self.chain = self.prompt | self.structured_llm

        logger.debug(
            "SafetyAgent initialized | "
            "llm={} | safety_rules={}",
            llm.__class__.__name__,
            len(self.safety_rules),
        )

    # ------------------------------------------------------------------
    # Execute Deterministic Rules
    # ------------------------------------------------------------------

    def _run_safety_rules(
        self,
        state: Dict[str, Any],
    ) -> List[SafetyFlag]:
        """
        Execute all deterministic safety rules.

        Args:
            state:
                Current LangGraph clinical state.

        Returns:
            Safety flags.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        all_flags: List[SafetyFlag] = []

        logger.debug(
            "Running deterministic safety rules | "
            "patient_id={} | rules={}",
            patient_id,
            len(self.safety_rules),
        )

        for rule in self.safety_rules:

            try:

                flags = rule(state)

                if flags:
                    all_flags.extend(flags)

            except Exception as exc:

                logger.exception(
                    "Safety rule failed | "
                    "patient_id={} | rule={} | error={}",
                    patient_id,
                    getattr(
                        rule,
                        "__name__",
                        "unknown",
                    ),
                    str(exc),
                )

        logger.info(
            "Deterministic safety evaluation completed | "
            "patient_id={} | flags={}",
            patient_id,
            len(all_flags),
        )

        return all_flags

    # ------------------------------------------------------------------
    # Run Agent
    # ------------------------------------------------------------------

    async def run(
        self,
        state: Dict[str, Any],
    ) -> SafetyAssessment:
        """
        Execute the Safety Agent.

        Workflow:

            Clinical State
                  ↓
            Deterministic Rules
                  ↓
            LLM Safety Analysis
                  ↓
            Merge Safety Findings
                  ↓
            Safety Assessment
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.info(
            "SafetyAgent started | patient_id={}",
            patient_id,
        )

        try:

            # ----------------------------------------------------------
            # Step 1: Deterministic safety evaluation
            # ----------------------------------------------------------

            deterministic_flags = (
                self._run_safety_rules(state)
            )

            # ----------------------------------------------------------
            # Step 2: Gather clinical state
            # ----------------------------------------------------------

            clinical_context = state.get(
                "clinical_context",
                {},
            )

            vitals = state.get(
                "vitals",
                {},
            )

            abnormal_findings = state.get(
                "abnormal_findings",
                [],
            )

            risk_factors = state.get(
                "risk_factors",
                [],
            )

            differential_diagnoses = state.get(
                "differential_diagnosis",
                [],
            )

            missing_information = state.get(
                "missing_information",
                [],
            )

            critical_information_gaps = state.get(
                "critical_information_gaps",
                [],
            )

            diagnostic_uncertainties = state.get(
                "diagnostic_uncertainties",
                [],
            )

            suggested_tests = state.get(
                "suggested_tests",
                [],
            )

            medications = state.get(
                "medications",
                [],
            )

            allergies = state.get(
                "allergies",
                [],
            )

            # ----------------------------------------------------------
            # Step 3: LLM safety analysis
            # ----------------------------------------------------------

            logger.debug(
                "Running contextual safety analysis | "
                "patient_id={}",
                patient_id,
            )

            try:
                result = await self.chain.ainvoke(
                    {
                        "clinical_context": clinical_context,
                        "vitals": vitals,
                        "abnormal_findings": abnormal_findings,
                        "risk_factors": risk_factors,
                        "differential_diagnoses": differential_diagnoses,
                        "missing_information": missing_information,
                        "critical_information_gaps": critical_information_gaps,
                        "diagnostic_uncertainties": diagnostic_uncertainties,
                        "suggested_tests": suggested_tests,
                        "medications": medications,
                        "allergies": allergies,
                        "deterministic_flags": [
                            flag.model_dump()
                            for flag in deterministic_flags
                        ],
                    }
                )

                if not isinstance(result, SafetyAssessment):
                    result = SafetyAssessment.model_validate(result)

            except Exception as llm_exc:
                # Fail safely: retain deterministic findings and make the
                # incomplete LLM analysis explicit. Never convert an LLM
                # failure into a "no safety concerns" assessment.
                logger.exception(
                    "Safety LLM analysis failed; returning deterministic "
                    "safety assessment | error_type={}",
                    type(llm_exc).__name__,
                )

                high_or_critical = [
                    flag for flag in deterministic_flags
                    if str(flag.severity).lower() in {"critical", "high"}
                ]

                return SafetyAssessment(
                    safety_flags=list(deterministic_flags),
                    immediate_attention_required=bool(high_or_critical),
                    data_quality_concerns=[
                        "LLM safety analysis did not complete; "
                        "the assessment is based on deterministic findings only."
                    ],
                    medication_safety_concerns=[
                        flag.flag for flag in deterministic_flags
                        if str(flag.category).lower() == "medication"
                    ],
                    diagnostic_safety_concerns=[
                        flag.flag for flag in deterministic_flags
                        if str(flag.category).lower()
                        in {"diagnostic", "workflow"}
                    ],
                    overall_assessment=(
                        "INCOMPLETE SAFETY ASSESSMENT: contextual LLM analysis "
                        "failed or could not be parsed. Deterministic findings "
                        "are retained, but absence of additional flags must not "
                        "be interpreted as absence of clinical risk. Clinician "
                        "review is required."
                    ),
                    requires_human_review=True,
                )

            # ----------------------------------------------------------
            # Step 4: Merge deterministic findings
            # ----------------------------------------------------------

            existing_flags = {
                (
                    flag.flag,
                    flag.category,
                )
                for flag in result.safety_flags
            }

            for flag in deterministic_flags:

                key = (
                    flag.flag,
                    flag.category,
                )

                if key not in existing_flags:

                    result.safety_flags.append(
                        flag
                    )

            # ----------------------------------------------------------
            # Step 5: Enforce human review
            # ----------------------------------------------------------

            result.requires_human_review = True

            if result.safety_flags:
                result.requires_human_review = True

            # Deterministic critical/high findings should
            # always force attention.
            if any(
                flag.severity.lower()
                in {"critical", "high"}
                for flag in result.safety_flags
            ):
                result.immediate_attention_required = True

            logger.info(
                "SafetyAgent completed | "
                "patient_id={} | flags={} | "
                "immediate_attention={} | "
                "human_review={}",
                patient_id,
                len(result.safety_flags),
                result.immediate_attention_required,
                result.requires_human_review,
            )

            return result

        except Exception as exc:

            logger.exception(
                "SafetyAgent failed | "
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
                Current LangGraph clinical state.

        Returns:
            State updates containing safety assessment.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.debug(
            "SafetyAgent LangGraph node executing | "
            "patient_id={}",
            patient_id,
        )

        result = await self.run(state)

        return {
            "safety_flags": [
                flag.model_dump()
                for flag in result.safety_flags
            ],

            "immediate_attention_required": (
                result.immediate_attention_required
            ),

            "data_quality_concerns": (
                result.data_quality_concerns
            ),

            "medication_safety_concerns": (
                result.medication_safety_concerns
            ),

            "diagnostic_safety_concerns": (
                result.diagnostic_safety_concerns
            ),

            "safety_assessment": (
                result.overall_assessment
            ),

            # Safety concerns always require
            # clinician review.
            "requires_human_review": True,
        }