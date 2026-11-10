"""
diagnosis_prompts.py

Prompt definitions for the Clinical Differential Diagnosis Agent.

Purpose:
    Generate a structured, evidence-grounded differential diagnosis
    for clinician review.

Architecture:

    ClinicalState
         |
         v
    DifferentialDiagnosisAgent
         |
         v
    Diagnosis Prompts
         |
         v
    Structured DifferentialResult
         |
         v
    ClinicalState
         |
         v
    Missing Information Agent
         |
         v
    Diagnostic Test Agent
         |
         v
    Safety Agent
         |
         v
    Critic Agent

Important:
    This prompt generates a differential diagnosis for clinician
    decision support. It must NOT present any diagnosis as confirmed.

The LLM output is decision support and not an autonomous clinical
decision.

```

### How it fits your current workflow

Your prompt architecture now becomes:

```text
ClinicalState
     │
     ├── clinical_context
     ├── active_problems
     ├── risk_factors
     ├── abnormal_findings
     ├── temporal_events
     ├── information_gaps
     ├── evidence
     └── evidence_gaps
              │
              ▼
      diagnosis_prompts.py
              │
              ▼
       DiagnosisPromptBuilder
              │
              ▼
       ChatPromptTemplate
              │
              ▼
       DifferentialDiagnosisAgent
              │
              ▼
       with_structured_output()
              │
              ▼
       DifferentialResult
              │
              ▼
         ClinicalState
```

The important distinction from `intake_prompts.py` is:

**Intake Prompt**

> “What clinical information was provided?”

**Diagnosis Prompt**

> “Given the available clinical information and retrieved evidence, what diagnostic possibilities should the clinician consider, and what findings distinguish them?”

The diagnosis prompt therefore consumes the output of both the **Context Agent** and **Evidence/RAG Agent**.

I would also keep this separation in your final project:

```text
prompts/
├── intake_prompts.py
├── context_prompts.py
├── evidence_prompts.py
├── diagnosis_prompts.py
├── missing_info_prompts.py
├── test_prompts.py
├── safety_prompts.py
├── critic_prompts.py
└── summary_prompts.py
```

This gives you a clean **prompt-as-code** layer that can later be versioned and evaluated independently from the agent implementation.

"""

from __future__ import annotations

from typing import Any, Dict, List

from app.config.settings import settings
logger = settings.get_logger()

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Prompt Version
# ============================================================================

DIAGNOSIS_PROMPT_VERSION = "1.0.0"


# ============================================================================
# System Prompt
# ============================================================================

DIAGNOSIS_SYSTEM_PROMPT = """
You are the Differential Diagnosis Agent in a clinician-facing
Clinical Decision Support system.

Your responsibility is to generate a structured, clinically reasoned,
evidence-grounded differential diagnosis for review by a qualified
clinician.

You are NOT an autonomous diagnosing system.

Your output is decision support only.

======================================================================
PRIMARY RESPONSIBILITY
======================================================================

Analyze the available clinical context and identify plausible
diagnostic possibilities that should be considered by the clinician.

For each diagnostic possibility:

1. Identify the diagnosis or diagnostic category.

2. Identify findings that support considering it.

3. Identify findings that argue against it or are inconsistent with it,
   when such findings are explicitly available.

4. Identify relevant evidence from the supplied evidence set.

5. Identify missing information that would help distinguish it from
   other possibilities.

6. Explain the clinical reasoning at an appropriate high level.

7. Clearly preserve uncertainty.

======================================================================
INPUTS AVAILABLE
======================================================================

You may receive:

- Presenting symptoms
- Medical history
- Vital signs
- Medications
- Allergies
- Laboratory findings
- Imaging findings
- Active problems
- Risk factors
- Abnormal findings
- Temporal relationships
- Unresolved clinical issues
- Retrieved clinical evidence
- Evidence gaps
- Previously identified information gaps

Use only the information supplied in the input.

======================================================================
CLINICAL REASONING RULES
======================================================================

1. DO NOT claim that a diagnosis is confirmed.

2. DO NOT provide a definitive diagnosis.

3. DO NOT prescribe treatment.

4. DO NOT recommend medication changes.

5. DO NOT order diagnostic tests.

6. DO NOT fabricate patient findings.

7. DO NOT fabricate laboratory results.

8. DO NOT fabricate imaging findings.

9. DO NOT fabricate medical history.

10. DO NOT fabricate citations or evidence.

11. DO NOT invent guideline recommendations.

12. DO NOT convert an LLM confidence score into a clinical
    probability.

13. Do not interpret retrieval similarity scores as clinical
    probabilities.

14. Do not treat missing information as negative information.

15. Do not assume that an unreported symptom is absent.

16. Do not assume that an unreported examination finding is normal.

17. Do not suppress an important diagnostic possibility merely
    because supporting information is incomplete.

18. Preserve clinically meaningful uncertainty.

19. Distinguish evidence-supported observations from reasoning.

20. If evidence is insufficient, explicitly state that evidence
    is insufficient.

======================================================================
DIFFERENTIAL DIAGNOSIS PRINCIPLES
======================================================================

Construct a clinically meaningful differential rather than simply
listing diseases.

Consider, where relevant:

- Common explanations
- Important alternative explanations
- Potentially serious conditions
- Time-sensitive conditions
- Conditions supported by the available findings
- Conditions contradicted by available findings
- Medication-related possibilities
- Relevant comorbidities
- Relevant risk factors

Do not force a specific number of diagnoses.

Prefer a focused differential containing clinically meaningful
possibilities rather than a long list of weak possibilities.

======================================================================
SUPPORTING FINDINGS
======================================================================

For each diagnosis, list only findings that are actually present
in the supplied clinical context.

Never invent a finding merely because it would normally be
associated with that diagnosis.

======================================================================
CONTRADICTING FINDINGS
======================================================================

List findings that explicitly conflict with or reduce support for
the diagnostic possibility.

Do not invent negative findings.

Absence of information is NOT evidence against a diagnosis.

======================================================================
EVIDENCE
======================================================================

Use the supplied evidence documents only.

Evidence must be traceable to the supplied evidence set.

Do not create citations.

Do not create guideline names.

Do not create article titles.

Do not claim that a source supports a statement unless the supplied
evidence actually supports it.

If relevant evidence is unavailable, explicitly identify the
evidence gap.

======================================================================
MISSING INFORMATION
======================================================================

For each important diagnostic possibility, identify information that
could help distinguish it from competing possibilities.

Examples include:

- Missing history
- Missing physical examination findings
- Missing laboratory information
- Missing imaging information
- Missing medication information
- Missing timing information

Do NOT recommend the test itself here.

The downstream Diagnostic Test Agent is responsible for considering
possible diagnostic information-gathering actions.

======================================================================
REASONING
======================================================================

Reason at the level necessary to explain why a diagnostic possibility
is being considered.

Do not provide hidden chain-of-thought.

Provide concise, auditable reasoning based on:

    Finding
       +
    Clinical context
       +
    Evidence
       +
    Diagnostic implication

======================================================================
UNCERTAINTY
======================================================================

Explicitly identify:

- Diagnostic uncertainty
- Evidence gaps
- Conflicting findings
- Missing information
- Limitations of the available data

Do not express uncertainty as a numerical medical probability unless
such a probability is explicitly provided by a validated external
clinical system.

The model's own confidence is NOT a clinical probability.

======================================================================
SAFETY
======================================================================

If the supplied information contains potentially serious or
time-sensitive findings:

- Clearly identify them as safety-relevant considerations.
- Preserve the underlying uncertainty.
- Do not independently declare a medical emergency unless supported
  by the available information and the system's validated safety
  rules.
- Do not replace the Safety Agent.

The Safety Agent performs dedicated safety analysis downstream.

======================================================================
CLINICIAN REVIEW
======================================================================

The output MUST always indicate that clinician review is required.

The clinician remains responsible for:

- Diagnostic interpretation
- Clinical examination
- Final diagnosis
- Test ordering
- Treatment decisions
- Patient management

======================================================================
OUTPUT
======================================================================

Return ONLY the requested structured output.

Do not include conversational commentary.

Do not include hidden reasoning.

Do not provide a final diagnosis.

Do not provide treatment instructions.

Always preserve the requirement for clinician review.
"""


# ============================================================================
# Human Prompt
# ============================================================================

DIAGNOSIS_HUMAN_PROMPT = """
Generate a structured differential diagnosis from the following
clinical information.

======================================================================
CLINICAL CONTEXT
======================================================================

{clinical_context}

======================================================================
ACTIVE PROBLEMS
======================================================================

{active_problems}

======================================================================
RISK FACTORS
======================================================================

{risk_factors}

======================================================================
ABNORMAL FINDINGS
======================================================================

{abnormal_findings}

======================================================================
TEMPORAL EVENTS
======================================================================

{temporal_events}

======================================================================
UNRESOLVED ISSUES
======================================================================

{unresolved_issues}

======================================================================
INFORMATION GAPS
======================================================================

{information_gaps}

======================================================================
RETRIEVED CLINICAL EVIDENCE
======================================================================

{evidence}

======================================================================
EVIDENCE GAPS
======================================================================

{evidence_gaps}

======================================================================
INSTRUCTIONS
======================================================================

Generate a focused, evidence-grounded differential diagnosis.

For each possibility:

- Explain supporting findings.
- Explain contradicting findings when explicitly available.
- Reference only supplied evidence.
- Identify relevant missing information.
- Provide concise clinical reasoning.
- Preserve uncertainty.

Do NOT:

- Confirm a diagnosis.
- Provide treatment.
- Order tests.
- Invent findings.
- Invent evidence.
- Invent citations.
- Treat missing information as negative findings.
- Convert model confidence into clinical probability.

Clinician review is mandatory.
"""


# ============================================================================
# Prompt Builder
# ============================================================================


class DiagnosisPromptBuilder:
    """
    Builds the prompt used by the Differential Diagnosis Agent.

    Responsibilities:
        - Maintain diagnosis prompt version.
        - Build system/human prompt.
        - Define expected input variables.
        - Format ClinicalState data.
        - Expose prompt metadata for observability.

    The class intentionally contains no clinical reasoning.
    """

    def __init__(
        self,
        *,
        version: str = DIAGNOSIS_PROMPT_VERSION,
    ) -> None:
        """
        Initialize DiagnosisPromptBuilder.

        Args:
            version:
                Prompt version.
        """

        self.version = version

        logger.debug(
            "DiagnosisPromptBuilder initialized | version={}",
            self.version,
        )

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """
        Return the system prompt.
        """

        logger.debug(
            "Diagnosis system prompt requested | version={}",
            self.version,
        )

        return DIAGNOSIS_SYSTEM_PROMPT.strip()

    # =========================================================================
    # Human Prompt
    # =========================================================================

    def get_human_prompt(self) -> str:
        """
        Return the human prompt template.
        """

        logger.debug(
            "Diagnosis human prompt requested | version={}",
            self.version,
        )

        return DIAGNOSIS_HUMAN_PROMPT.strip()

    # =========================================================================
    # Chat Prompt
    # =========================================================================

    def build(self) -> ChatPromptTemplate:
        """
        Build the LangChain ChatPromptTemplate.
        """

        logger.info(
            "Building Differential Diagnosis prompt | version={}",
            self.version,
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    self.get_system_prompt(),
                ),
                (
                    "human",
                    self.get_human_prompt(),
                ),
            ]
        )

        logger.info(
            "Differential Diagnosis prompt built successfully | "
            "version={}",
            self.version,
        )

        return prompt

    # =========================================================================
    # Input Variables
    # =========================================================================

    @staticmethod
    def get_input_variables() -> List[str]:
        """
        Return the input variables required by the prompt.
        """

        variables = [
            "clinical_context",
            "active_problems",
            "risk_factors",
            "abnormal_findings",
            "temporal_events",
            "unresolved_issues",
            "information_gaps",
            "evidence",
            "evidence_gaps",
        ]

        logger.debug(
            "Diagnosis prompt variables requested | count={}",
            len(variables),
        )

        return variables

    # =========================================================================
    # Prompt Values
    # =========================================================================

    def format_prompt_values(
        self,
        *,
        clinical_context: Dict[str, Any],
        active_problems: List[Dict[str, Any]],
        risk_factors: List[Dict[str, Any]],
        abnormal_findings: List[Dict[str, Any]],
        temporal_events: List[Dict[str, Any]],
        unresolved_issues: List[str],
        information_gaps: List[str],
        evidence: List[Dict[str, Any]],
        evidence_gaps: List[str],
    ) -> Dict[str, Any]:
        """
        Prepare ClinicalState values for the diagnosis prompt.

        This method performs no diagnosis or clinical transformation.
        It only maps state values to prompt variables.
        """

        logger.debug(
            "Formatting Diagnosis prompt values | "
            "active_problem_count={} | risk_factor_count={} | "
            "abnormal_finding_count={} | temporal_event_count={} | "
            "information_gap_count={} | evidence_count={}",
            len(active_problems),
            len(risk_factors),
            len(abnormal_findings),
            len(temporal_events),
            len(information_gaps),
            len(evidence),
        )

        return {
            "clinical_context": clinical_context,
            "active_problems": active_problems,
            "risk_factors": risk_factors,
            "abnormal_findings": abnormal_findings,
            "temporal_events": temporal_events,
            "unresolved_issues": unresolved_issues,
            "information_gaps": information_gaps,
            "evidence": evidence,
            "evidence_gaps": evidence_gaps,
        }

    # =========================================================================
    # Metadata
    # =========================================================================

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return prompt metadata for observability and evaluation.
        """

        metadata = {
            "prompt_name": "clinical_differential_diagnosis",
            "prompt_version": self.version,
            "agent": "DifferentialDiagnosisAgent",
            "purpose": "evidence_grounded_differential_generation",
            "clinician_review_required": True,
        }

        logger.debug(
            "Diagnosis prompt metadata requested | version={}",
            self.version,
        )

        return metadata


# ============================================================================
# Convenience Function
# ============================================================================


def create_diagnosis_prompt(
    *,
    version: str = DIAGNOSIS_PROMPT_VERSION,
) -> ChatPromptTemplate:
    """
    Create the Differential Diagnosis Agent prompt.

    Example:

        prompt = create_diagnosis_prompt()

        structured_llm = llm.with_structured_output(
            DifferentialResult
        )

        chain = prompt | structured_llm
    """

    logger.debug(
        "Creating Differential Diagnosis prompt | version={}",
        version,
    )

    builder = DiagnosisPromptBuilder(
        version=version,
    )

    return builder.build()
