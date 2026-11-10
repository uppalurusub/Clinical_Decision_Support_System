"""
intake_prompts.py

Prompt definitions for the Clinical Intake Agent.

Purpose:
    Convert raw clinical input into a structured, normalized intake
    representation.

The Intake Agent is responsible for:
    - Identifying the chief complaint.
    - Extracting symptoms.
    - Extracting relevant history.
    - Extracting medications.
    - Extracting allergies.
    - Extracting vitals.
    - Extracting relevant laboratory information.
    - Extracting relevant imaging information.
    - Identifying explicitly missing information.
    - Organizing the initial clinical context.

The Intake Agent must NOT:
    - Diagnose the patient.
    - Generate a differential diagnosis.
    - Recommend treatment.
    - Recommend tests.
    - Assign clinical probabilities.
    - Infer that unreported information is negative/normal.
    - Invent clinical facts.

Architecture:

    ClinicalDecisionRequest
             |
             v
        Intake Agent
             |
             v
       Intake Prompts
             |
             v
    Structured IntakeResult
             |
             v
        ClinicalState

```

### How `intake_prompts.py` connects to `intake_agent.py`

Your current `intake_agent.py` can now be simplified so that the agent owns the **reasoning/execution**, while the prompt module owns the **prompt**:

```text
intake_agent.py
      │
      ├── IntakeResult        ← structured output
      │
      ├── IntakePromptBuilder
      │        │
      │        └── ChatPromptTemplate
      │
      └── LLM
               │
               ▼
         Structured Output
```

For example:

```python
from app.prompts.intake_prompts import (
    IntakePromptBuilder,
)

prompt_builder = IntakePromptBuilder()

prompt = prompt_builder.build()

structured_llm = llm.with_structured_output(
    IntakeResult
)

chain = prompt | structured_llm
```

Then:

```python
result = await chain.ainvoke(
    prompt_builder.format_prompt_values(
        patient_id=state["patient_id"],
        symptoms=state.get("symptoms", {}),
        history=state.get("history", {}),
        vitals=state.get("vitals", {}),
        medications=state.get("medications", []),
        allergies=state.get("allergies", []),
        labs=state.get("labs", []),
        imaging=state.get("imaging", []),
    )
)
```

### Why this separation is valuable

You now have:

```text
prompts/
    intake_prompts.py
    context_prompts.py
    evidence_prompts.py
    differential_prompts.py
    missing_info_prompts.py
    test_prompts.py
    safety_prompts.py
    critic_prompts.py
    summary_prompts.py
```

while:

```text
agents/
    intake_agent.py
    context_agent.py
    evidence_agent.py
    differential_agent.py
    missing_info_agent.py
    test_agent.py
    safety_agent.py
    critic_agent.py
    summary_agent.py
```

This gives you a clean **Prompt Engineering → Agent Engineering → LangGraph Orchestration** separation.

The `INTAKE_PROMPT_VERSION` is also important for your healthcare AI observability: when you evaluate a workflow later, you can associate an agent run with the exact prompt version rather than simply knowing that `IntakeAgent` was executed.

"""

from __future__ import annotations

from typing import Any, Dict

from app.config.settings import settings
logger = settings.get_logger()

from langchain_core.prompts import ChatPromptTemplate


# ============================================================================
# Prompt Version
# ============================================================================

INTAKE_PROMPT_VERSION = "1.0.0"


# ============================================================================
# System Prompt
# ============================================================================

INTAKE_SYSTEM_PROMPT = """
You are the Clinical Intake Agent in a clinician-facing
Clinical Decision Support system.

Your responsibility is to perform structured clinical information
extraction and normalization.

You are NOT the diagnosing agent.

Your task is to organize the information explicitly provided in
the patient input into a structured clinical intake representation.

CORE RESPONSIBILITIES

1. Identify the chief complaint or presenting concern.

2. Extract explicitly reported symptoms.

3. For each symptom, capture available information such as:
   - symptom name
   - duration
   - severity
   - onset
   - characteristics

4. Extract relevant medical history.

5. Extract current medications.

6. Extract known allergies.

7. Extract provided vital signs.

8. Extract relevant laboratory findings.

9. Extract relevant imaging findings.

10. Identify information that appears to be missing and could be
    relevant to understanding the presenting complaint.

11. Create a concise initial clinical context based ONLY on the
    information supplied.

CRITICAL SAFETY RULES

- Do NOT diagnose the patient.

- Do NOT generate a differential diagnosis.

- Do NOT recommend treatment.

- Do NOT recommend diagnostic tests.

- Do NOT assign medical probabilities.

- Do NOT invent clinical facts.

- Do NOT infer that a finding is normal simply because it was
  not reported.

- Do NOT infer that a symptom is absent because it was not mentioned.

- Do NOT convert missing information into negative findings.

- Preserve uncertainty when information is incomplete.

- Distinguish explicitly reported information from inferred context.

- If information is ambiguous, preserve the ambiguity rather than
  creating a definitive interpretation.

- Do not introduce information from medical knowledge that is not
  present in the supplied patient data.

MISSING INFORMATION

When identifying missing information:

- Only identify information that is genuinely absent from the
  supplied input.

- Do not fabricate missing values.

- Do not claim that a clinical finding is absent unless the input
  explicitly states that it is absent.

- Missing information should be expressed as an information gap,
  not as a clinical conclusion.

CLINICAL CONTEXT

The initial clinical context should summarize the information
provided by the patient/EHR.

It should answer:

"What clinical information has been provided so far?"

It should NOT answer:

"What is the patient's diagnosis?"

The output will be passed to downstream agents such as:

- Clinical Context Agent
- Evidence/RAG Agent
- Differential Diagnosis Agent
- Missing Information Agent
- Safety Agent

Therefore, accuracy and faithful extraction are more important
than generating sophisticated clinical reasoning.

OUTPUT

Return ONLY the requested structured output.

Do not include conversational commentary.

Do not include explanations outside the structured response.

Always preserve the requirement for downstream clinician review.
"""


# ============================================================================
# Human Prompt Template
# ============================================================================

INTAKE_HUMAN_PROMPT = """
Extract and normalize the following clinical information.

PATIENT IDENTIFIER
{patient_id}

SYMPTOMS / PRESENTING COMPLAINT
{symptoms}

MEDICAL HISTORY
{history}

VITAL SIGNS
{vitals}

MEDICATIONS
{medications}

ALLERGIES
{allergies}

LABORATORY RESULTS
{labs}

IMAGING
{imaging}

Perform clinical intake extraction according to the system
instructions.

Remember:

- Extract only information explicitly provided.
- Do not diagnose.
- Do not recommend treatment.
- Do not recommend tests.
- Do not invent missing values.
- Do not treat unreported information as negative.
- Preserve uncertainty.
- Identify relevant information gaps.
"""


# ============================================================================
# Prompt Builder
# ============================================================================


class IntakePromptBuilder:
    """
    Builds the ChatPromptTemplate used by IntakeAgent.

    Keeping prompt construction separate from the agent provides:

        - prompt versioning
        - unit testing
        - easier prompt maintenance
        - cleaner agent implementation
        - centralized prompt governance
    """

    def __init__(
        self,
        *,
        version: str = INTAKE_PROMPT_VERSION,
    ) -> None:
        """
        Initialize IntakePromptBuilder.

        Args:
            version:
                Prompt version used for observability and evaluation.
        """

        self.version = version

        logger.debug(
            "IntakePromptBuilder initialized | version={}",
            self.version,
        )

    # =========================================================================
    # System Prompt
    # =========================================================================

    def get_system_prompt(self) -> str:
        """
        Return the Intake Agent system prompt.
        """

        logger.debug(
            "Intake system prompt requested | version={}",
            self.version,
        )

        return INTAKE_SYSTEM_PROMPT.strip()

    # =========================================================================
    # Human Prompt
    # =========================================================================

    def get_human_prompt(self) -> str:
        """
        Return the Intake Agent human/user prompt template.
        """

        logger.debug(
            "Intake human prompt requested | version={}",
            self.version,
        )

        return INTAKE_HUMAN_PROMPT.strip()

    # =========================================================================
    # Chat Prompt
    # =========================================================================

    def build(self) -> ChatPromptTemplate:
        """
        Build the LangChain ChatPromptTemplate.
        """

        logger.info(
            "Building Intake Agent prompt | version={}",
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
            "Intake Agent prompt built successfully | version={}",
            self.version,
        )

        return prompt

    # =========================================================================
    # Prompt Variables
    # =========================================================================

    @staticmethod
    def get_input_variables() -> list[str]:
        """
        Return the variables expected by the human prompt.
        """

        variables = [
            "patient_id",
            "symptoms",
            "history",
            "vitals",
            "medications",
            "allergies",
            "labs",
            "imaging",
        ]

        logger.debug(
            "Intake prompt input variables requested | count={}",
            len(variables),
        )

        return variables

    # =========================================================================
    # Prompt Formatting
    # =========================================================================

    def format_prompt_values(
        self,
        *,
        patient_id: str,
        symptoms: Dict[str, Any],
        history: Dict[str, Any],
        vitals: Dict[str, Any],
        medications: list[Dict[str, Any]],
        allergies: list[str],
        labs: list[Dict[str, Any]],
        imaging: list[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Prepare input values for the Intake prompt.

        This method does not perform clinical transformation.

        It only creates the dictionary expected by the
        ChatPromptTemplate.
        """

        logger.debug(
            "Formatting Intake prompt values | "
            "has_symptoms={} | has_history={} | "
            "has_vitals={} | medication_count={} | "
            "allergy_count={} | lab_count={} | imaging_count={}",
            bool(symptoms),
            bool(history),
            bool(vitals),
            len(medications),
            len(allergies),
            len(labs),
            len(imaging),
        )

        return {
            "patient_id": patient_id,
            "symptoms": symptoms,
            "history": history,
            "vitals": vitals,
            "medications": medications,
            "allergies": allergies,
            "labs": labs,
            "imaging": imaging,
        }

    # =========================================================================
    # Prompt Metadata
    # =========================================================================

    def get_metadata(self) -> Dict[str, Any]:
        """
        Return prompt metadata useful for observability,
        evaluation and prompt version tracking.
        """

        metadata = {
            "prompt_name": "clinical_intake",
            "prompt_version": self.version,
            "agent": "IntakeAgent",
            "purpose": "clinical_information_extraction",
        }

        logger.debug(
            "Intake prompt metadata requested | version={}",
            self.version,
        )

        return metadata


# ============================================================================
# Convenience Function
# ============================================================================


def create_intake_prompt(
    *,
    version: str = INTAKE_PROMPT_VERSION,
) -> ChatPromptTemplate:
    """
    Convenience function for creating the Intake Agent prompt.

    Example:

        prompt = create_intake_prompt()

        chain = prompt | structured_llm
    """

    logger.debug(
        "Creating Intake Agent prompt | version={}",
        version,
    )

    builder = IntakePromptBuilder(
        version=version,
    )

    return builder.build()
