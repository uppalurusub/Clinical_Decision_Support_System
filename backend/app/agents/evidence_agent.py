"""
evidence_agent.py

Clinical Evidence / RAG Agent.

Responsibilities:
    - Receive structured clinical context
    - Generate an evidence-focused search query
    - Retrieve relevant clinical evidence
    - Filter and organize retrieved evidence
    - Preserve source metadata and citations
    - Provide evidence to downstream diagnostic agents

Architecture:

    ContextAgent
         ↓
    EvidenceAgent
         ↓
    LangChain Retriever
         ↓
    Vector Database / Clinical Knowledge Base
         ↓
    Structured Evidence
         ↓
    Differential Diagnosis Agent

Important:
    This agent retrieves and organizes evidence.
    It does NOT make a diagnosis or treatment decision.

```

## How this agent fits your architecture

You now have:

```text
Raw Clinical Data
        │
        ▼
┌───────────────────┐
│   IntakeAgent     │
│ Extract / Normalize
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   ContextAgent    │
│ Build Clinical    │
│ Context           │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│   EvidenceAgent   │
│                   │
│ Query Generation  │
│       ↓           │
│ Retriever         │
│       ↓           │
│ Evidence Analysis │
└─────────┬─────────┘
          │
          ▼
┌────────────────────────┐
│ Differential Diagnosis │
│ Agent                  │
└────────────────────────┘
```

### LangGraph registration

```python
graph.add_node(
    "evidence",
    evidence_agent.node,
)

graph.add_edge(
    "context",
    "evidence",
)

graph.add_edge(
    "evidence",
    "differential",
)
```

So the graph becomes:

```text
START
  │
  ▼
Intake
  │
  ▼
Context
  │
  ▼
Evidence
  │
  ▼
Differential
  │
  ▼
Missing Information
  │
  ▼
Diagnostic Tests
  │
  ▼
Safety
  │
  ▼
Clinical Critic
  │
  ├── FAIL ──────► Differential
  │
  ▼
Summary
  │
  ▼
Human Review
  │
  ▼
END
```

## Important RAG design

I would **not** use a generic similarity search over every medical document. The Evidence Agent should ideally retrieve from a curated clinical knowledge base:

```text
                    Evidence Knowledge Base
                             │
        ┌────────────────────┼─────────────────────┐
        ▼                    ▼                     ▼
 Clinical Guidelines   Hospital Protocols   Medical Literature
        │                    │                     │
        └────────────────────┼─────────────────────┘
                             ▼
                       Vector Store
                             │
                             ▼
                         Retriever
                             │
                             ▼
                       EvidenceAgent
```

And document metadata should support filtering such as:

```python
{
    "document_type": "clinical_guideline",
    "specialty": "cardiology",
    "condition": "acute_coronary_syndrome",
    "organization": "...",
    "version": "2026",
    "effective_date": "...",
    "jurisdiction": "...",
}
```

This allows you to move from:

```text
Semantic similarity
```

toward:

```text
Metadata filtering
       +
Semantic retrieval
       +
Source ranking
       +
Evidence validation
```

which is much more appropriate for a clinical RAG system.

### One important distinction

The `EvidenceAgent` should **not** produce:

```text
"The patient has myocardial infarction."
```

It should produce evidence such as:

```text
Clinical context
       ↓
Relevant guideline
       ↓
Diagnostic criteria
       ↓
Supporting/contradicting evidence
       ↓
Source/citation
```

Then your **Differential Diagnosis Agent** can reason over:

```text
Clinical Context
       +
Retrieved Evidence
       +
Patient Findings
       +
Validated Clinical Rules
       ↓
Differential Diagnosis
```

This separation gives you a much stronger **Agentic RAG architecture** than asking one LLM to simultaneously retrieve evidence, diagnose the patient, and generate recommendations.

"""

from typing import Any, Dict, List, Optional
import time
from langchain_openai import ChatOpenAI
from app.config.settings import settings
logger = settings.get_logger()

llm = settings.get_model_client()

from pydantic import BaseModel, Field

from langchain_core.documents import Document

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever


# ============================================================================
# Structured Output Models
# ============================================================================


class EvidenceItem(BaseModel):
    """A single piece of retrieved clinical evidence."""

    evidence_id: Optional[str] = Field(
        default=None,
        description="Unique identifier for the evidence item."
    )

    title: Optional[str] = Field(
        default=None,
        description="Title of the source document."
    )

    source: Optional[str] = Field(
        default=None,
        description="Source organization or publication."
    )

    source_type: Optional[str] = Field(
        default=None,
        description=(
            "Type of source such as clinical guideline, "
            "protocol, review, or medical literature."
        ),
    )

    condition: Optional[str] = Field(
        default=None,
        description="Clinical condition relevant to the evidence."
    )

    statement: str = Field(
        description="Concise evidence statement."
    )

    relevance: str = Field(
        description="Why the evidence is relevant to the patient context."
    )

    citation: Optional[str] = Field(
        default=None,
        description="Citation or source reference."
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional source metadata."
    )


class EvidenceSearchResult(BaseModel):
    """Structured result generated by the Evidence Agent."""

    search_query: str = Field(
        default="",
        description="Clinical evidence search query."
    )

    evidence: List[EvidenceItem] = Field(
        default_factory=list,
        description="Relevant clinical evidence."
    )

    evidence_gaps: List[str] = Field(
        default_factory=list,
        description="Areas where sufficient evidence was not found."
    )

    source_count: int = Field(
        default=0,
        description="Number of relevant evidence sources."
    )


# ============================================================================
# Evidence Agent
# ============================================================================


class EvidenceAgent:
    """
    Clinical Evidence / RAG Agent.

    Uses a LangChain retriever to find relevant clinical
    evidence based on the structured patient context.

    The agent does NOT:

        - Make a diagnosis
        - Prescribe treatment
        - Determine patient eligibility
        - Order tests
        - Make autonomous clinical decisions
    """

    def __init__(
        self,
        llm: ChatOpenAI,
        retriever: BaseRetriever,
        top_k: int = 5,
    ) -> None:
        """
        Initialize EvidenceAgent.

        Args:
            llm:
                LangChain-compatible ChatModel.

            retriever:
                LangChain-compatible clinical evidence retriever.

            top_k:
                Maximum number of evidence documents to retrieve.
        """

        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

        self.query_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Clinical Evidence Retrieval Agent.

Your task is to create a precise evidence-search query
from the provided clinical context.

The query should emphasize:

- Chief complaint
- Active clinical problems
- Major symptoms
- Relevant risk factors
- Important abnormal findings
- Relevant medications
- Relevant clinical conditions
- Diagnostic uncertainty

Rules:

- Do not make a diagnosis.
- Do not recommend treatment.
- Do not invent information.
- Use only the supplied clinical context.
- Produce a concise clinical evidence-search query.
""",
                ),
                (
                    "human",
                    """
Clinical Context:

{clinical_context}

Active Problems:

{active_problems}

Risk Factors:

{risk_factors}

Abnormal Findings:

{abnormal_findings}

Unresolved Issues:

{unresolved_issues}
""",
                ),
            ]
        )

        self.query_chain = (
            self.query_prompt
            | self.llm
        )

        self.evidence_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
You are a Clinical Evidence Analysis Agent.

Review the retrieved clinical documents and identify
the evidence that is directly relevant to the supplied
clinical context.

Rules:

1. Use only information contained in the retrieved documents.
2. Do not invent evidence.
3. Do not fabricate citations.
4. Preserve source information.
5. Prefer current clinical guidelines and authoritative
   clinical sources when available.
6. Clearly distinguish evidence from interpretation.
7. Do not make a final diagnosis.
8. Do not provide autonomous treatment recommendations.
9. Identify evidence gaps when retrieved information is
   insufficient.
10. If the source does not support a statement, do not
    include the statement.

Return structured output only.
""",
                ),
                (
                    "human",
                    """
Clinical Context:

{clinical_context}

Retrieved Clinical Documents:

{documents}
""",
                ),
            ]
        )

        self.structured_evidence_llm = (
            self.llm.with_structured_output(
                EvidenceSearchResult
            )
        )

        self.evidence_chain = (
            self.evidence_prompt
            | self.structured_evidence_llm
        )

        logger.debug(
            "EvidenceAgent initialized | "
            "llm={} | retriever={} | top_k={}",
            llm.__class__.__name__,
            retriever.__class__.__name__,
            top_k,
        )

    # ------------------------------------------------------------------
    # Build Search Query
    # ------------------------------------------------------------------

    async def _build_search_query(
        self,
        state: Dict[str, Any],
    ) -> str:
        """
        Generate an evidence retrieval query.

        Args:
            state:
                Current LangGraph state.

        Returns:
            Search query string.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.debug(
            "Generating clinical evidence query | "
            "patient_id={}",
            patient_id,
        )

        response = await self.query_chain.ainvoke(
            {
                "clinical_context": state.get(
                    "clinical_context",
                    {},
                ),

                "active_problems": state.get(
                    "active_problems",
                    [],
                ),

                "risk_factors": state.get(
                    "risk_factors",
                    [],
                ),

                "abnormal_findings": state.get(
                    "abnormal_findings",
                    [],
                ),

                "unresolved_issues": state.get(
                    "unresolved_issues",
                    [],
                ),
            }
        )

        query = response.content.strip()

        logger.debug(
            "Clinical evidence query generated | "
            "patient_id={} | query_length={}",
            patient_id,
            len(query),
        )

        return query

    # ------------------------------------------------------------------
    # Retrieve Documents
    # ------------------------------------------------------------------

    async def _retrieve_documents(
        self,
        query: str,
        patient_id: str,
    ) -> List[Document]:
        """
        Retrieve relevant clinical documents.

        Args:
            query:
                Evidence retrieval query.

            patient_id:
                Patient identifier used only for logging.

        Returns:
            Retrieved LangChain documents.
        """

        logger.info(
            "Retrieving clinical evidence | "
            "patient_id={} | top_k={}",
            patient_id,
            self.top_k,
        )

        """ documents = await self.retriever.ainvoke(
            query
        ) """

        documents = await self.retriever.aretrieve( query=query )

        documents = documents[: self.top_k]

        logger.info(
            "Clinical evidence retrieval completed | "
            "patient_id={} | documents={}",
            patient_id,
            len(documents),
        )

        return documents

    # ------------------------------------------------------------------
    # Analyze Evidence
    # ------------------------------------------------------------------

    async def _analyze_evidence(
        self,
        state: Dict[str, Any],
        documents: List[Document],
        query: str,
    ) -> EvidenceSearchResult:
        """
        Analyze retrieved documents using structured LLM output.

        Args:
            state:
                Current LangGraph state.

            documents:
                Retrieved clinical documents.

            query:
                Evidence search query.

        Returns:
            EvidenceSearchResult.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        formatted_documents = []

        for index, document in enumerate(
            documents,
            start=1,
        ):

            formatted_documents.append(
                {
                    "document_number": index,
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )

        logger.debug(
            "Analyzing retrieved evidence | "
            "patient_id={} | documents={}",
            patient_id,
            len(formatted_documents),
        )

        result = await self.evidence_chain.ainvoke(
            {
                "clinical_context": state.get(
                    "clinical_context",
                    {},
                ),

                "documents": formatted_documents,
            }
        )

        # Ensure the generated result retains the actual query.
        result.search_query = query

        result.source_count = len(
            result.evidence
        )

        logger.info(
            "Clinical evidence analysis completed | "
            "patient_id={} | evidence_items={} | "
            "evidence_gaps={}",
            patient_id,
            len(result.evidence),
            len(result.evidence_gaps),
        )

        return result

    # ------------------------------------------------------------------
    # Run Agent
    # ------------------------------------------------------------------

    async def run(
        self,
        state: Dict[str, Any],
    ) -> EvidenceSearchResult:
        """
        Execute the Evidence Agent.

        Workflow:

            Clinical Context
                  ↓
            Generate Search Query
                  ↓
            Retrieve Documents
                  ↓
            Analyze Evidence
                  ↓
            Structured Evidence Result
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.info(
            "EvidenceAgent started | patient_id={}",
            patient_id,
        )

        try:

            # ----------------------------------------------------------
            # Step 1: Build query
            # ----------------------------------------------------------

            
            query = await self._timed_stage(
                "query_generation",
                self._build_search_query,
                state,
            )

            if not query:
                logger.warning(
                    "Evidence search query is empty | "
                    "patient_id={}",
                    patient_id,
                )

                return EvidenceSearchResult(
                    search_query="",
                    evidence=[],
                    evidence_gaps=[
                        "Unable to construct evidence search query."
                    ],
                    source_count=0,
                )

            # ----------------------------------------------------------
            # Step 2: Retrieve documents
            # ----------------------------------------------------------

            documents = await self._retrieve_documents(
                query=query,
                patient_id=patient_id,
            )

            documents = await self._timed_stage(
                "document_retrieval",
                self._retrieve_documents,
                query,
                patient_id,
            )

            if not documents:

                logger.warning(
                    "No clinical evidence documents retrieved | "
                    "patient_id={}",
                    patient_id,
                )

                return EvidenceSearchResult(
                    search_query=query,
                    evidence=[],
                    evidence_gaps=[
                        "No relevant clinical evidence retrieved."
                    ],
                    source_count=0,
                )

            # ----------------------------------------------------------
            # Step 3: Analyze evidence
            # ----------------------------------------------------------

            

            result = await self._timed_stage(
                "evidence_analysis",
                self._analyze_evidence,
                state,
                documents,
                query,
            )

            logger.info(
                "EvidenceAgent completed | "
                "patient_id={} | sources={}",
                patient_id,
                result.source_count,
            )

            return result

        except Exception as exc:

            logger.exception(
                "EvidenceAgent failed | "
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
            State updates containing clinical evidence.
        """

        patient_id = state.get(
            "patient_id",
            "unknown",
        )

        logger.debug(
            "EvidenceAgent LangGraph node executing | "
            "patient_id={}",
            patient_id,
        )

        result = await self.run(state)

        return {
            "evidence_query": result.search_query,

            "evidence": [
                evidence.model_dump()
                for evidence in result.evidence
            ],

            "evidence_gaps": (
                result.evidence_gaps
            ),

            "evidence_source_count": (
                result.source_count
            ),
        }

    async def _timed_stage(self, stage_name, operation, *args, **kwargs):
        started = time.perf_counter()

        logger.info(
            "Evidence stage started | stage={}",
            stage_name,
        )

        try:
            return await operation(*args, **kwargs)
        except Exception:
            logger.exception(
                "Evidence stage failed | stage={}",
                stage_name,
            )
            raise
        finally:
            logger.info(
                "Evidence stage finished | stage={} | "
                "duration_seconds={:.3f}",
                stage_name,
                time.perf_counter() - started,
            )
