"""
clinical_graph.py

LangGraph orchestration layer for the Agentic Clinical Decision
Support and Diagnosis workflow.

Workflow:

    START
      |
      v
    Context
      |
      v
    Evidence / RAG
      |
      v
    Differential Diagnosis
      |
      v
    
    Safety
   

Important:
    - This module orchestrates agents.
    - It does not perform clinical reasoning itself.
    - Agents operate on ClinicalState.
    - The Critic Agent can route execution back to an earlier stage.
    - Final output always requires clinician review.

```

### How this `clinical_graph.py` fits your architecture

Your orchestration is now:

```text
FastAPI Router
      │
      ▼
ClinicalDecisionService
      │
      ▼
ClinicalDecisionImplementation
      │
      ▼
ClinicalGraph
      │
      ▼
┌───────────────────────────────┐
│         LangGraph             │
│                               │
│ START                         │
│   ↓                           │
│ Context Agent                 │
│   ↓                           │
│ Evidence / RAG Agent          │
│   ↓                           │
│ Differential Agent            │
│   ↓                           │
│ Safety Agent                  │
│   ↓                           │
│ 
```

### The key distinction

`clinical_state.py` defines **what the workflow knows**.

`clinical_graph.py` defines **how the workflow moves**.

The individual agent files define **how each agent reasons/operates**.

So the separation is:

| Component                   | Responsibility                           |
| --------------------------- | ---------------------------------------- |
| `clinical_state.py`         | Shared LangGraph state                   |
| `clinical_graph.py`         | Workflow/orchestration                   |
| `intake_agent.py`           | Extract/normalize                        |
| `context_agent.py`          | Organize clinical context                |
| `evidence_agent.py`         | Retrieve/ground evidence                 |
| `differential_agent.py`     | Generate differential                    |
| `missing_info_agent.py`     | Find information gaps                    |
| `test_agent.py`             | Suggest diagnostic information gathering |
| `safety_agent.py`           | Safety analysis                          |
| `critic_agent.py`           | Challenge/validate                       |
| `summary_agent.py`          | Clinician-facing synthesis               |
| `clinical_decision_impl.py` | Application-to-graph adapter             |

One important production feature in this implementation is the **`max_iterations` guard**. Without it, a Critic that continually requests re-analysis could create an unbounded LangGraph loop. Here, the workflow can re-analyze, but eventually exits to the Summary stage with `requires_human_review=True`.

Also, `ClinicalGraph` should normally be **constructed once during application startup/dependency initialization**, not recreated for every API request. That lets the compiled graph and agent dependencies be reused efficiently.

"""

from __future__ import annotations

from typing import Any, Dict, Literal

from app.config.settings import settings
logger = settings.get_logger()



from langgraph.graph import END, START, StateGraph
import time
from app.agents.context_agent import ContextAgent
from app.agents.evidence_agent import EvidenceAgent
from app.agents.differential_agent import DifferentialAgent
from app.agents.safety_agent import SafetyAgent

from app.graph.clinical_state import (
    ClinicalState,
    increment_graph_iteration,
    mark_workflow_completed,
    mark_workflow_requires_review,
)


# ============================================================================
# Type Aliases
# ============================================================================

CriticRoute = Literal[
    "evidence",
    "differential",
    "missing_information",
    "safety",
    "summary",
]


# ============================================================================
# Clinical Graph
# ============================================================================

class ClinicalGraph:
    """
    Four-agent Clinical Decision Support POC.

    Flow:
        Context -> Evidence -> Differential -> Safety -> END

    Clinical decisions remain with a qualified clinician.
    """

    def __init__(
        self,
        *,
        context_agent: ContextAgent,
        evidence_agent: EvidenceAgent,
        differential_agent: DifferentialAgent,
        safety_agent: SafetyAgent,
    ) -> None:

        agents = {
            "context_agent": context_agent,
            "evidence_agent": evidence_agent,
            "differential_agent": differential_agent,
            "safety_agent": safety_agent,
        }

        missing_agents = [
            name for name, agent in agents.items()
            if agent is None
        ]

        if missing_agents:
            raise ValueError(
                "ClinicalGraph initialization failed. "
                f"Missing agents: {', '.join(missing_agents)}"
            )

        self.context_agent = context_agent
        self.evidence_agent = evidence_agent
        self.differential_agent = differential_agent
        self.safety_agent = safety_agent

        workflow = StateGraph(ClinicalState)

        workflow.add_node("context", self._context_node)
        workflow.add_node("evidence", self._evidence_node)
        workflow.add_node("differential", self._differential_node)
        workflow.add_node("safety", self._safety_node)

        workflow.add_edge(START, "context")
        workflow.add_edge("context", "evidence")
        workflow.add_edge("evidence", "differential")
        workflow.add_edge("differential", "safety")
        workflow.add_edge("safety", END)

        self.graph = workflow.compile()

        logger.info(
            "ClinicalGraph initialized | agent_count=4"
        )

    async def _context_node(
        self, state: ClinicalState
    ) -> Dict[str, Any]:
        logger.info("POC node started | node=context")
        #return await self.context_agent.node(state)
        return await self._run_timed_node(
            "context", self.context_agent, state
        )

    async def _evidence_node(
        self, state: ClinicalState
    ) -> Dict[str, Any]:
        logger.info("POC node started | node=evidence")
        #return await self.evidence_agent.node(state)
        return await self._run_timed_node(
            "evidence", self.evidence_agent, state
        )

    async def _differential_node(
        self, state: ClinicalState
    ) -> Dict[str, Any]:
        logger.info("POC node started | node=differential")
        #return await self.differential_agent.node(state)
        return await self._run_timed_node(
            "differential", self.differential_agent, state
        )

    async def _safety_node(
        self, state: ClinicalState
    ) -> Dict[str, Any]:
        logger.info("POC node started | node=safety")

        safety_updates = await self.safety_agent.node(state)

        # Build a response without an additional Summary LLM call.
        final_summary = {
            "clinical_context": state.get("clinical_context", {}),
            "evidence": state.get("evidence", []),
            "evidence_gaps": state.get("evidence_gaps", []),
            "differential_diagnoses": state.get(
                "differential_diagnosis", []
            ),
            "key_discriminating_findings": state.get(
                "key_discriminating_findings", []
            ),
            "diagnostic_uncertainties": state.get(
                "diagnostic_uncertainties", []
            ),
            "diagnostic_reasoning": state.get(
                "diagnostic_reasoning", ""
            ),
            "safety_flags": safety_updates.get("safety_flags", []),
            "safety_assessment": safety_updates.get(
                "safety_assessment", ""
            ),
            "immediate_attention_required": safety_updates.get(
                "immediate_attention_required", True
            ),
            "requires_human_review": True,
        }

        logger.info(
            "POC safety assessment completed | human_review_required=true"
        )

        return {
            **safety_updates,
            "final_summary": final_summary,
            "requires_human_review": True,
            "human_review_status": "pending",
            "human_review_reason": (
                "Clinical decision-support output requires "
                "review by a qualified clinician."
            ),
            "workflow_status": "requires_review",
            "completed_agents": [
                "context",
                "evidence",
                "differential",
                "safety",
            ],
        }

    async def ainvoke(
        self, state: ClinicalState,
        config=None
    ) -> ClinicalState:
        # Preserve clinician oversight for every invocation.
        initial_state = {
            **state,
            "requires_human_review": True,
            "human_review_status": "pending",
            "workflow_status": "running",
        }

        result = await self.graph.ainvoke(initial_state,config=config)

        # Fail closed if the workflow returns an incomplete result.
        if not result.get("final_summary"):
            result["workflow_status"] = "requires_review"
            result["requires_human_review"] = True
            result["human_review_status"] = "pending"
            result["human_review_reason"] = (
                "POC did not produce a complete final summary."
            )

        return result

    def get_graph(self):
        return self.graph

    def get_graph_structure(self) -> Dict[str, Any]:
        return {
            "nodes": [
                "context",
                "evidence",
                "differential",
                "safety",
            ],
            "edges": [
                ("context", "evidence"),
                ("evidence", "differential"),
                ("differential", "safety"),
            ],
            "human_review_required": True,
        }

    async def _run_timed_node(
        self,
        node_name: str,
        agent,
        state: ClinicalState,
    ) -> Dict[str, Any]:
        started = time.perf_counter()

        logger.info(
            "Clinical node started | node={}",
            node_name,
        )

        try:
            result = await agent.node(state)
            return result

        except Exception:
            logger.exception(
                "Clinical node failed | node={}",
                node_name,
            )
            raise

        finally:
            elapsed = time.perf_counter() - started

            logger.info(
                "Clinical node finished | node={} | "
                "execution_time_seconds={:.3f}",
                node_name,
                elapsed,
            )