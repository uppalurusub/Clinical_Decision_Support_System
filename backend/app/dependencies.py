"""
app/dependencies.py

Application dependency injection and component construction.

Architecture:

    FastAPI Router
          |
          v
    ClinicalDecisionService
          |
          v
    ClinicalDecisionImplementation
          |
          v
    ClinicalGraph
          |
          +-----------------------------+
          |                             |
          v                             v
       Agents                       RAG Layer
          |                             |
          v                             v
        LLM                       ClinicalRetriever
                                        |
                                        v
                                  Vector Store
                                        |
                                        v
                                   Embeddings

Responsibilities:
    - Create long-lived application components
    - Create the LLM once
    - Create embeddings/vector store/retriever once
    - Create agents once
    - Create ClinicalGraph once
    - Create Service/Implementation once
    - Provide FastAPI dependency functions
    - Avoid rebuilding expensive objects per request
    - Use Loguru for safe application logging

Security:
    - Never log PHI
    - Never log prompts
    - Never log LLM responses
    - Never log embeddings
    - Never log API keys/secrets
    - Only log component names and lifecycle status
"""

from __future__ import annotations

from typing import Optional



from app.services.clinical_decision_service import (
    ClinicalDecisionService,
)
from app.implementations.clinical_decision_impl import (
    ClinicalDecisionImplementation,
)

from app.graph.clinical_graph import ClinicalGraph

from app.agents.context_agent import ContextAgent
from app.agents.evidence_agent import EvidenceAgent
from app.agents.differential_agent import DifferentialAgent
from app.agents.safety_agent import SafetyAgent

from app.rag.embeddings import EmbeddingManager
from app.rag.vector_store import VectorStoreManager
from app.rag.retriever import ClinicalRetriever

from app.config.settings import settings
logger = settings.get_logger()

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class ApplicationDependencies:
    """
    Application-level dependency container.

    All expensive/shared objects are created once during application
    startup and reused across requests.

    This object should NOT be created inside a request handler.
    """

    def __init__(self) -> None:
        self._initialized: bool = False

        # ---------------------------------------------------------
        # Infrastructure
        # ---------------------------------------------------------

        self.llm = None
        self.embedding_manager = None
        self.vector_store_manager = None
        self.clinical_retriever = None

        # ---------------------------------------------------------
        # Agents
        # ---------------------------------------------------------

        self.context_agent = None
        self.evidence_agent = None
        self.differential_agent = None
        self.safety_agent = None
        
        # ---------------------------------------------------------
        # Application layers
        # ---------------------------------------------------------

        self.clinical_graph = None
        self.clinical_implementation = None
        self.clinical_service = None

    # =================================================================
    # INITIALIZATION
    # =================================================================

    async def initialize(self) -> None:
        """
        Initialize all long-lived application dependencies.

        This method should be called exactly once during FastAPI
        application startup.
        """

        if self._initialized:
            logger.warning(
                "Application dependencies already initialized"
            )
            return

        logger.info(
            "Initializing application dependencies"
        )

        try:
            # ---------------------------------------------------------
            # 1. LLM
            # ---------------------------------------------------------

            self.llm = self._create_llm()

            logger.info(
                "LLM initialized"
            )

            # ---------------------------------------------------------
            # 2. Embeddings
            # ---------------------------------------------------------

            self.embedding_manager = self._create_embeddings()

            logger.info(
                "Embedding manager initialized"
            )

            # ---------------------------------------------------------
            # 3. Vector Store
            # ---------------------------------------------------------

            self.vector_store_manager = (
                self._create_vector_store(
                    self.embedding_manager
                )
            )

            logger.info(
                "Vector store manager initialized"
            )

            # ---------------------------------------------------------
            # 4. Retriever
            # ---------------------------------------------------------

            """ self.clinical_retriever = ClinicalRetriever(
                vector_store=self.vector_store_manager
            ) """

            self.clinical_retriever = self._create_clinical_retriever()

            logger.info(
                "Clinical retriever initialized"
            )

            # ---------------------------------------------------------
            # 5. Agents
            # ---------------------------------------------------------

            

            self.context_agent = ContextAgent(
                llm=self.llm
            )

            self.evidence_agent = EvidenceAgent(
                llm=self.llm,
                retriever=self.clinical_retriever,
            )

            self.differential_agent = DifferentialAgent(
                llm=self.llm
            )

            self.safety_agent = SafetyAgent(
                llm=self.llm
            )
            

            

            logger.info(
                "Clinical agents initialized | "
                "agent_count=4"
            )

            # ---------------------------------------------------------
            # 6. LangGraph
            # ---------------------------------------------------------

            self.clinical_graph = ClinicalGraph(
                context_agent=self.context_agent,
                evidence_agent=self.evidence_agent,
                differential_agent=self.differential_agent,
                safety_agent=self.safety_agent,
                
            )

            logger.info(
                "ClinicalGraph initialized"
            )

            # ---------------------------------------------------------
            # 7. Implementation
            # ---------------------------------------------------------

            self.clinical_implementation = (
                ClinicalDecisionImplementation(
                    clinical_graph=self.clinical_graph
                )
            )

            logger.info(
                "ClinicalDecisionImplementation initialized"
            )

            # ---------------------------------------------------------
            # 8. Service
            # ---------------------------------------------------------

            self.clinical_service = (
                ClinicalDecisionService(
                    implementation=self.clinical_implementation
                )
            )

            logger.info(
                "ClinicalDecisionService initialized"
            )

            # ---------------------------------------------------------
            # IMPORTANT:
            # Mark initialized only after every component succeeds.
            # ---------------------------------------------------------

            self._initialized = True

            logger.info(
                "Application dependencies initialized successfully"
            )

        except Exception:
            self._initialized = False

            logger.exception(
                "Application dependency initialization failed"
            )

            # Clear partially-created components.
            await self.shutdown()

            raise

    # =================================================================
    # SHUTDOWN
    # =================================================================

    async def shutdown(self) -> None:
        """
        Release application resources.

        Called during FastAPI application shutdown.
        """

        if not self._initialized:
            return

        logger.info(
            "Shutting down application dependencies"
        )

        try:
            # ---------------------------------------------------------
            # Application layers
            # ---------------------------------------------------------

            self.clinical_service = None
            self.clinical_implementation = None
            self.clinical_graph = None

            # ---------------------------------------------------------
            # Agents
            # ---------------------------------------------------------

            self.summary_agent = None
            self.critic_agent = None
            self.safety_agent = None
            self.test_agent = None
            self.missing_information_agent = None
            self.differential_agent = None
            self.evidence_agent = None
            self.context_agent = None
            self.intake_agent = None

            # ---------------------------------------------------------
            # RAG
            # ---------------------------------------------------------

            self.clinical_retriever = None
            self.vector_store_manager = None
            self.embedding_manager = None

            # ---------------------------------------------------------
            # LLM
            # ---------------------------------------------------------

            self.llm = None

            self._initialized = False

            logger.info(
                "Application dependencies shut down successfully"
            )

        except Exception:
            logger.exception(
                "Application dependency shutdown failed"
            )

            raise

    # =================================================================
    # FASTAPI DEPENDENCY ACCESSOR
    # =================================================================

    def get_clinical_service(
        self,
    ) -> ClinicalDecisionService:
        """
        Return the initialized ClinicalDecisionService.

        This method does NOT create the service.

        It only returns the already-created singleton-like
        application service.
        """

        if not self._initialized:
            raise RuntimeError(
                "Application dependencies have not been initialized."
            )

        if self.clinical_service is None:
            raise RuntimeError(
                "Clinical Decision Service is not available."
            )

        return self.clinical_service

    # =================================================================
    # COMPONENT FACTORIES
    # =================================================================

    def _create_llm(self):
        """
        Create the shared LLM.

        The actual provider/model configuration is delegated to
        app.config.llm_config.get_llm().
        """

        llm = settings.get_model_client()

        if llm is None:
            raise RuntimeError(
                "LLM factory returned None."
            )

        return llm

    def _create_embeddings(self):
        """
        Create the application-wide Hugging Face embedding provider.

        Uses a local Sentence Transformers model.
        No OpenAI API key is required.
        """

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": "cpu",
            },
            encode_kwargs={
                "normalize_embeddings": True,
            },
        )

        embedding_manager = EmbeddingManager(
            embeddings=embeddings
        )

        return embedding_manager

    
    def _create_clinical_retriever(self):
        retriever = self.vector_store_manager.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
            },
        )

        return ClinicalRetriever(
            retriever=retriever,
            top_k=5,
            min_content_length=20,
        )

    def _create_vector_store(self, embedding_manager):
        """
        Create the application-wide Chroma vector store.
        """

        vector_store = Chroma(
            collection_name="clinical_knowledge",
            persist_directory="./data/chroma",
            embedding_function=embedding_manager.embeddings,
        )

        vector_store_manager = VectorStoreManager(
            vector_store=vector_store
        )

        return vector_store_manager


# =====================================================================
# GLOBAL APPLICATION DEPENDENCY CONTAINER
# =====================================================================

_dependencies = ApplicationDependencies()


# =====================================================================
# APPLICATION LIFECYCLE FUNCTIONS
# =====================================================================

async def initialize_dependencies() -> None:
    """
    Initialize global application dependencies.

    Called from FastAPI lifespan startup.
    """

    await _dependencies.initialize()


async def shutdown_dependencies() -> None:
    """
    Shutdown global application dependencies.

    Called from FastAPI lifespan shutdown.
    """

    await _dependencies.shutdown()


# =====================================================================
# FASTAPI DEPENDENCY
# =====================================================================

def get_clinical_decision_service() -> ClinicalDecisionService:
    """
    FastAPI dependency provider.

    IMPORTANT:
        This function does NOT instantiate a new service.

    It returns the already initialized service from the
    application dependency container.

    Usage:

        service: ClinicalDecisionService = Depends(
            get_clinical_decision_service
        )
    """

    return _dependencies.get_clinical_service()


# =====================================================================
# OPTIONAL HEALTH CHECK
# =====================================================================

def dependencies_initialized() -> bool:
    """
    Return whether application dependencies are initialized.
    """

    return _dependencies._initialized


def get_dependency_health() -> dict:
    """
    Return safe dependency health information.

    No secrets, PHI, prompts, responses, or embeddings are returned.
    """

    return {
        "initialized": _dependencies._initialized,
        "llm": _dependencies.llm is not None,
        "embedding_manager": (
            _dependencies.embedding_manager is not None
        ),
        "vector_store_manager": (
            _dependencies.vector_store_manager is not None
        ),
        "clinical_retriever": (
            _dependencies.clinical_retriever is not None
        ),
        "clinical_graph": (
            _dependencies.clinical_graph is not None
        ),
        "clinical_service": (
            _dependencies.clinical_service is not None
        ),
    }