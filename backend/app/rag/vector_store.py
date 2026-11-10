"""
vector_store.py

Vector store abstraction for the Agentic Clinical Decision Support
and Diagnosis system.

Architecture:

    Clinical Knowledge Documents
              |
              v
        Document Loader
              |
              v
          Chunker
              |
              v
        Embedding Model
              |
              v
        VectorStoreManager
              |
              v
       Vector Database
              |
              v
      ClinicalRetriever
              |
              v
        EvidenceAgent

Responsibilities:
    - Manage vector-store lifecycle.
    - Add clinical documents.
    - Delete documents.
    - Perform similarity search.
    - Support metadata filtering.
    - Expose the underlying LangChain vector store.
    - Provide health/diagnostic information.
    - Log operational information using Loguru.

Non-responsibilities:
    - Clinical reasoning.
    - Diagnosis.
    - Treatment recommendations.
    - Differential diagnosis.
    - Patient-specific decision making.

Important:
    Do not log document content, embeddings, clinical notes,
    laboratory values, medications, or other PHI.

```

### Where it fits

Your RAG architecture is now:

```text
                  Clinical Guidelines
                  Medical Literature
                  Protocols
                  Institutional Knowledge
                         │
                         ▼
                 Document Loader
                         │
                         ▼
                    Chunker
                         │
                         ▼
                  Embedding Model
                         │
                         ▼
              ┌─────────────────────┐
              │ VectorStoreManager  │
              │                     │
              │ add_documents()     │
              │ similarity_search() │
              │ MMR search          │
              │ metadata filters    │
              └──────────┬──────────┘
                         │
                         ▼
                   Vector Database
                         │
                         ▼
                ClinicalRetriever
                         │
                         ▼
                  EvidenceAgent
                         │
                         ▼
                  ClinicalState
```

### Important separation

The responsibilities of the three files are now clean:

| File                | Responsibility                       |
| ------------------- | ------------------------------------ |
| `vector_store.py`   | Store/index/search vectors           |
| `retriever.py`      | Application-level evidence retrieval |
| `evidence_agent.py` | LLM-based evidence interpretation    |

For example:

```text
VectorStoreManager
        ↓
"Find the most relevant chunks."

ClinicalRetriever
        ↓
"Retrieve and filter the relevant clinical documents."

EvidenceAgent
        ↓
"Analyze those documents and produce grounded evidence."
```

The **similarity score returned by a vector store must not be treated as clinical confidence or probability**. It is a retrieval-related score whose meaning depends on the underlying vector database and distance metric.

For your production healthcare architecture, I would also recommend keeping the knowledge-base metadata roughly like:

```python
{
    "document_id": "...",
    "document_type": "clinical_guideline",
    "title": "...",
    "source": "...",
    "source_type": "guideline",
    "specialty": "cardiology",
    "condition": "heart_failure",
    "organization": "...",
    "version": "...",
    "effective_date": "...",
    "jurisdiction": "..."
}
```

That gives your **Evidence Agent + Critic Agent** a much stronger foundation for traceability and evidence grounding.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.config.settings import settings
logger = settings.get_logger()


from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


# ============================================================================
# Vector Store Manager
# ============================================================================


class VectorStoreManager:
    """
    Application-level wrapper around a LangChain VectorStore.

    This abstraction keeps the rest of the application independent
    of the actual vector database implementation.

    Supported underlying implementations can include:

        - Chroma
        - PGVector
        - Pinecone
        - FAISS
        - Azure AI Search
        - Elasticsearch
        - Other LangChain VectorStore implementations
    """

    def __init__(
        self,
        vector_store: VectorStore,
        *,
        collection_name: str = "clinical_knowledge",
        default_top_k: int = 5,
    ) -> None:
        """
        Initialize VectorStoreManager.

        Args:
            vector_store:
                Configured LangChain VectorStore.

            collection_name:
                Logical name of the clinical knowledge collection/index.

            default_top_k:
                Default number of documents returned by similarity search.

        Raises:
            ValueError:
                If default_top_k is invalid.
        """

        if default_top_k < 1:
            raise ValueError(
                "default_top_k must be greater than or equal to 1."
            )

        if not collection_name.strip():
            raise ValueError(
                "collection_name cannot be empty."
            )

        self.vector_store = vector_store
        self.collection_name = collection_name
        self.default_top_k = default_top_k

        logger.debug(
            "VectorStoreManager initialized | "
            "vector_store={} | collection={} | default_top_k={}",
            vector_store.__class__.__name__,
            collection_name,
            default_top_k,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def store(self) -> VectorStore:
        """
        Return the underlying LangChain vector store.

        Useful when another component needs direct access to
        LangChain-specific functionality.
        """

        return self.vector_store

    # =========================================================================
    # Document Ingestion
    # =========================================================================

    def add_documents(
        self,
        documents: Sequence[Document],
    ) -> List[str]:
        """
        Add documents to the vector store.

        Embedding generation is normally performed internally by
        the configured vector store using its embedding model.

        Args:
            documents:
                Documents/chunks to index.

        Returns:
            List of generated document IDs.
        """

        if not documents:
            logger.warning(
                "Vector document ingestion skipped | reason=no_documents"
            )
            return []

        logger.info(
            "Clinical knowledge ingestion started | "
            "collection={} | document_count={}",
            self.collection_name,
            len(documents),
        )

        try:
            document_ids = self.vector_store.add_documents(
                list(documents)
            )

            logger.info(
                "Clinical knowledge ingestion completed | "
                "collection={} | document_count={} | indexed_count={}",
                self.collection_name,
                len(documents),
                len(document_ids),
            )

            return list(document_ids)

        except Exception as exc:
            logger.exception(
                "Clinical knowledge ingestion failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    async def aadd_documents(
        self,
        documents: Sequence[Document],
    ) -> List[str]:
        """
        Add documents asynchronously.

        Prefer this method for asynchronous ingestion pipelines.
        """

        if not documents:
            logger.warning(
                "Async vector document ingestion skipped | "
                "reason=no_documents"
            )
            return []

        logger.info(
            "Async clinical knowledge ingestion started | "
            "collection={} | document_count={}",
            self.collection_name,
            len(documents),
        )

        try:
            document_ids = await self.vector_store.aadd_documents(
                list(documents)
            )

            logger.info(
                "Async clinical knowledge ingestion completed | "
                "collection={} | document_count={} | indexed_count={}",
                self.collection_name,
                len(documents),
                len(document_ids),
            )

            return list(document_ids)

        except Exception as exc:
            logger.exception(
                "Async clinical knowledge ingestion failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Document Deletion
    # =========================================================================

    def delete(
        self,
        ids: Sequence[str],
    ) -> bool:
        """
        Delete documents by vector-store document IDs.

        Args:
            ids:
                IDs assigned by the vector store.

        Returns:
            True if the operation completes successfully.
        """

        if not ids:
            logger.warning(
                "Vector document deletion skipped | reason=no_ids"
            )
            return True

        logger.info(
            "Vector document deletion started | "
            "collection={} | document_count={}",
            self.collection_name,
            len(ids),
        )

        try:
            result = self.vector_store.delete(
                list(ids)
            )

            logger.info(
                "Vector document deletion completed | "
                "collection={} | document_count={}",
                self.collection_name,
                len(ids),
            )

            return bool(result) if result is not None else True

        except Exception as exc:
            logger.exception(
                "Vector document deletion failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Similarity Search
    # =========================================================================

    def similarity_search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform synchronous similarity search.

        Args:
            query:
                Natural-language retrieval query.

            k:
                Number of documents to return.

            filter:
                Optional vector-store metadata filter.

        Returns:
            Relevant LangChain Documents.
        """

        if not query or not query.strip():
            logger.warning(
                "Vector similarity search skipped | reason=empty_query"
            )
            return []

        search_k = k or self.default_top_k

        if search_k < 1:
            raise ValueError(
                "k must be greater than or equal to 1."
            )

        logger.info(
            "Vector similarity search started | "
            "collection={} | k={} | has_filter={}",
            self.collection_name,
            search_k,
            bool(filter),
        )

        try:
            documents = self.vector_store.similarity_search(
                query=query.strip(),
                k=search_k,
                filter=filter,
            )

            logger.info(
                "Vector similarity search completed | "
                "collection={} | result_count={}",
                self.collection_name,
                len(documents),
            )

            return documents

        except Exception as exc:
            logger.exception(
                "Vector similarity search failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Async Similarity Search
    # =========================================================================

    async def asimilarity_search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform asynchronous similarity search.

        This is the preferred method for EvidenceAgent in the
        FastAPI async request path.
        """

        if not query or not query.strip():
            logger.warning(
                "Async vector similarity search skipped | "
                "reason=empty_query"
            )
            return []

        search_k = k or self.default_top_k

        if search_k < 1:
            raise ValueError(
                "k must be greater than or equal to 1."
            )

        logger.info(
            "Async vector similarity search started | "
            "collection={} | k={} | has_filter={}",
            self.collection_name,
            search_k,
            bool(filter),
        )

        try:
            documents = await self.vector_store.asimilarity_search(
                query=query.strip(),
                k=search_k,
                filter=filter,
            )

            logger.info(
                "Async vector similarity search completed | "
                "collection={} | result_count={}",
                self.collection_name,
                len(documents),
            )

            return documents

        except Exception as exc:
            logger.exception(
                "Async vector similarity search failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Similarity Search With Score
    # =========================================================================

    def similarity_search_with_score(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Perform similarity search and return relevance scores.

        Note:
            Score semantics depend on the underlying vector store.
            A score should not automatically be interpreted as a
            clinical probability or diagnostic confidence.
        """

        if not query or not query.strip():
            logger.warning(
                "Similarity search with score skipped | "
                "reason=empty_query"
            )
            return []

        search_k = k or self.default_top_k

        logger.info(
            "Vector similarity search with score started | "
            "collection={} | k={} | has_filter={}",
            self.collection_name,
            search_k,
            bool(filter),
        )

        try:
            results = (
                self.vector_store.similarity_search_with_score(
                    query=query.strip(),
                    k=search_k,
                    filter=filter,
                )
            )

            logger.info(
                "Vector similarity search with score completed | "
                "collection={} | result_count={}",
                self.collection_name,
                len(results),
            )

            return results

        except Exception as exc:
            logger.exception(
                "Vector similarity search with score failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    async def asimilarity_search_with_score(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        Perform asynchronous similarity search with scores.
        """

        if not query or not query.strip():
            logger.warning(
                "Async similarity search with score skipped | "
                "reason=empty_query"
            )
            return []

        search_k = k or self.default_top_k

        logger.info(
            "Async vector similarity search with score started | "
            "collection={} | k={} | has_filter={}",
            self.collection_name,
            search_k,
            bool(filter),
        )

        try:
            results = (
                await self.vector_store.asimilarity_search_with_score(
                    query=query.strip(),
                    k=search_k,
                    filter=filter,
                )
            )

            logger.info(
                "Async vector similarity search with score completed | "
                "collection={} | result_count={}",
                self.collection_name,
                len(results),
            )

            return results

        except Exception as exc:
            logger.exception(
                "Async similarity search with score failed | "
                "collection={} | error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Maximal Marginal Relevance Search
    # =========================================================================

    def max_marginal_relevance_search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Perform Maximal Marginal Relevance (MMR) search.

        MMR can help reduce highly redundant retrieved chunks.

        Args:
            query:
                Retrieval query.

            k:
                Final number of documents.

            fetch_k:
                Number of candidates considered before MMR selection.

            lambda_mult:
                Relevance/diversity trade-off.

            filter:
                Optional metadata filter.
        """

        if not query or not query.strip():
            logger.warning(
                "MMR search skipped | reason=empty_query"
            )
            return []

        search_k = k or self.default_top_k

        if fetch_k < search_k:
            raise ValueError(
                "fetch_k must be greater than or equal to k."
            )

        if not 0.0 <= lambda_mult <= 1.0:
            raise ValueError(
                "lambda_mult must be between 0.0 and 1.0."
            )

        logger.info(
            "Vector MMR search started | collection={} | "
            "k={} | fetch_k={} | lambda_mult={} | has_filter={}",
            self.collection_name,
            search_k,
            fetch_k,
            lambda_mult,
            bool(filter),
        )

        try:
            documents = (
                self.vector_store.max_marginal_relevance_search(
                    query=query.strip(),
                    k=search_k,
                    fetch_k=fetch_k,
                    lambda_mult=lambda_mult,
                    filter=filter,
                )
            )

            logger.info(
                "Vector MMR search completed | "
                "collection={} | result_count={}",
                self.collection_name,
                len(documents),
            )

            return documents

        except Exception as exc:
            logger.exception(
                "Vector MMR search failed | collection={} | "
                "error_type={} | error={}",
                self.collection_name,
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Metadata Utilities
    # =========================================================================

    @staticmethod
    def validate_document_metadata(
        document: Document,
    ) -> bool:
        """
        Validate minimum metadata required for a clinical knowledge
        document.

        Recommended metadata:

            document_id
            document_type
            title
            source
            source_type
            specialty
            condition
            organization
            version
            effective_date
            jurisdiction

        Returns:
            True when the document has useful metadata.
        """

        metadata = document.metadata or {}

        required_fields = [
            "document_id",
            "source",
        ]

        return all(
            metadata.get(field)
            for field in required_fields
        )

    @staticmethod
    def build_metadata_filter(
        *,
        specialty: Optional[str] = None,
        condition: Optional[str] = None,
        document_type: Optional[str] = None,
        organization: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build a standard clinical knowledge-base metadata filter.
        """

        filter_data: Dict[str, Any] = {}

        if specialty:
            filter_data["specialty"] = specialty

        if condition:
            filter_data["condition"] = condition

        if document_type:
            filter_data["document_type"] = document_type

        if organization:
            filter_data["organization"] = organization

        if jurisdiction:
            filter_data["jurisdiction"] = jurisdiction

        if version:
            filter_data["version"] = version

        logger.debug(
            "Clinical vector metadata filter created | "
            "filter_fields={}",
            list(filter_data.keys()),
        )

        return filter_data

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return vector-store operational metadata.

        This does not perform a clinical search.
        """

        result = {
            "status": "configured",
            "vector_store_type": (
                self.vector_store.__class__.__name__
            ),
            "collection_name": self.collection_name,
            "default_top_k": self.default_top_k,
        }

        logger.debug(
            "VectorStoreManager health check completed | "
            "status={} | vector_store={}",
            result["status"],
            result["vector_store_type"],
        )

        return result

    # =========================================================================
    # Underlying Store Access
    # =========================================================================

    def get_vector_store(self) -> VectorStore:
        """
        Return the underlying LangChain VectorStore.

        Use this only when provider-specific functionality is required.
        """

        logger.debug(
            "Underlying vector store requested | "
            "type={}",
            self.vector_store.__class__.__name__,
        )

        return self.vector_store