"""
retriever.py

Clinical evidence retrieval layer for the Agentic Clinical Decision
Support and Diagnosis system.

Responsibilities:
    - Wrap the configured LangChain retriever.
    - Retrieve relevant clinical evidence.
    - Apply optional metadata filters.
    - Limit the number of returned documents.
    - Provide async and sync retrieval.
    - Log retrieval lifecycle safely using Loguru.

Non-responsibilities:
    - Diagnosis
    - Differential diagnosis
    - Clinical reasoning
    - Treatment recommendations
    - Test recommendations
    - Clinical decision making

Architecture:

    EvidenceAgent
          |
          v
    ClinicalRetriever
          |
          v
    LangChain BaseRetriever
          |
          v
    Vector Store
          |
          v
    Clinical Knowledge Base

```

### How it fits with your `EvidenceAgent`

The dependency chain becomes:

```text
ClinicalGraph
      │
      ▼
EvidenceAgent
      │
      ▼
ClinicalRetriever
      │
      ▼
LangChain BaseRetriever
      │
      ▼
Vector Store
      │
      ▼
Clinical Knowledge Base
```

For example, your `EvidenceAgent` can receive the wrapper:

```python
evidence_agent = EvidenceAgent(
    llm=llm,
    retriever=clinical_retriever,
    top_k=5,
)
```

and internally:

```python
documents = await self.retriever.aretrieve(
    query=query,
)
```

### One architectural improvement

I would keep **retrieval filtering separate from clinical reasoning**:

```text
Retriever
    ↓
"Which documents are relevant?"

Evidence Agent
    ↓
"What does the retrieved evidence say?"

Differential Agent
    ↓
"Which diagnostic possibilities should be considered?"

Critic Agent
    ↓
"Is the reasoning grounded, consistent and safe?"
```

This separation is particularly important in your healthcare architecture because the retriever should be **deterministic and auditable wherever possible**, while the LLM agents perform synthesis.

Also, the `retriever.py` above intentionally does **not** log the actual clinical query, document contents, lab values, medications, or retrieved text. For a production healthcare deployment, those can contain PHI or sensitive clinical information.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.config.settings import settings
logger = settings.get_logger()


from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


# ============================================================================
# Clinical Retriever
# ============================================================================


class ClinicalRetriever:
    """
    Production-oriented wrapper around a LangChain BaseRetriever.

    The wrapper provides a stable application-level interface so that
    the EvidenceAgent does not need to know which vector database or
    retrieval implementation is being used.

    Example implementations behind BaseRetriever:

        - Chroma
        - Pinecone
        - FAISS
        - Elasticsearch
        - Azure AI Search
        - PGVector
        - Custom retriever
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        *,
        top_k: int = 5,
        min_content_length: int = 20,
    ) -> None:
        """
        Initialize ClinicalRetriever.

        Args:
            retriever:
                Configured LangChain BaseRetriever.

            top_k:
                Maximum number of documents returned.

            min_content_length:
                Minimum acceptable document content length.

        Raises:
            ValueError:
                If top_k is invalid.
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be greater than or equal to 1."
            )

        if min_content_length < 0:
            raise ValueError(
                "min_content_length cannot be negative."
            )

        self.retriever = retriever
        self.top_k = top_k
        self.min_content_length = min_content_length

        logger.debug(
            "ClinicalRetriever initialized | "
            "retriever={} | top_k={} | min_content_length={}",
            retriever.__class__.__name__,
            top_k,
            min_content_length,
        )

    # =========================================================================
    # Async Retrieval
    # =========================================================================

    async def aretrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve clinical evidence asynchronously.

        Args:
            query:
                Evidence retrieval query.

            filters:
                Optional metadata filters.

        Returns:
            List of filtered and limited LangChain Documents.

        Note:
            Metadata filtering is only applied locally when the underlying
            retriever does not natively support filtering.

            For production vector databases, prefer native metadata
            filtering at the vector-store level.
        """

        if not query or not query.strip():
            logger.warning(
                "Clinical evidence retrieval skipped | reason=empty_query"
            )
            return []

        normalized_query = query.strip()

        logger.info(
            "Clinical evidence retrieval started | "
            "query_length={} | filter_count={} | top_k={}",
            len(normalized_query),
            len(filters or {}),
            self.top_k,
        )

        try:
            documents = await self.retriever.ainvoke(
                normalized_query
            )

            logger.debug(
                "Raw clinical documents retrieved | count={}",
                len(documents),
            )

            documents = self._apply_filters(
                documents=documents,
                filters=filters,
            )

            documents = self._filter_invalid_documents(
                documents
            )

            documents = documents[: self.top_k]

            logger.info(
                "Clinical evidence retrieval completed | "
                "result_count={}",
                len(documents),
            )

            return documents

        except Exception as exc:
            logger.exception(
                "Clinical evidence retrieval failed | "
                "error_type={} | error={}",
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Sync Retrieval
    # =========================================================================

    def retrieve(
        self,
        query: str,
        *,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        Retrieve clinical evidence synchronously.

        Prefer aretrieve() in the FastAPI asynchronous request path.
        """

        if not query or not query.strip():
            logger.warning(
                "Clinical evidence retrieval skipped | reason=empty_query"
            )
            return []

        normalized_query = query.strip()

        logger.info(
            "Clinical evidence synchronous retrieval started | "
            "query_length={} | filter_count={} | top_k={}",
            len(normalized_query),
            len(filters or {}),
            self.top_k,
        )

        try:
            documents = self.retriever.invoke(
                normalized_query
            )

            logger.debug(
                "Raw clinical documents retrieved | count={}",
                len(documents),
            )

            documents = self._apply_filters(
                documents=documents,
                filters=filters,
            )

            documents = self._filter_invalid_documents(
                documents
            )

            documents = documents[: self.top_k]

            logger.info(
                "Clinical evidence synchronous retrieval completed | "
                "result_count={}",
                len(documents),
            )

            return documents

        except Exception as exc:
            logger.exception(
                "Clinical evidence synchronous retrieval failed | "
                "error_type={} | error={}",
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Filtering
    # =========================================================================

    def _apply_filters(
        self,
        *,
        documents: List[Document],
        filters: Optional[Dict[str, Any]],
    ) -> List[Document]:
        """
        Apply metadata filters to retrieved documents.

        Example metadata:

            {
                "document_type": "clinical_guideline",
                "specialty": "cardiology",
                "condition": "heart_failure",
                "organization": "example_org",
                "version": "2026.1",
                "jurisdiction": "US",
                "effective_date": "2026-01-01"
            }

        Important:
            For large production vector stores, filtering should ideally
            happen before similarity retrieval using the vector store's
            native metadata-filtering capabilities.
        """

        if not filters:
            return documents

        logger.debug(
            "Applying clinical evidence metadata filters | "
            "document_count={} | filter_count={}",
            len(documents),
            len(filters),
        )

        filtered_documents: List[Document] = []

        for document in documents:
            metadata = document.metadata or {}

            if self._matches_filters(
                metadata=metadata,
                filters=filters,
            ):
                filtered_documents.append(document)

        logger.debug(
            "Clinical evidence metadata filtering completed | "
            "before={} | after={}",
            len(documents),
            len(filtered_documents),
        )

        return filtered_documents

    @staticmethod
    def _matches_filters(
        *,
        metadata: Dict[str, Any],
        filters: Dict[str, Any],
    ) -> bool:
        """
        Determine whether a document satisfies all requested filters.

        Current behavior:
            Exact equality matching.

        Example:

            filters = {
                "specialty": "cardiology",
                "document_type": "guideline",
            }
        """

        for key, expected_value in filters.items():
            actual_value = metadata.get(key)

            if actual_value != expected_value:
                return False

        return True

    # =========================================================================
    # Document Validation
    # =========================================================================

    def _filter_invalid_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """
        Remove documents that do not contain usable content.
        """

        valid_documents: List[Document] = []

        invalid_count = 0

        for document in documents:
            if not isinstance(document, Document):
                invalid_count += 1
                continue

            content = document.page_content

            if not content:
                invalid_count += 1
                continue

            if len(content.strip()) < self.min_content_length:
                invalid_count += 1
                continue

            valid_documents.append(document)

        if invalid_count:
            logger.warning(
                "Invalid clinical evidence documents removed | "
                "invalid_count={}",
                invalid_count,
            )

        return valid_documents

    # =========================================================================
    # Retrieval With Clinical Context
    # =========================================================================

    async def aretrieve_for_clinical_context(
        self,
        *,
        query: str,
        specialty: Optional[str] = None,
        condition: Optional[str] = None,
        document_type: Optional[str] = None,
        jurisdiction: Optional[str] = None,
        organization: Optional[str] = None,
    ) -> List[Document]:
        """
        Retrieve evidence using common clinical knowledge-base metadata.

        This method provides an application-level interface for the
        EvidenceAgent.

        Example:

            await retriever.aretrieve_for_clinical_context(
                query="heart failure diagnostic evaluation",
                specialty="cardiology",
                condition="heart_failure",
                document_type="clinical_guideline",
            )
        """

        filters: Dict[str, Any] = {}

        if specialty:
            filters["specialty"] = specialty

        if condition:
            filters["condition"] = condition

        if document_type:
            filters["document_type"] = document_type

        if jurisdiction:
            filters["jurisdiction"] = jurisdiction

        if organization:
            filters["organization"] = organization

        logger.debug(
            "Clinical context retrieval requested | "
            "specialty={} | condition={} | document_type={} | "
            "jurisdiction={} | organization={}",
            specialty,
            condition,
            document_type,
            jurisdiction,
            organization,
        )

        return await self.aretrieve(
            query=query,
            filters=filters or None,
        )

    # =========================================================================
    # Document Metadata
    # =========================================================================

    @staticmethod
    def get_document_metadata(
        document: Document,
    ) -> Dict[str, Any]:
        """
        Return safe metadata associated with a retrieved document.

        This method deliberately returns metadata only and does not
        expose the document content.
        """

        return dict(document.metadata or {})

    @staticmethod
    def get_document_ids(
        documents: List[Document],
    ) -> List[str]:
        """
        Extract document identifiers from retrieved documents.

        Supported metadata keys:

            - document_id
            - source_id
            - id

        Returns:
            Unique document identifiers.
        """

        document_ids: List[str] = []

        for document in documents:
            metadata = document.metadata or {}

            document_id = (
                metadata.get("document_id")
                or metadata.get("source_id")
                or metadata.get("id")
            )

            if document_id is not None:
                document_ids.append(str(document_id))

        return list(dict.fromkeys(document_ids))

    # =========================================================================
    # Health / Diagnostics
    # =========================================================================

    def health_check(self) -> Dict[str, Any]:
        """
        Return retriever configuration information.

        This does not execute a clinical retrieval.
        """

        health = {
            "status": "configured",
            "retriever_type": self.retriever.__class__.__name__,
            "top_k": self.top_k,
            "min_content_length": self.min_content_length,
        }

        logger.debug(
            "ClinicalRetriever health check completed | status={}",
            health["status"],
        )

        return health