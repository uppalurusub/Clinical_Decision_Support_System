"""
embeddings.py

Embedding layer for the Agentic Clinical Decision Support
and Diagnosis system.

Architecture:

    Clinical Documents
          |
          v
    Document Loader
          |
          v
        Chunker
          |
          v
     EmbeddingManager
          |
          v
    VectorStoreManager
          |
          v
     Vector Database

Responsibilities:
    - Encapsulate LangChain embedding model.
    - Generate embeddings for documents.
    - Generate embeddings for queries.
    - Support synchronous and asynchronous operations.
    - Provide configuration and health information.
    - Log operational information using Loguru.

Non-responsibilities:
    - Document retrieval.
    - Similarity search.
    - Clinical reasoning.
    - Diagnosis.
    - Treatment recommendations.
    - Patient-specific decisions.

Important:
    Never log raw document text, clinical notes, patient data,
    embeddings, prompts, or other PHI/sensitive information.

```

### How this fits with your current RAG architecture

You now have a clean four-layer RAG foundation:

```text
                    Clinical Knowledge
                           │
                           ▼
                  ┌─────────────────┐
                  │ Document Loader │
                  └────────┬────────┘
                           │
                           ▼
                     Chunking
                           │
                           ▼
                  ┌─────────────────┐
                  │ EmbeddingManager│
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ VectorStore     │
                  │ Manager         │
                  └────────┬────────┘
                           │
                           ▼
                    Vector Database
                           │
                           ▼
                  ┌─────────────────┐
                  │ Clinical        │
                  │ Retriever       │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ Evidence Agent  │
                  └────────┬────────┘
                           │
                           ▼
                     ClinicalState
```

The responsibilities remain deliberately separated:

| Component           | Responsibility               |
| ------------------- | ---------------------------- |
| `embeddings.py`     | Text → embedding vector      |
| `vector_store.py`   | Store/search vectors         |
| `retriever.py`      | Retrieve relevant documents  |
| `evidence_agent.py` | Interpret retrieved evidence |

### Example initialization

For example, with a LangChain embedding implementation:

```python
from langchain_openai import OpenAIEmbeddings

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large",
)

embedding_manager = EmbeddingManager(
    embeddings=embedding_model,
    model_name="text-embedding-3-large",
    batch_size=32,
)
```

Then your vector store can receive the underlying LangChain embeddings:

```python
vector_store = SomeLangChainVectorStore(
    embedding=embedding_manager.get_embeddings(),
)
```

and your application wrapper:

```python
vector_store_manager = VectorStoreManager(
    vector_store=vector_store,
    collection_name="clinical_knowledge",
    default_top_k=5,
)
```

One architectural point I would preserve throughout this project: **the embedding model should never be responsible for deciding what clinical evidence means**. Its job is semantic representation; retrieval determines relevance; the Evidence Agent performs grounded interpretation; and the Critic Agent challenges that interpretation.

"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from app.config.settings import settings
logger = settings.get_logger()

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ============================================================================
# Embedding Manager
# ============================================================================


class EmbeddingManager:
    """
    Application-level wrapper around a LangChain Embeddings implementation.

    The wrapper keeps the rest of the application independent from the
    specific embedding provider.

    Possible underlying implementations include:

        - OpenAI embeddings
        - Azure OpenAI embeddings
        - Hugging Face embeddings
        - Sentence Transformers
        - Cohere embeddings
        - Custom enterprise embedding models
    """

    def __init__(
        self,
        embeddings: Embeddings,
        *,
        model_name: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        """
        Initialize EmbeddingManager.

        Args:
            embeddings:
                Configured LangChain Embeddings implementation.

            model_name:
                Optional logical model name used for observability.

            batch_size:
                Batch size used when embedding multiple documents.

        Raises:
            ValueError:
                If batch_size is invalid.
        """

        if batch_size < 1:
            raise ValueError(
                "batch_size must be greater than or equal to 1."
            )

        self.embeddings = embeddings
        self.model_name = (
            model_name
            or embeddings.__class__.__name__
        )
        self.batch_size = batch_size

        logger.debug(
            "EmbeddingManager initialized | "
            "embedding_provider={} | model_name={} | batch_size={}",
            embeddings.__class__.__name__,
            self.model_name,
            batch_size,
        )

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def embedding_model(self) -> Embeddings:
        """
        Return the underlying LangChain embedding implementation.
        """

        return self.embeddings

    # =========================================================================
    # Single Text Embedding
    # =========================================================================

    def embed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate an embedding for a query.

        Args:
            text:
                Query text.

        Returns:
            Embedding vector.
        """

        if not text or not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        normalized_text = text.strip()

        logger.debug(
            "Generating query embedding | "
            "text_length={}",
            len(normalized_text),
        )

        try:
            vector = self.embeddings.embed_query(
                normalized_text
            )

            logger.debug(
                "Query embedding generated | "
                "dimension={}",
                len(vector),
            )

            return vector

        except Exception as exc:
            logger.exception(
                "Query embedding generation failed | "
                "error_type={} | error={}",
                exc.__class__.__name__,
                str(exc),
            )
            raise

    async def aembed_query(
        self,
        text: str,
    ) -> List[float]:
        """
        Generate a query embedding asynchronously.

        Preferred for asynchronous application workflows.
        """

        if not text or not text.strip():
            raise ValueError(
                "Query text cannot be empty."
            )

        normalized_text = text.strip()

        logger.debug(
            "Generating async query embedding | "
            "text_length={}",
            len(normalized_text),
        )

        try:
            vector = await self.embeddings.aembed_query(
                normalized_text
            )

            logger.debug(
                "Async query embedding generated | "
                "dimension={}",
                len(vector),
            )

            return vector

        except Exception as exc:
            logger.exception(
                "Async query embedding generation failed | "
                "error_type={} | error={}",
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # Multiple Text Embeddings
    # =========================================================================

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Documents are processed in batches to avoid unnecessarily
        large provider requests.

        Args:
            texts:
                Sequence of document/chunk texts.

        Returns:
            List of embedding vectors.
        """

        if not texts:
            logger.warning(
                "Document embedding skipped | reason=no_texts"
            )
            return []

        normalized_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not normalized_texts:
            logger.warning(
                "Document embedding skipped | "
                "reason=no_valid_texts"
            )
            return []

        logger.info(
            "Document embedding started | "
            "document_count={} | batch_size={} | model={}",
            len(normalized_texts),
            self.batch_size,
            self.model_name,
        )

        vectors: List[List[float]] = []

        try:
            for start in range(
                0,
                len(normalized_texts),
                self.batch_size,
            ):
                batch = normalized_texts[
                    start : start + self.batch_size
                ]

                logger.debug(
                    "Embedding document batch | "
                    "batch_start={} | batch_size={}",
                    start,
                    len(batch),
                )

                batch_vectors = (
                    self.embeddings.embed_documents(
                        batch
                    )
                )

                vectors.extend(batch_vectors)

            logger.info(
                "Document embedding completed | "
                "document_count={} | vector_count={}",
                len(normalized_texts),
                len(vectors),
            )

            return vectors

        except Exception as exc:
            logger.exception(
                "Document embedding failed | "
                "document_count={} | error_type={} | error={}",
                len(normalized_texts),
                exc.__class__.__name__,
                str(exc),
            )
            raise

    async def aembed_documents(
        self,
        texts: Sequence[str],
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple documents asynchronously.

        Documents are processed in batches.
        """

        if not texts:
            logger.warning(
                "Async document embedding skipped | reason=no_texts"
            )
            return []

        normalized_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not normalized_texts:
            logger.warning(
                "Async document embedding skipped | "
                "reason=no_valid_texts"
            )
            return []

        logger.info(
            "Async document embedding started | "
            "document_count={} | batch_size={} | model={}",
            len(normalized_texts),
            self.batch_size,
            self.model_name,
        )

        vectors: List[List[float]] = []

        try:
            for start in range(
                0,
                len(normalized_texts),
                self.batch_size,
            ):
                batch = normalized_texts[
                    start : start + self.batch_size
                ]

                logger.debug(
                    "Async embedding document batch | "
                    "batch_start={} | batch_size={}",
                    start,
                    len(batch),
                )

                batch_vectors = (
                    await self.embeddings.aembed_documents(
                        batch
                    )
                )

                vectors.extend(batch_vectors)

            logger.info(
                "Async document embedding completed | "
                "document_count={} | vector_count={}",
                len(normalized_texts),
                len(vectors),
            )

            return vectors

        except Exception as exc:
            logger.exception(
                "Async document embedding failed | "
                "document_count={} | error_type={} | error={}",
                len(normalized_texts),
                exc.__class__.__name__,
                str(exc),
            )
            raise

    # =========================================================================
    # LangChain Document Helpers
    # =========================================================================

    def embed_documents_from_objects(
        self,
        documents: Sequence[Document],
    ) -> List[List[float]]:
        """
        Generate embeddings directly from LangChain Documents.

        The document metadata is preserved by the caller; this method
        only extracts page_content for embedding.
        """

        if not documents:
            logger.warning(
                "Document-object embedding skipped | "
                "reason=no_documents"
            )
            return []

        logger.info(
            "Embedding LangChain documents | "
            "document_count={}",
            len(documents),
        )

        texts = [
            document.page_content
            for document in documents
            if document.page_content
            and document.page_content.strip()
        ]

        logger.debug(
            "Valid document contents prepared | count={}",
            len(texts),
        )

        return self.embed_documents(texts)

    async def aembed_documents_from_objects(
        self,
        documents: Sequence[Document],
    ) -> List[List[float]]:
        """
        Asynchronously generate embeddings from LangChain Documents.
        """

        if not documents:
            logger.warning(
                "Async document-object embedding skipped | "
                "reason=no_documents"
            )
            return []

        logger.info(
            "Async embedding LangChain documents | "
            "document_count={}",
            len(documents),
        )

        texts = [
            document.page_content
            for document in documents
            if document.page_content
            and document.page_content.strip()
        ]

        logger.debug(
            "Valid async document contents prepared | count={}",
        )

        return await self.aembed_documents(texts)

    # =========================================================================
    # Validation
    # =========================================================================

    @staticmethod
    def validate_embedding(
        vector: Sequence[float],
    ) -> bool:
        """
        Validate that an embedding vector is structurally usable.

        This checks structure only. It does not assess semantic quality.
        """

        if not vector:
            return False

        for value in vector:
            if not isinstance(value, (int, float)):
                return False

        return True

    def validate_embedding_dimension(
        self,
        vector: Sequence[float],
        expected_dimension: int,
    ) -> bool:
        """
        Validate embedding dimensionality.
        """

        if expected_dimension < 1:
            raise ValueError(
                "expected_dimension must be greater than zero."
            )

        valid = (
            self.validate_embedding(vector)
            and len(vector) == expected_dimension
        )

        logger.debug(
            "Embedding dimension validation | "
            "expected={} | actual={} | valid={}",
            expected_dimension,
            len(vector),
            valid,
        )

        return valid

    # =========================================================================
    # Document Validation
    # =========================================================================

    @staticmethod
    def validate_document(
        document: Document,
        *,
        min_length: int = 20,
    ) -> bool:
        """
        Validate whether a LangChain Document contains embeddable text.
        """

        if not isinstance(document, Document):
            return False

        if not document.page_content:
            return False

        if len(document.page_content.strip()) < min_length:
            return False

        return True

    def filter_valid_documents(
        self,
        documents: Sequence[Document],
        *,
        min_length: int = 20,
    ) -> List[Document]:
        """
        Remove documents that cannot reasonably be embedded.
        """

        valid_documents = [
            document
            for document in documents
            if self.validate_document(
                document,
                min_length=min_length,
            )
        ]

        logger.debug(
            "Embedding document validation completed | "
            "input_count={} | valid_count={} | removed_count={}",
            len(documents),
            len(valid_documents),
            len(documents) - len(valid_documents),
        )

        return valid_documents

    # =========================================================================
    # Configuration
    # =========================================================================

    def get_configuration(self) -> Dict[str, Any]:
        """
        Return non-sensitive embedding configuration.

        Never expose API keys, credentials, provider tokens,
        or connection strings.
        """

        configuration = {
            "provider": (
                self.embeddings.__class__.__name__
            ),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
        }

        logger.debug(
            "Embedding configuration requested | "
            "provider={} | model={}",
            configuration["provider"],
            configuration["model_name"],
        )

        return configuration

    # =========================================================================
    # Health Check
    # =========================================================================

    def health_check(
        self,
        *,
        test_embedding: bool = False,
    ) -> Dict[str, Any]:
        """
        Return embedding service health information.

        Args:
            test_embedding:
                If True, execute a small embedding request.

        Warning:
            test_embedding=True may incur provider/API cost.
        """

        result: Dict[str, Any] = {
            "status": "configured",
            "provider": (
                self.embeddings.__class__.__name__
            ),
            "model_name": self.model_name,
            "batch_size": self.batch_size,
        }

        if not test_embedding:
            logger.debug(
                "Embedding health check completed | "
                "status=configured"
            )
            return result

        try:
            vector = self.embed_query(
                "clinical knowledge health check"
            )

            result["status"] = (
                "healthy"
                if self.validate_embedding(vector)
                else "unhealthy"
            )

            result["embedding_dimension"] = len(vector)

            logger.info(
                "Embedding health check executed | "
                "status={} | dimension={}",
                result["status"],
                result["embedding_dimension"],
            )

            return result

        except Exception as exc:
            logger.exception(
                "Embedding health check failed | "
                "error_type={} | error={}",
                exc.__class__.__name__,
                str(exc),
            )

            result["status"] = "unhealthy"
            result["error_type"] = exc.__class__.__name__

            return result

    # =========================================================================
    # Underlying Provider Access
    # =========================================================================

    def get_embeddings(self) -> Embeddings:
        """
        Return the underlying LangChain Embeddings implementation.

        Useful when initializing a LangChain VectorStore.
        """

        logger.debug(
            "Underlying embedding implementation requested | "
            "provider={}",
            self.embeddings.__class__.__name__,
        )

        return self.embeddings
