
import os
import hashlib
import json
import logging

from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


logger = logging.getLogger(__name__)

# Configuration
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "text-embedding-3-small"


class RetrieverBuilder:
    """
    Manage ChromaDB and build a hybrid BM25 + vector retriever.

    Parameters
    ----------
    persist_directory : str
        Local directory used to persist ChromaDB.
    collection_name : str
        ChromaDB collection name.
    embedding_model : str
        OpenAI embedding model.
    api_key : str, optional
        OpenAI API key. Defaults to OPENAI_API_KEY.
    """

    def __init__(
        self,
        persist_directory="./data/chroma",
        collection_name=COLLECTION_NAME,
        embedding_model=EMBEDDING_MODEL,
        api_key=None,
    ):

        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
        )

        self.vector_store = None

        # Create persistence directory
        os.makedirs(self.persist_directory, exist_ok=True)

        logger.info(
            "RetrieverBuilder initialized with model %s",
            self.embedding_model
        )



    def create_vector_store(self):
        """
        Create or open the persistent ChromaDB collection.
        """

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
            collection_configuration={
                "hnsw": {"space": "cosine"}
            },
        )

        logger.info(
            "Chroma collection '%s' is ready.",
            self.collection_name
        )

        return self.vector_store

   

    def _get_vector_store(self):
        """
        Return the vector store, initializing it if necessary.
        """

        if self.vector_store is None:
            return self.create_vector_store()

        return self.vector_store

  

    @staticmethod
    def _make_document_id(document):
        """
        Generate a deterministic SHA256 identifier.

        The identifier changes if text or metadata changes.
        """

        payload = json.dumps(
            {
                "text": document.page_content,
                "metadata": document.metadata,
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

        return hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()



    def add_documents(self, documents):
        """
        Add documents to ChromaDB.

        Existing identical documents are skipped.

        Returns
        -------
        int
            Number of documents actually added.
        """

        vector_store = self._get_vector_store()

        # Remove empty documents
        valid_documents = [
            doc for doc in documents
            if isinstance(doc, Document)
            and doc.page_content.strip()
        ]

        if not valid_documents:
            logger.warning("No valid documents to add.")
            return 0

        # Filter complex metadata
        cleaned_documents = filter_complex_metadata(
            valid_documents
        )

        # Remove duplicate documents within the input
        unique_documents = {
            self._make_document_id(doc): doc
            for doc in cleaned_documents
        }

        if not unique_documents:
            return 0

        # Check existing document IDs
        existing_ids = set(
            vector_store.get(
                ids=list(unique_documents.keys()),
                include=[],
            )["ids"]
        )

        # Keep only new documents
        new_documents = {
            doc_id: doc
            for doc_id, doc in unique_documents.items()
            if doc_id not in existing_ids
        }

        if not new_documents:
            logger.info("All documents already exist.")
            return 0

        # Insert new documents and embeddings
        vector_store.add_documents(
            documents=list(new_documents.values()),
            ids=list(new_documents.keys()),
        )

        logger.info(
            "Added %d documents, skipped %d.",
            len(new_documents),
            len(valid_documents) - len(new_documents),
        )

        return len(new_documents)



    def _load_documents(self):
        """
        Load all documents stored in ChromaDB.
        """

        stored = self._get_vector_store().get(
            include=["documents", "metadatas"]
        )

        return [
            Document(
                page_content=text,
                metadata=metadata or {},
            )
            for text, metadata in zip(
                stored["documents"],
                stored["metadatas"],
            )
            if text and text.strip()
        ]


    def build_hybrid_retriever(
        self,
        k=5,
        weights=None,
    ):
        """
        Build a hybrid BM25 + vector retriever.

        Parameters
        ----------
        k : int
            Number of documents retrieved by each retriever.
        weights : list
            Weights for BM25 and vector search.

        Returns
        -------
        EnsembleRetriever
            Hybrid retriever.
        """

        vector_store = self._get_vector_store()

        documents = self._load_documents()

        if not documents:
            raise ValueError(
                "Cannot build a retriever from an empty database."
            )

        # BM25 retriever
        bm25_retriever = BM25Retriever.from_documents(
            documents,
            k=k,
        )

        # Vector retriever
        vector_retriever = vector_store.as_retriever(
            search_kwargs={"k": k}
        )

        # Default weights
        if weights is None:
            weights = [0.5, 0.5]

        if len(weights) != 2:
            raise ValueError(
                "Weights must contain exactly two values."
            )

        if any(weight < 0 for weight in weights):
            raise ValueError("Weights cannot be negative.")

        if sum(weights) <= 0:
            raise ValueError("At least one weight must be positive.")

        # Hybrid retriever
        hybrid_retriever = EnsembleRetriever(
            retrievers=[
                bm25_retriever,
                vector_retriever,
            ],
            weights=weights,
        )

        logger.info(
            "Hybrid retriever built with %d documents.",
            len(documents),
        )

        return hybrid_retriever


    def search(self, query, k=5):
        """
        Perform semantic vector search.
        """

        vector_store = self._get_vector_store()

        return vector_store.similarity_search(
            query,
            k=k,
        )


    def get_collection_stats(self):
        """
        Return basic collection statistics.
        """

        vector_store = self._get_vector_store()

        return {
            "collection_name": self.collection_name,
            "document_count": vector_store._collection.count(),
            "embedding_model": self.embedding_model,
            "persist_directory": self.persist_directory,
        }
