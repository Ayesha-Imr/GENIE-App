#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from abc import ABC, abstractmethod
from typing import Any
from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.base_vector_store import (
    BaseVectorStore,
)
from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    from langchain_core.documents import Document
except ImportError:
    raise MissingExtension("langchain")


class BaseRetriever(ABC):
    """Abstract class for all retriever handlers for the chosen vector store.
    Returns some document chunks in RAG pipeline using concrete ``retrieve`` implementation.

    :param vector_store: vector store used in document retrieval
    :type vector_store: BaseVectorStore
    """

    def __init__(self, vector_store: BaseVectorStore) -> None:
        super().__init__()
        self.vector_store: BaseVectorStore = vector_store

    @abstractmethod
    def retrieve(self, query: str, **kwargs: Any) -> list[Document]:
        """Retrieve elements from the VectorStore using the provided `query`.

        :param query: text query to be used for searching
        :type query: str

        :return: list of retrieved LangChain documents
        :rtype: list[langchain_core.documents.Document]
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """Serializes this Retriever ``init_parameters`` so that this Retriever can be reconstructed by ``from_vector_store`` class method.

        :return: serialized ``init_parameters``
        :rtype: dict
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_vector_store(
        cls,
        vector_store: BaseVectorStore,
        init_parameters: dict[str, Any] | None = None,
    ) -> "BaseRetriever":
        """Deserialize this ``Retriever`` into concrete one using arguments.

        :param vector_store: vector store used to create a Retriever
        :type vector_store: BaseVectorStore

        :param init_parameters: parameters to initialize retriever with
        :type init_parameters: dict[str, Any]

        :return: concrete Retriever or None if data is incorrect
        :rtype: BaseRetriever | None
        """
        raise NotImplementedError
