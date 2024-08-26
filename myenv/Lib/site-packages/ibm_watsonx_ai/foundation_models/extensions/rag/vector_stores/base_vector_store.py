#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document

from ibm_watsonx_ai.foundation_models.embeddings import BaseEmbeddings


class BaseVectorStore(ABC):
    """Base abstract class for all vector store-like classes. Interface that support simple database operations."""

    @abstractmethod
    def get_client(self) -> Any:
        """Returns underlying native VectorStore client.

        :return: wrapped VectorStore client
        :rtype: Any
        """
        pass

    @abstractmethod
    def set_embeddings(self, embedding_fn: BaseEmbeddings) -> None:
        """If possible, sets a default embedding function.
        Use types inheirted from ``BaseEmbeddings`` if you want to make it capable for ``RAGPattern`` deployment.
        Argument ``embedding_fn`` can be a LangChain embeddings but issues with serialization will occur.

        *Deprecated:* Method `set_embeddings` for class `VectorStore` is deprecated, since it may cause issues for 'langchain >= 0.2.0'.


        :param embedding_fn: embedding function
        :type embedding_fn: BaseEmbeddings
        """
        raise NotImplementedError(
            "This vector store cannot have embedding function set up."
        )

    @abstractmethod
    def add_documents(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        """Adds a list of documents to the RAG's vector store as upsert operation.
        IDs are determined by the text content of the document (hash) and redundant duplicates will not be added.

        List must contain either strings, dicts with a required field ``content`` of str type or LangChain ``Document``.

        :param content: unstructured list of data to be added
        :type content: list[str] | list[dict] | list

        :return: list of ids
        :rtype: list[str]
        """
        pass

    @abstractmethod
    async def add_documents_async(
        self, content: list[str] | list[dict] | list, **kwargs: Any
    ) -> list[str]:
        """Add document to the RAG's vector store asynchronously.
        List must contain either strings, dicts with a required field ``content`` of str type or LangChain ``Document``.

        :param content: unstructured list of data to be added
        :type content: list[str] | list[dict] | list

        :return: list of ids
        :rtype: list[str]
        """
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list:
        """Get documents that would fit the query.

        :param query: question asked by a user
        :type query: str

        :param k: max number of similar documents
        :type k: int

        :param include_scores: return scores for documents, defaults to False
        :type include_scores: bool, optional

        :param verbose: print formatted response to the output, defaults to False
        :type verbose: bool, optional

        :return: list of found documents
        :rtype: list
        """
        pass

    @abstractmethod
    def window_search(
        self,
        query: str,
        k: int,
        include_scores: bool = False,
        verbose: bool = False,
        window_size: int = 2,
        **kwargs: Any,
    ) -> list[Document]:
        """
        Similarly to search method, get documents (chunks) that would fit the query.
        Each chunk is extended to its adjacent chunks (if exist) from the same origin document.
        The adjacent chunks are merged into one chunk while keeping their order,
        and intersecting text between them is merged (if exists).
        This requires chunks to have "document_id" and "sequence_number" in their metadata.

        :param query: question asked by a user
        :type query: str

        :param k: max number of similar documents
        :type k: int

        :param include_scores: return scores for documents, defaults to False
        :type include_scores: bool, optional

        :param verbose: print formatted response to the output, defaults to False
        :type verbose: bool, optional

        :param window_size: number of adjacent chunks to retrieve before and after the center, according to the sequence_number.
        :type window_size: int

        :return: list of found documents (extended into windows).
        :rtype: list
        """
        raise NotImplementedError()

    @abstractmethod
    def delete(self, ids: list[str], **kwargs: Any) -> None:
        """Delete documents with provided ids.

        :param ids: IDs of documents to delete
        :type ids: list[str]
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clears the current collection that is being used by the VectorStore.
        Removes all documents with all their metadata and embeddings.
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """Return the number of docs in the current collection.

        :return: count of all documents in the collection
        :rtype: int
        """
        pass

    @abstractmethod
    def as_langchain_retriever(self, **kwargs: Any) -> Any:
        """Creates a LangChain retriever from this vector store.

        :return: LangChain retriever which can be used in LangChain pipelines
        :rtype: langchain_core.vectorstores.VectorStoreRetriever
        """
        pass
