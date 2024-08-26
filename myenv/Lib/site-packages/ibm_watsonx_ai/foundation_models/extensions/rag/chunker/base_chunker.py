#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from typing import Sequence, Any, Generic, TypeVar
from abc import ABC, abstractmethod


__all__ = [
    "BaseChunker",
]

ChunkType = TypeVar("ChunkType")


class BaseChunker(ABC, Generic[ChunkType]):
    """
    Class responsible for handling operations of splitting documents
    within the RAG application.
    """

    @abstractmethod
    def split_documents(self, documents: Sequence[ChunkType]) -> list[ChunkType]:
        """
        Split series of documents into smaller parts based on
        the provided chunker settings.

        :param documents: sequence of elements that contain context in the format of text
        :type: Sequence[ChunkType]

        :return: list of documents splitter into smaller ones, having less content
        :rtype: list[ChunkType]
        """

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Return dict that can be used to recreate instance of the BaseChunker."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: dict[str, Any]) -> "BaseChunker":
        """Create instance from the dictionary"""
