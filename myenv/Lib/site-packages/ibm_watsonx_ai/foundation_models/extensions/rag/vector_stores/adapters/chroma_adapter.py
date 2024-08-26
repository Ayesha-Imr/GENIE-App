#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------

from ibm_watsonx_ai.foundation_models.extensions.rag.vector_stores.langchain_vector_store_adapter import (
    LangChainVectorStoreAdapter,
)
from ibm_watsonx_ai.wml_client_error import MissingExtension

try:
    from langchain_chroma import Chroma
except ImportError:
    raise MissingExtension("langchain_chroma")


class ChromaLangchainAdapter(LangChainVectorStoreAdapter):

    def __init__(self, vector_store: Chroma) -> None:
        super().__init__(vector_store)

    def get_client(self) -> Chroma:
        return super().get_client()

    def clear(self) -> None:
        client = self.get_client()
        all_docs_ids = client.get()["ids"]
        if len(all_docs_ids) > 0:
            self.delete(all_docs_ids)

    def count(self) -> int:
        client = self.get_client()
        return len(client.get()["ids"])
