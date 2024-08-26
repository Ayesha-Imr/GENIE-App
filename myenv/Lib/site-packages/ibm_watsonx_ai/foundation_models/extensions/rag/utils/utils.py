#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import sys
import ssl
import base64
import pandas as pd
import logging
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


# Verbose display in notebooks


def verbose_search(
    question: str, documents: list[Document] | list[tuple[Document, float]]
) -> None:
    """Display a table with found documents.

    :param question: question/query used for search
    :type question: str
    :param documents: list of documents found with question or list of tuples (if search was done with scores)
    :type documents: list[langchain_core.documents.Document] | list[Document, float]
    :raises ImportError: if it is notebook environment but IPython is not found
    """

    # Unzip if list have tuples (Document, float)
    if all(isinstance(doc, tuple) for doc in documents):
        documents, scores = [doc[0] for doc in documents], [doc[1] for doc in documents]  # type: ignore[index]
    else:
        scores = []

    from langchain_core.documents import Document

    documents = cast(list[Document], documents)
    if "ipykernel" in sys.modules:
        try:
            from IPython.display import display, Markdown
        except ImportError:
            raise ImportError(
                "To use verbose search, please install make sure IPython package is installed."
            )

        display(Markdown(f"**Question:** {question}"))

        if len(documents) > 0:
            metadata_fields: set = set()

            for doc in documents:
                metadata_fields.update(doc.metadata.keys())

            if scores:
                metadata_fields.add("score")

            df = pd.DataFrame(columns=["page_content"] + list(metadata_fields))

            # Parsing rows and adding them to the DataFrame
            for doc in documents:
                row = {"page_content": doc.page_content}
                row.update(doc.metadata)
                # Adding score (if provided)
                if scores:
                    row["score"] = scores.pop(0)
                df = pd.concat(
                    [df, pd.DataFrame({key: [value] for key, value in row.items()})],
                    ignore_index=True,
                )

            display(df)
        else:
            display(Markdown("No documents were found."))
    else:
        if len(documents) > 0:
            if scores:
                for i, (d, s) in enumerate(zip(documents, scores)):
                    logger.info(f"{i} | {s} |  {d.page_content}   | {d.metadata}")
            else:
                for i, d in enumerate(documents):
                    logger.info(f"{i} |  {d.page_content}   | {d.metadata}")
        else:
            logger.info("No documents were found.")


# SSL Certificates


def is_valid_certificate(cert_string: str) -> bool:
    try:
        ssl.PEM_cert_to_DER_cert(cert_string)
        return True
    except Exception:
        return False


def get_ssl_certificate(cert: str) -> str:
    if is_valid_certificate(cert):
        return cert
    else:
        try:
            cert_decoded = base64.b64decode(cert).decode()
            if is_valid_certificate(cert_decoded):
                return cert_decoded
            else:
                raise ValueError("Not a valid SSL certificate.")
        except Exception as e:
            raise ValueError(
                f"Error occured when trying to get the SSL certificate: {e}"
            )


def save_ssl_certificate_as_file(ssl_certificate_content: str, file_path: str) -> str:
    ssl_certificate_content = get_ssl_certificate(ssl_certificate_content)
    with open(file_path, "w") as file:
        file.write(ssl_certificate_content)

    logger.info(
        f"SSL certificate was found and written to {file_path}. It will be used for the connection for the VectorStore."
    )
    return file_path
