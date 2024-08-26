#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------


from __future__ import annotations
import time
import os
from typing import TypeAlias, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from functools import reduce, partial
from enum import Enum
import threading

from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    InvalidMultipleArguments,
    ParamOutOfRange,
)
from .base_embeddings import BaseEmbeddings
from ibm_watsonx_ai.wml_resource import WMLResource
import ibm_watsonx_ai._wrappers.requests as requests

from ibm_watsonx_ai.foundation_models.utils.utils import (
    _set_session_default_params,
    _get_requests_session,
)

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from ibm_watsonx_ai import Credentials

# Type Aliasses
ParamsType: TypeAlias = dict[str, str | dict[str, str]]
PayloadType: TypeAlias = dict[str, str | list[str] | ParamsType]


__all__ = ["Embeddings"]

# Do not change below, required by service
_MAX_INPUTS_LENGTH = 1000


class Embeddings(BaseEmbeddings, WMLResource):
    """Instantiate the embeddings service.

    :param model_id: the type of model to use
    :type model_id: str, optional

    :param params: parameters to use during generate requests, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames
    :type params: dict, optional

    :param credentials: credentials to Watson Machine Learning instance
    :type credentials: dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param api_client: Initialized APIClient object with set project or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param verify: user can pass as verify one of following:

        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    :param persistent_connection: Whether to keep persistent connection when evaluating `generate`, 'embed_query' and 'embed_documents` methods with one prompt
                                  or batch of prompts not longer than documentation allows (for more details see https://cloud.ibm.com/apidocs/watsonx-ai#text-embeddings).
                                  To close connection run `embeddings.close_persistent_connection()`, defaults to True. Added in 1.1.2.
    :type persistent_connection: bool, optional

    .. note::
        One of these parameters is required: [``project_id``, ``space_id``] when ``credentials`` parameter passed.

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import Embeddings
        from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames as EmbedParams
        from ibm_watsonx_ai.foundation_models.utils.enums import EmbeddingTypes

       embed_params = {
            EmbedParams.TRUNCATE_INPUT_TOKENS: 3,
            EmbedParams.RETURN_OPTIONS: {
            'input_text': True
            }
        }

        embedding = Embeddings(
            model_id=EmbeddingTypes.IBM_SLATE_30M_ENG,
            params=embed_params,
            credentials=Credentials(
                api_key = "***",
                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    # thread local storage definition
    _thread_local = threading.local()

    def __init__(
        self,
        *,
        model_id: str,
        params: ParamsType | None = None,
        credentials: Credentials | dict[str, str] | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        api_client: APIClient | None = None,
        verify: bool | str | None = None,
        persistent_connection: bool = True,
    ) -> None:
        if isinstance(model_id, Enum):
            self.model_id = model_id.value
        else:
            self.model_id = model_id

        self.params = params

        Embeddings._validate_type(params, "params", dict, False)

        if credentials:
            from ibm_watsonx_ai import APIClient

            self._client = APIClient(credentials, verify=verify)
        elif api_client:
            self._client = api_client
        else:
            raise InvalidMultipleArguments(
                params_names_list=["credentials", "api_client"],
                reason="None of the arguments were provided.",
            )

        if space_id:
            self._client.set.default_space(space_id)
        elif project_id:
            self._client.set.default_project(project_id)
        elif not api_client:
            raise InvalidMultipleArguments(
                params_names_list=["space_id", "project_id"],
                reason="None of the arguments were provided.",
            )
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 5.0:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        self._persistent_connection = persistent_connection
        self.__session: requests.Session | None = None
        WMLResource.__init__(self, __name__, self._client)

    @property
    def _session(self) -> requests.Session:
        if self.__session is None:
            self.__session = _set_session_default_params(requests.Session())
        return self.__session

    def generate(
        self,
        inputs: list[str],
        params: ParamsType | None = None,
        concurrency_limit: int = 10,
    ) -> dict:
        """Generates embeddings vectors for the given input with the given
        parameters and returns a REST API response.

        :param inputs: List of texts for which embedding vectors will be generated.
        :type inputs: list[str]
        :param params: meta props for embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :param concurrency_limit: number of requests that will be sent in parallel, max is 10, defaults to 10
        :type concurrency_limit: int, optional
        :return: scoring results containing generated embeddings vectors
        :rtype: dict
        """
        self._validate_type(inputs, "inputs", list, True)
        generate_url = (
            self._client.service_instance._href_definitions.get_fm_embeddings_href()
        )
        if concurrency_limit > 10 or concurrency_limit < 1:
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )

        try:
            if len(inputs) > _MAX_INPUTS_LENGTH:
                generated_responses = []
                inputs_splited = [
                    inputs[i : i + _MAX_INPUTS_LENGTH]
                    for i in range(0, len(inputs), _MAX_INPUTS_LENGTH)
                ]
                _generate_partial = partial(self._generate, generate_url, params=params)
                if (inputs_length := len(inputs_splited)) <= concurrency_limit:
                    with ThreadPoolExecutor(max_workers=inputs_length) as executor:
                        generated_responses = list(
                            executor.map(_generate_partial, inputs_splited)
                        )
                else:
                    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                        generated_responses = list(
                            executor.map(_generate_partial, inputs_splited)
                        )

                def reduce_response(left: dict, right: dict) -> dict:
                    import copy

                    left_copy = copy.deepcopy(left)
                    left_copy["results"].extend(right["results"])
                    left_copy["input_token_count"] += right["input_token_count"]
                    return left_copy

                return reduce(
                    reduce_response, generated_responses[1:], generated_responses[0]
                )

            else:
                results = self._generate(
                    generate_url, inputs, params, _session=self._session
                )
        except Exception:
            # closing the connection if an error occurs and if persistent connection is true
            if self._persistent_connection and self._session is not None:
                self._session.close()
            raise
        finally:
            # closing the connection if an error occurs or if persistent connection is false
            if not self._persistent_connection and self._session is not None:
                self._session.close()
        return results

    def embed_documents(
        self,
        texts: list[str],
        params: ParamsType | None = None,
        concurrency_limit: int = 10,
    ) -> list[list[float]]:
        """Return list of embedding vectors for provided texts.

        :param texts: List of texts for which embedding vectors will be generated.
        :type texts: list[str]
        :param params: meta props for embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :param concurrency_limit: number of requests that will be sent in parallel, max is 10, defaults to 10
        :type concurrency_limit: int, optional

        :return: List of embedding vectors
        :rtype: list[list[float]]

        **Example**

        .. code-block:: python

            q = [
                "What is a Generative AI?",
                "Generative AI refers to a type of artificial intelligence that can original content."
                ]

            embedding_vectors = embedding.embed_documents(texts=q)
            print(embedding_vectors)
        """
        return [
            vector.get("embedding")
            for vector in self.generate(
                inputs=texts, params=params, concurrency_limit=concurrency_limit
            ).get("results", [{}])
        ]

    def embed_query(self, text: str, params: ParamsType | None = None) -> list[float]:
        """Return embedding vector for provided text.

        :param text: Text for which embedding vector will be generated.
        :type text: str
        :param params: meta props for embedding generation, use ``ibm_watsonx_ai.metanames.EmbedTextParamsMetaNames().show()`` to view the list of MetaNames, defaults to None
        :type params: ParamsType | None, optional
        :return: Embedding vector
        :rtype: list[float]

        **Example**

        .. code-block:: python

            q = "What is a Generative AI?"
            embedding_vector = embedding.embed_query(text=q)
            print(embedding_vector)
        """
        return (
            self.generate(inputs=[text], params=params)
            .get("results", [{}])[0]
            .get("embedding")
        )

    def _prepare_payload(
        self, inputs: list[str], params: ParamsType | None = None
    ) -> PayloadType:
        """Prepare payload based in provided inputs and params."""
        payload: PayloadType = {"model_id": self.model_id, "inputs": inputs}

        if params is not None:
            payload["parameters"] = params
        elif self.params is not None:
            payload["parameters"] = self.params

        if self._client.default_project_id:
            payload["project_id"] = self._client.default_project_id
        elif self._client.default_space_id:
            payload["space_id"] = self._client.default_space_id

        return payload

    def _generate(
        self,
        generate_url: str,
        inputs: list[str],
        params: ParamsType | None = None,
        _session: requests.Session | None = None,
    ) -> dict:
        """Send request with post and return service response."""

        payload = self._prepare_payload(inputs, params)

        retries = 0
        # use separate session for each thread when user provide list of prompts
        session = (
            _session
            if _session is not None
            else _get_requests_session(Embeddings._thread_local)
        )
        while retries < 3:
            with session.post(
                url=generate_url,
                json=payload,
                params=self._client._params(skip_for_create=True, skip_userfs=True),
                headers=self._client._get_headers(),
            ) as response_scoring:
                if response_scoring.status_code in [429, 500, 503, 504, 520]:
                    time.sleep(2**retries)
                    retries += 1
                else:
                    break

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="embedding",
        )

    def to_dict(self) -> dict:
        data = super().to_dict()

        data.update(
            {
                "model_id": self.model_id,
                "params": self.params,
                "credentials": self._client.credentials.to_dict(),
                "project_id": self._client.default_project_id,
                "space_id": self._client.default_space_id,
                "verify": os.environ.get("WML_CLIENT_VERIFY_REQUESTS"),
            }
        )

        return data

    def close_persistent_connection(self) -> None:
        """Only applicable if persistent_connection was set to True in Embeddings initialization."""
        if self._persistent_connection and self._session is not None:
            self._session.close()
