#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

from typing import TYPE_CHECKING, Generator, AsyncGenerator, cast, overload, Literal
from enum import Enum
import asyncio

from ibm_watsonx_ai.wml_client_error import (
    WMLClientError,
    ParamOutOfRange,
    InvalidMultipleArguments,
    MissingExtension,
)
from .base_model_inference import BaseModelInference
from .fm_model_inference import FMModelInference
from .deployment_model_inference import DeploymentModelInference

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient, Credentials
    from langchain_ibm import WatsonxLLM


class ModelInference(BaseModelInference):
    """Instantiate the model interface.

    .. hint::
        To use the ModelInference class with LangChain, use the :func:`WatsonxLLM <ibm_watsonx_ai.foundation_models.extensions.langchain.WatsonxLLM>` wrapper.

    :param model_id: the type of model to use
    :type model_id: str, optional

    :param deployment_id: ID of tuned model's deployment
    :type deployment_id: str, optional

    :param credentials: credentials to Watson Machine Learning instance
    :type credentials: ibm_watsonx_ai.Credentials or dict, optional

    :param params: parameters to use during generate requests
    :type params: dict, optional

    :param project_id: ID of the Watson Studio project
    :type project_id: str, optional

    :param space_id: ID of the Watson Studio space
    :type space_id: str, optional

    :param verify: user can pass as verify one of following:

        - the path to a CA_BUNDLE file
        - the path of directory with certificates of trusted CAs
        - `True` - default path to truststore will be taken
        - `False` - no verification will be made
    :type verify: bool or str, optional

    :param api_client: Initialized APIClient object with set project or space ID. If passed, ``credentials`` and ``project_id``/``space_id`` are not required.
    :type api_client: APIClient, optional

    :param validate: Model id validation, defaults to True
    :type validate: bool, optional

    :param persistent_connection: Whether to keep persistent connection when evaluating `generate`, `generate_text` or `tokenize` methods.
                                  Note, this parameter is only applicable for mentioned methods when prompt is of type str.
                                  To close connection run `model.close_persistent_connection()`, defaults to True. Added in 1.1.2.
    :type persistent_connection: bool, optional

    .. note::
        One of these parameters is required: [``model_id``, ``deployment_id``]

    .. note::
        One of these parameters is required: [``project_id``, ``space_id``] when ``credentials`` parameter passed.

    .. hint::
        You can copy the project_id from the Project's Manage tab (Project -> Manage -> General -> Details).

    **Example**

    .. code-block:: python

        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
        from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes, DecodingMethods

        # To display example params enter
        GenParams().get_example_values()

        generate_params = {
            GenParams.MAX_NEW_TOKENS: 25
        }

        model_inference = ModelInference(
            model_id=ModelTypes.FLAN_UL2,
            params=generate_params,
            credentials=Credentials(
                                api_key = "***",
                                url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    .. code-block:: python

        from ibm_watsonx_ai.foundation_models import ModelInference
        from ibm_watsonx_ai import Credentials

        deployment_inference = ModelInference(
            deployment_id="<ID of deployed model>",
            credentials=Credentials(
                            api_key = "***",
                            url = "https://us-south.ml.cloud.ibm.com"),
            project_id="*****"
            )

    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        deployment_id: str | None = None,
        params: dict | None = None,
        credentials: dict | Credentials | None = None,
        project_id: str | None = None,
        space_id: str | None = None,
        verify: bool | str | None = None,
        api_client: APIClient | None = None,
        validate: bool = True,
        persistent_connection: bool = True,
    ) -> None:
        self.model_id = model_id
        if isinstance(self.model_id, Enum):
            self.model_id = self.model_id.value

        self.deployment_id = deployment_id

        if self.model_id and self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="Both arguments were provided.",
            )
        elif not self.model_id and not self.deployment_id:
            raise InvalidMultipleArguments(
                params_names_list=["model_id", "deployment_id"],
                reason="None of the arguments were provided.",
            )

        self.params = params
        ModelInference._validate_type(params, "params", dict, False)

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
        if not self._client.CLOUD_PLATFORM_SPACES and self._client.CPD_version < 4.8:
            raise WMLClientError(error_msg="Operation is unsupported for this release.")

        self._inference: BaseModelInference
        if self.model_id:
            self._inference = FMModelInference(
                model_id=self.model_id,
                api_client=self._client,
                params=self.params,
                validate=validate,
                persistent_connection=persistent_connection,
            )
        else:
            self.deployment_id = cast(str, self.deployment_id)
            self._inference = DeploymentModelInference(
                deployment_id=self.deployment_id,
                api_client=self._client,
                params=self.params,
                persistent_connection=persistent_connection,
            )
        BaseModelInference.__init__(self, __name__, self._client)

    def get_details(self) -> dict:
        """Get model interface's details

        :return: details of model or deployment
        :rtype: dict

        **Example**

        .. code-block:: python

            model_inference.get_details()

        """
        return self._inference.get_details()

    @overload
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: Literal[False] = ...,
        validate_prompt_variables: bool = ...,
    ) -> dict | list[dict]: ...

    @overload
    def generate(
        self,
        prompt: str | list | None,
        params: dict | None,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        async_mode: Literal[True],
        validate_prompt_variables: bool,
    ) -> Generator: ...

    @overload
    def generate(
        self,
        prompt: str | list | None = ...,
        params: dict | None = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        async_mode: bool = ...,
        validate_prompt_variables: bool = ...,
    ) -> dict | list[dict] | Generator: ...

    def generate(
        self,
        prompt: str | list | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
        async_mode: bool = False,
        validate_prompt_variables: bool = True,
    ) -> dict | list[dict] | Generator:
        """Given a text prompt as input, and parameters the selected model (model_id) or deployment (deployment_id)
        will generate a completion text as generated_text. For prompt template deployment `prompt` should be None.

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict

        :param concurrency_limit: number of requests that will be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection
                           filter is toggle on for both prompt and generated text, defaults to False
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param async_mode: If True then yield results asynchronously (using generator). In this case both prompt and
                           generated text will be concatenated in the final response - under `generated_text`, defaults
                           to False
        :type async_mode: bool

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool, optional

        :return: scoring result containing generated content
        :rtype: dict

        **Example**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_response = model_inference.generate(prompt=q)
            print(generated_response['results'][0]['generated_text'])

        """
        self._validate_type(params, "params", dict, False)
        self._validate_type(
            concurrency_limit,
            "concurrency_limit",
            [int, float],
            False,
            raise_error_for_list=True,
        )

        if isinstance(concurrency_limit, float):  # convert float (ex. 10.0) to int
            concurrency_limit = int(concurrency_limit)

        if concurrency_limit > 10 or concurrency_limit < 1:
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )

        return self._inference.generate(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            async_mode=async_mode,
            validate_prompt_variables=validate_prompt_variables,
        )

    async def _agenerate(  # type: ignore[override]
        self,
        prompt: str | list | None = None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> AsyncGenerator:
        """
        Given a text prompt as input, and parameters the selected inference
        will return async generator with response.
        """
        self._validate_type(params, "params", dict, False)
        self._validate_type(
            concurrency_limit,
            "concurrency_limit",
            [int, float],
            False,
            raise_error_for_list=True,
        )

        if isinstance(concurrency_limit, float):  # convert float (ex. 10.0) to int
            concurrency_limit = int(concurrency_limit)

        if concurrency_limit > 10 or concurrency_limit < 1:
            raise ParamOutOfRange(
                param_name="concurrency_limit", value=concurrency_limit, min=1, max=10
            )
        return self._inference._agenerate(
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            loop=loop,
        )

    @overload
    def generate_text(
        self,
        prompt: str | None = ...,
        params: dict | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        validate_prompt_variables: bool = ...,
    ) -> str: ...

    @overload
    def generate_text(
        self,
        prompt: list,
        params: dict | None = ...,
        raw_response: Literal[False] = ...,
        guardrails: bool = ...,
        guardrails_hap_params: dict | None = ...,
        guardrails_pii_params: dict | None = ...,
        concurrency_limit: int = ...,
        validate_prompt_variables: bool = ...,
    ) -> list[str]: ...

    @overload
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | None,
        raw_response: Literal[True],
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        validate_prompt_variables: bool,
    ) -> list[dict] | dict: ...

    @overload
    def generate_text(
        self,
        prompt: str | list | None,
        params: dict | None,
        raw_response: bool,
        guardrails: bool,
        guardrails_hap_params: dict | None,
        guardrails_pii_params: dict | None,
        concurrency_limit: int,
        validate_prompt_variables: bool,
    ) -> str | list | dict: ...

    def generate_text(
        self,
        prompt: str | list | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
        validate_prompt_variables: bool = True,
    ) -> str | list | dict:
        """Given a text prompt as input, and parameters the selected model (model_id)
        will generate a completion text as generated_text. For prompt template deployment `prompt` should be None.

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict

        :param concurrency_limit: number of requests that will be sent in parallel, max is 10
        :type concurrency_limit: int

        :param prompt: the prompt string or list of strings. If list of strings is passed requests will be managed in parallel with the rate of concurency_limit, defaults to None
        :type prompt: (str | list | None), optional

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection filter is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param raw_response: return the whole response object
        :type raw_response: bool, optional

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: generated content
        :rtype: str | list | dict

        .. note::
            By default only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)


        **Example**

        .. code-block:: python

            q = "What is 1 + 1?"
            generated_text = model_inference.generate_text(prompt=q)
            print(generated_text)

        """
        metadata = ModelInference.generate(
            self,
            prompt=prompt,
            params=params,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            concurrency_limit=concurrency_limit,
            validate_prompt_variables=validate_prompt_variables,
        )
        if raw_response:
            return metadata
        else:
            if isinstance(prompt, list):
                return [
                    self._return_guardrails_stats(single_response)["generated_text"]
                    for single_response in metadata
                ]
            else:
                return self._return_guardrails_stats(metadata)["generated_text"]  # type: ignore[arg-type]

    def generate_text_stream(
        self,
        prompt: str | None = None,
        params: dict | None = None,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        validate_prompt_variables: bool = True,
    ) -> Generator:
        """Given a text prompt as input, and parameters the selected model (model_id)
        will generate a streamed text as generate_text_stream. For prompt template deployment `prompt` should be None.

        :param params: meta props for text generation, use ``ibm_watsonx_ai.metanames.GenTextParamsMetaNames().show()`` to view the list of MetaNames
        :type params: dict

        :param prompt: the prompt string, defaults to None
        :type prompt: str, optional

        :param raw_response: yields the whole response object
        :type raw_response: bool, optional

        :param guardrails: If True then potentially hateful, abusive, and/or profane language (HAP) detection filter is toggle on for both prompt and generated text, defaults to False
                           If HAP is detected the `HAPDetectionWarning` is issued
        :type guardrails: bool

        :param guardrails_hap_params: meta props for HAP moderations, use ``ibm_watsonx_ai.metanames.GenTextModerationsMetaNames().show()``
                                      to view the list of MetaNames
        :type guardrails_hap_params: dict

        :param validate_prompt_variables: If True, prompt variables provided in `params` are validated with the ones in Prompt Template Asset.
                                          This parameter is only applicable in a Prompt Template Asset deployment scenario and should not be changed for different cases, defaults to True
        :type validate_prompt_variables: bool

        :return: scoring result containing generated content
        :rtype: generator

        .. note::
            By default only the first occurrence of `HAPDetectionWarning` is displayed. To enable printing all warnings of this category, use:

            .. code-block:: python

                import warnings
                from ibm_watsonx_ai.foundation_models.utils import HAPDetectionWarning

                warnings.filterwarnings("always", category=HAPDetectionWarning)

        **Example**

        .. code-block:: python

            q = "Write an epigram about the sun"
            generated_response = model_inference.generate_text_stream(prompt=q)

            for chunk in generated_response:
                print(chunk, end='', flush=True)

        """
        self._validate_type(params, "params", dict, False)

        return self._inference.generate_text_stream(
            prompt=prompt,
            params=params,
            raw_response=raw_response,
            guardrails=guardrails,
            guardrails_hap_params=guardrails_hap_params,
            guardrails_pii_params=guardrails_pii_params,
            validate_prompt_variables=validate_prompt_variables,
        )

    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        """
        The text tokenize operation allows you to check the conversion of provided input to tokens for a given model.
        It splits text into words or sub-words, which then are converted to ids through a look-up table (vocabulary).
        Tokenization allows the model to have a reasonable vocabulary size.

        .. note::
            Method is not supported for deployments, available only for base models.

        :param prompt: the prompt string, defaults to None
        :type prompt: str, optional

        :param return_tokens: the parameter for text tokenization, defaults to False
        :type return_tokens: bool

        :return: the result of tokenizing the input string.
        :rtype: dict

        **Example**

        .. code-block:: python

            q = "Write an epigram about the moon"
            tokenized_response = model_inference.tokenize(prompt=q, return_tokens=True)
            print(tokenized_response["result"])

        """
        return self._inference.tokenize(prompt=prompt, return_tokens=return_tokens)

    def to_langchain(self) -> WatsonxLLM:
        """

        :return: WatsonxLLM wrapper for watsonx foundation models
        :rtype: WatsonxLLM

        **Example**

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            flan_ul2_model = ModelInference(
                model_id=ModelTypes.FLAN_UL2,
                credentials=Credentials(
                                    api_key = "***",
                                    url = "https://us-south.ml.cloud.ibm.com"),
                project_id="*****"
                )

            prompt_template = "What color is the {flower}?"

            llm_chain = LLMChain(llm=flan_ul2_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
            llm_chain('sunflower')

        .. code-block:: python

            from langchain import PromptTemplate
            from langchain.chains import LLMChain
            from ibm_watsonx_ai import Credentials
            from ibm_watsonx_ai.foundation_models import ModelInference
            from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes

            deployed_model = ModelInference(
                deployment_id="<ID of deployed model>",
                credentials=Credentials(
                                    api_key = "***",
                                    url = "https://us-south.ml.cloud.ibm.com"),
                space_id="*****"
                )

            prompt_template = "What color is the {car}?"

            llm_chain = LLMChain(llm=deployed_model.to_langchain(), prompt=PromptTemplate.from_template(prompt_template))
            llm_chain('sunflower')

        """
        try:
            from langchain_ibm import WatsonxLLM
        except ImportError:
            raise MissingExtension("langchain_ibm")
        return WatsonxLLM(watsonx_model=self)

    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        return self._inference.get_identifying_params()

    def close_persistent_connection(self) -> None:
        """Only applicable if persistent_connection was set to True in ModelInference initialization."""
        if (
            self._inference._persistent_connection
            and self._inference._session is not None
        ):
            self._inference._session.close()
