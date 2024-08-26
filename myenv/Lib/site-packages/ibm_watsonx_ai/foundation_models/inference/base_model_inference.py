#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import time
import json
import warnings
import asyncio
import threading

from functools import partial
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generator, AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

from ibm_watsonx_ai.foundation_models.utils.utils import (
    HAPDetectionWarning,
    PIIDetectionWarning,
    generate_event_loop,
    _set_session_default_params,
    _get_requests_session,
)
import ibm_watsonx_ai._wrappers.requests as requests
from ibm_watsonx_ai.wml_resource import WMLResource
from ibm_watsonx_ai.wml_client_error import WMLClientError

if TYPE_CHECKING:
    from ibm_watsonx_ai import APIClient
    from httpx import AsyncClient

__all__ = ["BaseModelInference"]


class BaseModelInference(WMLResource, ABC):
    """Base interface class for the model interface."""

    # thread local storage definition
    _thread_local = threading.local()

    def __init__(
        self, name: str, client: APIClient, persistent_connection: bool = True
    ):
        self._persistent_connection = persistent_connection
        self.__session: requests.Session | None = None
        WMLResource.__init__(self, name, client)

    @property
    def _session(self) -> requests.Session:
        if self.__session is None:
            self.__session = _set_session_default_params(requests.Session())
        return self.__session

    @abstractmethod
    def get_details(self) -> dict:
        """Get model interface's details

        :return: details of model or deployment
        :rtype: dict
        """
        raise NotImplementedError

    @abstractmethod
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
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generated_text response.
        """
        raise NotImplementedError

    @abstractmethod
    def _agenerate(
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
        raise NotImplementedError

    @abstractmethod
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
        """
        Given a text prompt as input, and parameters the selected inference
        will generate a completion text as generator.
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize(self, prompt: str, return_tokens: bool = False) -> dict:
        raise NotImplementedError

    @abstractmethod
    def get_identifying_params(self) -> dict:
        """Represent Model Inference's setup in dictionary"""
        raise NotImplementedError

    def _prepare_inference_payload(
        self,
        prompt: str | None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _prepare_beta_inference_payload(
        self,
        prompt: str | None,
        params: dict | None = None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        raise NotImplementedError

    def _send_inference_payload(
        self,
        prompt: str | None,
        params: dict | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        _session: requests.Session | None = None,
    ) -> dict:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        retries = 0
        # use separate session for each thread when user provide list of prompts
        session = (
            _session
            if _session is not None
            else _get_requests_session(BaseModelInference._thread_local)
        )
        while retries < 3:
            with session.post(
                url=generate_url,
                json=payload,
                params=self._client._params(skip_for_create=True, skip_userfs=True),
                headers=self._client._get_headers(),
            ) as response_scoring:
                if response_scoring.status_code in [429, 503, 504, 520]:
                    time.sleep(2**retries)
                    retries += 1
                else:
                    break

        return self._handle_response(
            200,
            "generate",
            response_scoring,
            _field_to_hide="generated_text",
        )

    async def _asend_inference_payload(
        self,
        prompt: str | None,
        params: dict,
        generate_url: str,
        client: AsyncClient,
        semaphore: asyncio.Semaphore,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> dict:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        async with semaphore:
            response = await client.post(
                url=generate_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
            )

            return self._handle_response(200, "agenerate", response)

    def _generate_with_url(
        self,
        prompt: list | str | None,
        params: dict | None,
        generate_url: str,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
    ) -> list | dict:
        """
        Helper method which implements multi-threading for with passed generate_url.
        """
        try:
            if isinstance(prompt, list):
                _send_inference_payload_partial = partial(
                    self._send_inference_payload,
                    params=params,
                    generate_url=generate_url,
                    guardrails=guardrails,
                    guardrails_hap_params=guardrails_hap_params,
                    guardrails_pii_params=guardrails_pii_params,
                )
                if (prompt_length := len(prompt)) <= concurrency_limit:
                    with ThreadPoolExecutor(max_workers=prompt_length) as executor:
                        generated_responses = list(
                            executor.map(_send_inference_payload_partial, prompt)
                        )
                else:
                    with ThreadPoolExecutor(max_workers=concurrency_limit) as executor:
                        generated_responses = list(
                            executor.map(_send_inference_payload_partial, prompt)
                        )
                return generated_responses

            else:

                response = self._send_inference_payload(
                    prompt,
                    params,
                    generate_url,
                    guardrails,
                    guardrails_hap_params,
                    guardrails_pii_params,
                    _session=self._session,
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
        return response

    def _generate_stream_with_url(
        self,
        prompt: str | None,
        params: dict | None,
        generate_stream_url: str,
        raw_response: bool = False,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
    ) -> Generator:
        if self._client._use_fm_ga_api:
            payload = self._prepare_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )
        else:  # Remove on CPD 5.0 release
            payload = self._prepare_beta_inference_payload(
                prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
            )

        s = requests.Session()
        retries = 0
        while retries < 3:
            with s.post(
                url=generate_stream_url,
                json=payload,
                headers=self._client._get_headers(),
                params=self._client._params(skip_for_create=True, skip_userfs=True),
                stream=True,
            ) as resp:
                if resp.status_code in [429, 503, 504, 520]:
                    time.sleep(2**retries)
                    retries += 1
                elif resp.status_code == 200:
                    for chunk in resp.iter_lines(decode_unicode=False):
                        chunk = chunk.decode("utf-8")
                        if "generated_text" in chunk:
                            response = chunk.replace("data: ", "")
                            try:
                                parsed_response = json.loads(response)
                            except json.JSONDecodeError:
                                raise Exception(f"Could not parse {response} as json")
                            if raw_response:
                                yield parsed_response
                                continue
                            yield self._return_guardrails_stats(parsed_response)[
                                "generated_text"
                            ]
                    break

        if resp.status_code != 200:
            raise WMLClientError(
                f"Request failed with: {resp.text} ({resp.status_code})"
            )

    def _tokenize_with_url(
        self,
        prompt: str,
        tokenize_url: str,
        return_tokens: bool,
    ) -> dict:
        payload = self._prepare_inference_payload(prompt)

        parameters = payload.get("parameters", {})
        parameters.update({"return_tokens": return_tokens})
        payload["parameters"] = parameters

        try:
            retries = 0
            while retries < 3:
                response_scoring = self._session.post(
                    url=tokenize_url,
                    json=payload,
                    params=self._client._params(skip_for_create=True, skip_userfs=True),
                    headers=self._client._get_headers(),
                )
                if response_scoring.status_code in [429, 503, 504, 520]:
                    time.sleep(2**retries)
                    retries += 1
                elif response_scoring.status_code == 404:
                    raise WMLClientError("Tokenize is not supported for this release")
                else:
                    break
            response = self._handle_response(200, "tokenize", response_scoring)
        except Exception:
            # closing the connection if an error occurs and if persistent connection is true
            if self._persistent_connection and self._session is not None:
                self._session.close()
            raise
        finally:
            # closing the connection if an error occurs or if persistent connection is false
            if not self._persistent_connection and self._session is not None:
                self._session.close()

        return response

    def _return_guardrails_stats(self, single_response: dict) -> dict:
        results = single_response["results"][0]
        hap_details = (
            results.get("moderations", {}).get("hap")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("hap")
        )  # Remove 'else' on CPD 5.0 release
        if hap_details:
            if hap_details[0].get("input"):
                warnings.warn(
                    next(
                        warning.get("message")
                        for warning in single_response.get("system", {}).get("warnings")
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    category=HAPDetectionWarning,
                )
            else:
                warnings.warn(
                    f"Potentially harmful text detected: {hap_details}",
                    category=HAPDetectionWarning,
                )
        pii_details = (
            results.get("moderations", {}).get("pii")
            if self._client._use_fm_ga_api
            else results.get("moderation", {}).get("pii")
        )  # Remove 'else' on CPD 5.0 release
        if pii_details:
            if pii_details[0].get("input"):
                warnings.warn(
                    next(
                        warning.get("message")
                        for warning in single_response.get("system", {}).get("warnings")
                        if warning.get("id") == "UNSUITABLE_INPUT"
                    ),
                    category=PIIDetectionWarning,
                )
            else:
                warnings.warn(
                    f"Personally identifiable information detected: {pii_details}",
                    category=PIIDetectionWarning,
                )
        return results

    @staticmethod
    def _update_moderations_params(additional_params: dict) -> dict:
        default_params = {"input": {"enabled": True}, "output": {"enabled": True}}
        if additional_params:
            for key, value in default_params.items():
                if key in additional_params:
                    if additional_params[key]:
                        if "threshold" in additional_params:
                            default_params[key]["threshold"] = additional_params[
                                "threshold"
                            ]
                    else:
                        default_params[key]["enabled"] = False
                else:
                    if "threshold" in additional_params:
                        default_params[key]["threshold"] = additional_params[
                            "threshold"
                        ]
            if "mask" in additional_params:
                default_params.update({"mask": additional_params["mask"]})
        return default_params

    def _get_as_generator(
        self,
        prompt: str | list | None,
        params: dict | None,
        guardrails: bool = False,
        guardrails_hap_params: dict | None = None,
        guardrails_pii_params: dict | None = None,
        concurrency_limit: int = 10,
    ) -> Generator:
        with generate_event_loop as loop:
            f = self._agenerate(
                prompt=prompt,
                params=params,
                guardrails=guardrails,
                guardrails_hap_params=guardrails_hap_params,
                guardrails_pii_params=guardrails_pii_params,
                concurrency_limit=concurrency_limit,
                loop=loop,
            )
            for _ in range(len(prompt) if isinstance(prompt, list) else 1):
                try:
                    future = asyncio.run_coroutine_threadsafe(f.__anext__(), loop)
                    yield future.result()
                    generate_event_loop._future = future
                except StopAsyncIteration:
                    break
