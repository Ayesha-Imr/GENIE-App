#  -----------------------------------------------------------------------------------------
#  (C) Copyright IBM Corp. 2023-2024.
#  https://opensource.org/licenses/BSD-3-Clause
#  -----------------------------------------------------------------------------------------
from __future__ import annotations

import os
import requests
import httpx
import json as js
import asyncio
from requests import packages, exceptions
from functools import wraps
from typing import Any, Coroutine


verify = os.environ.get("WX_CLIENT_VERIFY_REQUESTS")
additional_settings = {}


def set_verify_for_requests(func):
    @wraps(func)
    def wrapper(*args, **kw):
        global verify

        if verify is not None:
            if verify == "True":
                kw.update({"verify": True})

            elif verify == "False":
                kw.update({"verify": False})

            else:
                kw.update({"verify": verify})

        else:
            kw.update({"verify": True})

        try:
            res = func(*args, **kw)

        except OSError as e:

            # User can pass verify the path to a CA_BUNDLE file or directory with certificates of trusted CAs
            if isinstance(verify, str) and verify != "False":
                raise OSError(
                    f"Connection cannot be verified with default trusted CAs. "
                    f"Please provide correct path to a CA_BUNDLE file or directory with "
                    f"certificates of trusted CAs. Error: {e}"
                )

            # forced verify to True
            elif verify:
                raise e

            # default
            elif verify is None:
                verify = "False"
                kw.update({"verify": False})
                res = func(*args, **kw)

            # disabled verify
            else:
                raise e

        return res

    return wrapper


def set_additional_settings_for_requests(func):
    @wraps(func)
    def wrapper(*args, **kw):
        kwargs = {}
        kwargs.update(additional_settings)
        kwargs.update(kw)
        return func(*args, **kwargs)

    return wrapper


@set_verify_for_requests
@set_additional_settings_for_requests
def get(url, params=None, **kwargs):
    r"""Sends a GET request.

    :param url: URL for the new :class:`Request` object.
    :param params: (optional) Dictionary, list of tuples or bytes to send
        in the query string for the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return requests.get(url=url, params=params, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def options(url, **kwargs):
    r"""Sends an OPTIONS request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return requests.options(url=url, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def head(url, **kwargs):
    r"""Sends a HEAD request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes. If
        `allow_redirects` is not provided, it will be set to `False` (as
        opposed to the default :meth:`request` behavior).
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return requests.head(url=url, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def post(url, data=None, json=None, **kwargs):
    r"""Sends a POST request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) json data to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """
    from ibm_watsonx_ai.utils.utils import _requests_convert_json_to_data

    data, json, kwargs = _requests_convert_json_to_data(data, json, kwargs)

    return requests.post(url=url, data=data, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def put(url, data=None, **kwargs):
    r"""Sends a PUT request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) json data to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    from ibm_watsonx_ai.utils.utils import _requests_convert_json_to_data

    data, json, kwargs = _requests_convert_json_to_data(
        data, kwargs.get("json"), kwargs
    )

    return requests.put(url=url, data=data, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def patch(url, data=None, **kwargs):
    r"""Sends a PATCH request.

    :param url: URL for the new :class:`Request` object.
    :param data: (optional) Dictionary, list of tuples, bytes, or file-like
        object to send in the body of the :class:`Request`.
    :param json: (optional) json data to send in the body of the :class:`Request`.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """
    from ibm_watsonx_ai.utils.utils import _requests_convert_json_to_data

    data, json, kwargs = _requests_convert_json_to_data(
        data, kwargs.get("json"), kwargs
    )

    return requests.patch(url=url, data=data, **kwargs)


@set_verify_for_requests
@set_additional_settings_for_requests
def delete(url, **kwargs):
    r"""Sends a DELETE request.

    :param url: URL for the new :class:`Request` object.
    :param \*\*kwargs: Optional arguments that ``request`` takes.
    :return: :class:`Response <Response>` object
    :rtype: requests.Response
    """

    return requests.delete(url=url, **kwargs)


class Session(requests.Session):
    """A Requests session.

    Provides cookie persistence, connection-pooling, and configuration.

    Basic Usage::

      >>> import requests
      >>> s = requests.Session()
      >>> s.get('https://httpbin.org/get')
      <Response [200]>

    Or as a context manager::

      >>> with requests.Session() as s:
      ...     s.get('https://httpbin.org/get')
      <Response [200]>
    """

    def __init__(self):
        requests.Session.__init__(self)

    @set_verify_for_requests
    @set_additional_settings_for_requests
    def request(self, method, url, **kwargs):
        """Constructs a :class:`Request <Request>`, prepares it and sends it.
        Returns :class:`Response <Response>` object.

        :param method: method for the new :class:`Request` object.
        :param url: URL for the new :class:`Request` object.
        :param params: (optional) Dictionary or bytes to be sent in the query
            string for the :class:`Request`.
        :param data: (optional) Dictionary, list of tuples, bytes, or file-like
            object to send in the body of the :class:`Request`.
        :param json: (optional) json to send in the body of the
            :class:`Request`.
        :param headers: (optional) Dictionary of HTTP Headers to send with the
            :class:`Request`.
        :param cookies: (optional) Dict or CookieJar object to send with the
            :class:`Request`.
        :param files: (optional) Dictionary of ``'filename': file-like-objects``
            for multipart encoding upload.
        :param auth: (optional) Auth tuple or callable to enable
            Basic/Digest/Custom HTTP Auth.
        :param timeout: (optional) How long to wait for the server to send
            data before giving up, as a float, or a :ref:`(connect timeout,
            read timeout) <timeouts>` tuple.
        :type timeout: float or tuple
        :param allow_redirects: (optional) Set to True by default.
        :type allow_redirects: bool
        :param proxies: (optional) Dictionary mapping protocol or protocol and
            hostname to the URL of the proxy.
        :param stream: (optional) whether to immediately download the response
            content. Defaults to ``False``.
        :param verify: (optional) Either a boolean, in which case it controls whether we verify
            the server's TLS certificate, or a string, in which case it must be a path
            to a CA bundle to use. Defaults to ``True``. When set to
            ``False``, requests will accept any TLS certificate presented by
            the server, and will ignore hostname mismatches and/or expired
            certificates, which will make your application vulnerable to
            man-in-the-middle (MitM) attacks. Setting verify to ``False``
            may be useful during local development or testing.
        :param cert: (optional) if String, path to ssl client cert file (.pem).
            If Tuple, ('cert', 'key') pair.
        :rtype: requests.Response
        """

        kwargs["method"] = method
        kwargs["url"] = url
        from ibm_watsonx_ai.utils.utils import _requests_convert_json_to_data

        data, json, kwargs = _requests_convert_json_to_data(
            kwargs.get("data"), kwargs.get("json"), kwargs
        )

        return super().request(**{**kwargs, **{"data": data}})


def session():
    """
    Returns a :class:`Session` for context-management.

    .. deprecated:: 1.0.0

        This method has been deprecated since version 1.0.0 and is only kept for
        backwards compatibility. New code should use :class:`~requests.sessions.Session`
        to create a session. This may be removed at a future date.

    :rtype: Session
    """
    return Session()


class AsyncClient(httpx.AsyncClient):
    def __init__(self, verify: bool | None = None, **kwargs: Any):
        super().__init__(
            verify=verify, timeout=httpx.Timeout(timeout=600.0, connect=5.0), **kwargs
        )

    async def post(
        self,
        url: str,
        json: dict | None = None,
        data: dict | None = None,
        headers: dict | None = None,
        params: dict | None = None,
        **kwargs: Any,
    ) -> Coroutine:

        if json is not None and data is None:
            from ibm_watsonx_ai.utils.utils import NumpyTypeEncoder

            data = js.dumps(json, cls=NumpyTypeEncoder)

            if headers and not headers.get("Content-Type"):
                headers["Content-Type"] = "application/json"

        retries = 0
        while retries < 3:
            timeout = False
            try:
                response = await super().post(
                    url=url,
                    json=json,
                    data=data,
                    headers=headers,
                    params=params,
                    **kwargs,
                )
            except httpx.TimeoutException:
                timeout = True

            if timeout or response.status_code in [429, 503, 504, 520]:
                await asyncio.sleep(2**retries)
                retries += 1
            else:
                break

        return response


@set_verify_for_requests
@set_additional_settings_for_requests
def get_async_client(**kwargs: Any) -> AsyncClient:
    return AsyncClient(**kwargs)
