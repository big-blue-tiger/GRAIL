import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI

DEFAULT_TEXT_MODEL = "qwen3.7-plus"
DEFAULT_CHAT_MODEL = DEFAULT_TEXT_MODEL
DEFAULT_VISION_MODEL = "qwen3.5-omni-plus-2026-03-15"
DEFAULT_REASONING_MODEL = DEFAULT_VISION_MODEL
DEFAULT_IMAGE_MODEL = "wan2.2-t2i-plus"

DEFAULT_REGION = "cn-beijing"

_COMPATIBLE_PATH = "/compatible-mode/v1"
_TEXT2IMAGE_PATH = "/api/v1/services/aigc/text2image/image-synthesis"
_TASK_QUERY_PATH = "/api/v1/tasks"

_API_KEY_ENV_VARS = (
    "DASHSCOPE_API_KEY",
    "QWEN_API_KEY",
    "BAILIAN_API_KEY",
    "ALIYUN_API_KEY",
)
_WORKSPACE_ENV_VARS = (
    "DASHSCOPE_WORKSPACE_ID",
    "BAILIAN_WORKSPACE_ID",
    "QWEN_WORKSPACE_ID",
    "ALIYUN_WORKSPACE_ID",
)


def _first_env(names: tuple[str, ...]) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    return None


def _get_api_key() -> str | None:
    return _first_env(_API_KEY_ENV_VARS)


def _ensure_qwen_key_present() -> None:
    if not _get_api_key():
        names = ", ".join(_API_KEY_ENV_VARS)
        raise OSError(f"Qwen/DashScope API key is not set. Export one of: {names}.")


def _get_region() -> str:
    return os.getenv("DASHSCOPE_REGION") or os.getenv("BAILIAN_REGION") or DEFAULT_REGION


def _strip_known_api_suffix(base_url: str) -> str:
    url = base_url.rstrip("/")
    for suffix in (_COMPATIBLE_PATH, "/api/v1"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _service_root_url() -> str:
    explicit = os.getenv("DASHSCOPE_SERVICE_BASE_URL") or os.getenv(
        "BAILIAN_SERVICE_BASE_URL"
    )
    if explicit:
        return _strip_known_api_suffix(explicit)

    generic = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("BAILIAN_BASE_URL")
    if generic:
        return _strip_known_api_suffix(generic)

    workspace_id = _first_env(_WORKSPACE_ENV_VARS)
    if workspace_id:
        return f"https://{workspace_id}.{_get_region()}.maas.aliyuncs.com"

    return "https://dashscope.aliyuncs.com"


def _compatible_base_url() -> str:
    explicit = os.getenv("DASHSCOPE_COMPATIBLE_BASE_URL") or os.getenv(
        "BAILIAN_COMPATIBLE_BASE_URL"
    )
    if explicit:
        url = explicit.rstrip("/")
        return url if url.endswith(_COMPATIBLE_PATH) else f"{url}{_COMPATIBLE_PATH}"

    generic = os.getenv("DASHSCOPE_BASE_URL") or os.getenv("BAILIAN_BASE_URL")
    if generic:
        url = generic.rstrip("/")
        return url if url.endswith(_COMPATIBLE_PATH) else f"{url}{_COMPATIBLE_PATH}"

    return f"{_service_root_url()}{_COMPATIBLE_PATH}"


def _make_client() -> OpenAI:
    _ensure_qwen_key_present()
    return OpenAI(api_key=_get_api_key(), base_url=_compatible_base_url())


def _encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _image_path_to_data_uri(image_path: str) -> str:
    mime_type, _ = mimetypes.guess_type(image_path)
    if mime_type is None:
        mime_type = "image/png"
    b64 = _encode_image_to_base64(image_path)
    return f"data:{mime_type};base64,{b64}"


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _extract_response_text(response: Any) -> str:
    output_text = _get_value(response, "output_text")
    if isinstance(output_text, str) and output_text:
        return output_text.strip()

    texts: list[str] = []
    for item in _get_value(response, "output", []) or []:
        content = _get_value(item, "content", []) or []
        if isinstance(content, str):
            texts.append(content)
            continue
        for part in content:
            text = _get_value(part, "text")
            if isinstance(text, str):
                texts.append(text)

    if texts:
        return "".join(texts).strip()

    error = _get_value(response, "error")
    if error:
        raise RuntimeError(f"Qwen response failed: {error}")
    return ""


def _response_text_config(response_format: dict[str, Any] | None) -> dict[str, Any] | None:
    if response_format is None:
        return None

    # OpenAI Responses uses text.format while older call sites pass
    # Chat Completions-style response_format. Preserve that call style here.
    return {"format": response_format}


def chat_text(
    prompt_text: str,
    *,
    model: str = DEFAULT_TEXT_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = None,
    response_format: dict[str, Any] | None = None,
    previous_response_id: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: str | dict[str, Any] | None = None,
) -> str:
    """Plain text chat via Alibaba Bailian's OpenAI-compatible Responses API."""
    client = _make_client()

    create_kwargs: dict[str, Any] = {
        "model": model,
        "input": prompt_text,
        "max_output_tokens": max_tokens,
        "temperature": temperature,
    }
    if system_prompt:
        create_kwargs["instructions"] = system_prompt
    if previous_response_id:
        create_kwargs["previous_response_id"] = previous_response_id
    if tools is not None:
        create_kwargs["tools"] = tools
    if tool_choice is not None:
        create_kwargs["tool_choice"] = tool_choice

    text_config = _response_text_config(response_format)
    if text_config is not None:
        create_kwargs["text"] = text_config

    response = client.responses.create(**create_kwargs)
    return _extract_response_text(response)


def chat_with_image(
    prompt_text: str,
    image_path: str,
    *,
    model: str = DEFAULT_VISION_MODEL,
    max_tokens: int = 512,
    temperature: float = 0.7,
    system_prompt: str | None = "You are a helpful vision assistant.",
    response_format: dict[str, Any] | None = None,
    previous_response_id: str | None = None,
) -> str:
    """Vision chat via Qwen Omni using the OpenAI-compatible Chat Completions API.

    Note: Qwen-Omni requires stream=True, so this function collects the streamed response.
    """
    client = _make_client()

    data_uri = _image_path_to_data_uri(image_path)

    # Build messages array
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": data_uri}},
        ],
    })

    create_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "modalities": ["text"],  # Only text output for vision tasks
        "stream": True,  # Required by Qwen-Omni
        "stream_options": {"include_usage": True},
    }

    if response_format is not None:
        create_kwargs["response_format"] = response_format

    # Stream the response and collect text
    completion = client.chat.completions.create(**create_kwargs)

    text_parts: list[str] = []
    for chunk in completion:
        if chunk.choices and chunk.choices[0].delta.content:
            text_parts.append(chunk.choices[0].delta.content)

    return "".join(text_parts).strip()


def _normalize_size(size: str) -> str:
    return size.replace("x", "*")


def _http_json(url: str, payload: dict[str, Any], *, timeout: float, extra_headers: dict[str, str] | None = None) -> dict[str, Any]:
    api_key = _get_api_key()
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    request = Request(url, data=body, headers=headers, method="POST")

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen image request failed: HTTP {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"Qwen image request failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen image request returned non-JSON response: {raw[:500]}") from exc

    if data.get("code") or data.get("message"):
        raise RuntimeError(f"Qwen image request failed: {data}")
    return data


def _http_get(url: str, *, timeout: float) -> dict[str, Any]:
    """GET request for task status queries."""
    api_key = _get_api_key()
    request = Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )

    try:
        with urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen task query failed: HTTP {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"Qwen task query failed: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Qwen task query returned non-JSON response: {raw[:500]}") from exc

    return data


def _download_url(url: str, output_path: Path, *, timeout: float) -> None:
    request = Request(url, headers={"User-Agent": "grail-qwen-api/0.1"})
    try:
        with urlopen(request, timeout=timeout) as response:
            output_path.write_bytes(response.read())
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Qwen image download failed: HTTP {exc.code}: {raw}") from exc
    except URLError as exc:
        raise RuntimeError(f"Qwen image download failed: {exc}") from exc


def generate_image(
    prompt: str,
    output_path: Path | str,
    *,
    model: str = DEFAULT_IMAGE_MODEL,
    size: str = "512x512",
    system_prompt: str | None = None,
    negative_prompt: str | None = None,
    prompt_extend: bool = True,
    watermark: bool = False,
    seed: int | None = None,
    timeout: float = 180.0,
    n: int = 1,
    poll_interval: float = 10.0,
) -> Path:
    """Generate an image with Bailian's text2image API (wan2.2/wan2.5) and save it.

    For wan2.2 and wan2.5 models, uses async HTTP API with task polling.

    Args:
        prompt: The prompt describing the desired image
        output_path: Where to save the generated image
        model: Model to use (e.g., "wan2.2-t2i-plus", "wan2.5-t2i-preview")
        size: Image size in "WIDTHxHEIGHT" or "WIDTH*HEIGHT" format
        system_prompt: Optional system prompt (prepended to prompt)
        negative_prompt: Optional negative prompt for wan2.5
        prompt_extend: Whether to enable intelligent prompt rewriting
        watermark: Whether to add "AI生成" watermark
        seed: Optional random seed for reproducibility
        timeout: Total timeout for the entire operation
        n: Number of images to generate (1-4, only the first is downloaded)
        poll_interval: Seconds to wait between status checks

    Returns:
        Path object pointing to the saved image
    """
    _ensure_qwen_key_present()

    if system_prompt:
        prompt = f"{system_prompt.strip()}\n\n{prompt}"

    # Build parameters
    parameters: dict[str, Any] = {
        "size": _normalize_size(size),
        "n": n,
        "prompt_extend": prompt_extend,
        "watermark": watermark,
    }
    if seed is not None:
        parameters["seed"] = seed

    # Step 1: Create async task
    # wan2.5 supports negative_prompt in input, wan2.2 and below in parameters
    input_payload: dict[str, Any] = {"prompt": prompt}
    if negative_prompt is not None and model.startswith("wan2.5"):
        input_payload["negative_prompt"] = negative_prompt
    elif negative_prompt is not None:
        parameters["negative_prompt"] = negative_prompt

    payload = {
        "model": model,
        "input": input_payload,
        "parameters": parameters,
    }

    url = f"{_service_root_url()}{_TEXT2IMAGE_PATH}"

    # Must include X-DashScope-Async: enable header for async mode
    start_time = time.time()
    data = _http_json(url, payload, timeout=timeout, extra_headers={"X-DashScope-Async": "enable"})

    # Extract task_id from response
    output = data.get("output") or {}
    task_id = output.get("task_id")
    if not task_id:
        raise RuntimeError(f"No task_id returned by model {model!r}: {data}")

    task_status = output.get("task_status")
    print(f"Task created: {task_id}, status: {task_status}")

    # Step 2: Poll for task completion
    task_url = f"{_service_root_url()}{_TASK_QUERY_PATH}/{task_id}"

    while True:
        elapsed = time.time() - start_time
        if elapsed >= timeout:
            raise RuntimeError(f"Task {task_id} timed out after {timeout}s")

        # Wait before polling (skip on first iteration if already SUCCEEDED)
        if task_status not in ("SUCCEEDED", "FAILED", "CANCELED"):
            time.sleep(min(poll_interval, timeout - elapsed))

        # Query task status
        remaining_timeout = timeout - (time.time() - start_time)
        if remaining_timeout <= 0:
            raise RuntimeError(f"Task {task_id} timed out after {timeout}s")

        task_data = _http_get(task_url, timeout=min(30.0, remaining_timeout))

        output = task_data.get("output") or {}
        task_status = output.get("task_status")

        if task_status == "SUCCEEDED":
            print(f"Task {task_id} succeeded")
            break
        elif task_status == "FAILED":
            code = output.get("code") or task_data.get("code")
            message = output.get("message") or task_data.get("message")
            raise RuntimeError(f"Task {task_id} failed: {code} - {message}")
        elif task_status == "CANCELED":
            raise RuntimeError(f"Task {task_id} was canceled")
        elif task_status == "UNKNOWN":
            raise RuntimeError(f"Task {task_id} is unknown or expired")
        elif task_status in ("PENDING", "RUNNING"):
            print(f"Task {task_id} status: {task_status}, waiting...")
            continue
        else:
            raise RuntimeError(f"Task {task_id} has unexpected status: {task_status}")

    # Extract image URL from results
    results = output.get("results") or []
    if not results:
        raise RuntimeError(f"No results returned for task {task_id}: {task_data}")

    image_url = results[0].get("url")
    if not image_url:
        raise RuntimeError(f"No image URL in results for task {task_id}: {results[0]}")

    # Download and save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    remaining_timeout = timeout - (time.time() - start_time)
    _download_url(image_url, output_path, timeout=max(10.0, remaining_timeout))

    print(f"Image saved to {output_path}")
    return output_path
