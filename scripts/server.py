# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp==1.29.0",
# ]
# ///

"""Stdio MCP server for OpenAI-compatible image and video generation APIs."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp import types
from mcp.server.fastmcp import FastMCP
import aicopy as _aicopy
from aicopy import find_url as _aicopy_find_url
from aicopy import generate_image as _aicopy_generate_image
from aicopy import generate as _aicopy_generate
from aicopy import image_pricing_options as _aicopy_image_pricing_options
from aicopy import is_aicopy_profile
from aicopy import pricing_options as _aicopy_pricing_options


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = PLUGIN_ROOT / "config.json"
DEFAULT_IMAGE_OUTPUT_DIR = Path.home() / "Pictures" / "Codex Generated Images"
DEFAULT_VIDEO_OUTPUT_DIR = Path.home() / "Videos" / "Codex Generated Videos"
mcp = FastMCP("OpenAI Media")


def _read_config() -> dict[str, Any]:
    if not CONFIG_FILE.exists():
        return {}
    try:
        config = json.loads(CONFIG_FILE.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{CONFIG_FILE} is not valid JSON: {error.msg} (line {error.lineno}).") from error
    if not isinstance(config, dict):
        raise ValueError(f"{CONFIG_FILE} must contain a JSON object.")
    return config


def _string(value: object, default: str = "") -> str:
    return value.strip() if isinstance(value, str) else default


def _object(value: object, field_name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object.")
    return dict(value)


def _configured_endpoint(base_url: str, endpoint: str) -> str:
    endpoint = endpoint.strip()
    if not endpoint:
        raise ValueError("The configured generation endpoint is empty.")
    if urllib.parse.urlparse(endpoint).scheme in {"http", "https"}:
        return endpoint
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("The configured base_url is empty.")
    # Allow a profile base_url to include the operation path already.
    if base_url.endswith(f"/{endpoint.lstrip('/')}"):
        return base_url
    return f"{base_url}/{endpoint.lstrip('/')}"


def _error_message(error: urllib.error.HTTPError, media_type: str) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8", errors="replace"))
        detail = payload.get("error", payload) if isinstance(payload, dict) else payload
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or "request rejected"
    except Exception:
        detail = "request rejected"
    return f"{media_type.title()} API returned HTTP {error.code}: {str(detail)[:500]}"


def _download(url: str, media_type: str, api_key: str = "", base_url: str = "") -> tuple[bytes, str]:
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError(f"The {media_type} API returned an unsupported media URL.")
    headers: dict[str, str] = {}
    target_host = urllib.parse.urlparse(url).netloc
    base_host = urllib.parse.urlparse(base_url).netloc
    if api_key and target_host and target_host == base_host:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read(), response.headers.get_content_type()


def _extension(mime_type: str, media_type: str = "image") -> str:
    defaults = {"image": "png", "video": "mp4"}
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
        "image/gif": "gif",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/quicktime": "mov",
    }.get(mime_type.lower(), defaults.get(media_type, "bin"))


def _profile_matches(profile: object, operation: str) -> bool:
    if not isinstance(profile, dict):
        return False
    profile_type = _string(profile.get("type"))
    if operation == "video":
        return profile_type == "video"
    return profile_type in {"image", operation}


def _profile_names(config: dict[str, Any], operation: str) -> list[str]:
    profiles = config.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError(f"{CONFIG_FILE} field profiles must be a JSON object.")
    return sorted(
        name
        for name, profile in profiles.items()
        if isinstance(name, str) and _profile_matches(profile, operation)
    )


def _resolve_profile(
    operation: str,
    requested_name: str,
    config: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    profiles = _object(config.get("profiles"), "profiles")
    requested_name = requested_name.strip()
    available = _profile_names(config, operation)

    if requested_name:
        profile = profiles.get(requested_name)
        if not _profile_matches(profile, operation):
            choices = ", ".join(available) or "none"
            raise ValueError(f"Unknown {operation} profile {requested_name!r}. Available profiles: {choices}.")
        return requested_name, dict(profile)

    defaults = _object(config.get("defaults"), "defaults")
    default_name = _string(defaults.get(operation))
    if not default_name and operation in {"image_generate", "image_edit"}:
        default_name = _string(defaults.get("image"))
    if default_name:
        return _resolve_profile(operation, default_name, config)
    if len(available) == 1:
        return _resolve_profile(operation, available[0], config)
    if available:
        raise ValueError(
            f"More than one {operation} profile is configured. Set defaults.{operation} in {CONFIG_FILE} "
            f"or pass profile explicitly. Available profiles: {', '.join(available)}."
        )
    raise ValueError(
        f"No {operation} profile is configured. Copy config.example.json to {CONFIG_FILE.name}, then set its key."
    )


def _api_key(profile: dict[str, Any], profile_name: str) -> str:
    api_key = _string(profile.get("key"))
    if not api_key:
        raise ValueError(f"Set key for profile {profile_name!r} in {CONFIG_FILE}.")
    return api_key


def _model(profile: dict[str, Any], supplied_model: str) -> str:
    model = supplied_model.strip() or _string(profile.get("model")) or _string(profile.get("model_name"))
    if not model:
        raise ValueError("No model was supplied and the selected profile has no model or model_name setting.")
    return model


def _payload(
    prompt: str,
    model: str,
    *,
    size: str = "",
    quality: str = "",
    duration_seconds: int | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"model": model, "prompt": prompt}

    if size.strip():
        payload["size"] = size.strip()
    if quality.strip():
        payload["quality"] = quality.strip()
    if duration_seconds is not None:
        payload["seconds"] = duration_seconds
    return payload


def _request_json_or_media(
    url: str,
    api_key: str,
    payload: dict[str, Any],
    media_type: str,
) -> tuple[object, str]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            content_type = response.headers.get_content_type()
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise ValueError(_error_message(error, media_type)) from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach the {media_type} API: {error.reason}") from error

    if content_type == "application/json" or raw.lstrip().startswith((b"{", b"[")):
        try:
            return json.loads(raw.decode("utf-8")), content_type
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"The {media_type} API returned invalid JSON.") from error
    return raw, content_type


def _path_value(payload: object, path: str) -> object:
    value = payload
    for segment in path.split("."):
        if isinstance(value, dict):
            value = value.get(segment)
        elif isinstance(value, list) and segment.isdigit():
            index = int(segment)
            value = value[index] if 0 <= index < len(value) else None
        else:
            return None
    return value


def _poll_video(
    body: object,
    profile: dict[str, Any],
    api_key: str,
    profile_name: str,
) -> object:
    poll = _object(profile.get("poll"), "poll")
    if not poll.get("enabled"):
        return body

    task_id = _path_value(body, _string(poll.get("id_path"), "id"))
    if not isinstance(task_id, (str, int)) or not str(task_id):
        raise ValueError(f"Video profile {profile_name!r} has polling enabled but the initial response has no task id.")
    endpoint_template = _string(poll.get("endpoint"))
    if not endpoint_template:
        raise ValueError(f"Video profile {profile_name!r} has polling enabled but poll.endpoint is empty.")

    status_path = _string(poll.get("status_path"), "status")
    completed_values = poll.get("completed_values", ["completed", "succeeded"])
    failed_values = poll.get("failed_values", ["failed", "cancelled", "canceled"])
    if not isinstance(completed_values, list) or not isinstance(failed_values, list):
        raise ValueError("poll.completed_values and poll.failed_values must be JSON arrays.")
    completed = {str(value).lower() for value in completed_values}
    failed = {str(value).lower() for value in failed_values}
    interval = float(poll.get("interval_seconds", 5))
    timeout = float(poll.get("timeout_seconds", 600))
    if interval <= 0 or timeout <= 0:
        raise ValueError("poll.interval_seconds and poll.timeout_seconds must be greater than zero.")

    base_url = _string(profile.get("base_url"))
    poll_url = _configured_endpoint(base_url, endpoint_template.format(id=urllib.parse.quote(str(task_id), safe="")))
    deadline = time.monotonic() + timeout
    while True:
        request = urllib.request.Request(poll_url, headers={"Authorization": f"Bearer {api_key}"}, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            raise ValueError(_error_message(error, "video")) from error
        except (urllib.error.URLError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not retrieve the video task status: {error}") from error

        status = _path_value(result, status_path)
        normalized_status = str(status).lower() if status is not None else ""
        if normalized_status in completed:
            return result
        if normalized_status in failed:
            detail = _path_value(result, _string(poll.get("error_path"), "error.message"))
            raise ValueError(f"Video generation failed with status {status!r}: {detail or 'no error detail returned'}")
        if time.monotonic() >= deadline:
            raise ValueError(f"Timed out after {timeout:g} seconds waiting for video task {task_id} (last status: {status!r}).")
        time.sleep(interval)


def _media_item(body: object) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ValueError("Media API response was not a JSON object.")
    data = body.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        return data[0]
    if isinstance(data, dict):
        return data
    return body


def _absolute_media_url(url: str, profile: dict[str, Any]) -> str:
    if url.startswith("/"):
        base_url = _string(profile.get("base_url"))
        if not base_url:
            raise ValueError("The media API returned a relative URL but the profile has no base_url.")
        return urllib.parse.urljoin(f"{base_url.rstrip('/')}/", url)
    return url


def _save_media_response(
    response_body: object,
    content_type: str,
    profile: dict[str, Any],
    media_type: str,
) -> tuple[bytes, str, Path]:
    if isinstance(response_body, bytes):
        media_bytes = response_body
        mime_type = content_type
    else:
        item = _media_item(response_body)
        encoded = next(
            (item.get(key) for key in ("b64_json", "b64_video", "video_base64", "base64") if isinstance(item.get(key), str)),
            None,
        )
        url = _aicopy_find_url(response_body) if is_aicopy_profile(profile) else next(
            (item.get(key) for key in ("url", "video_url", "output_url", "download_url") if isinstance(item.get(key), str)),
            None,
        )
        if isinstance(encoded, str):
            try:
                media_bytes = base64.b64decode(encoded, validate=True)
            except (ValueError, binascii.Error) as error:
                raise ValueError(f"{media_type.title()} API returned invalid base64 media data.") from error
            mime_type = _string(profile.get("response_mime_type"), f"{media_type}/mp4" if media_type == "video" else "image/png")
        elif isinstance(url, str):
            media_bytes, mime_type = _download(
                _absolute_media_url(url, profile),
                media_type,
                _string(profile.get("key")),
                _string(profile.get("base_url")),
            )
        else:
            keys = ", ".join(sorted(str(key) for key in item))
            raise ValueError(
                f"{media_type.title()} API response did not contain a supported media field. "
                f"Expected url, video_url, output_url, b64_json, or b64_video; received keys: {keys or 'none'}."
            )

    default_dir = DEFAULT_VIDEO_OUTPUT_DIR if media_type == "video" else DEFAULT_IMAGE_OUTPUT_DIR
    output_dir = Path(_string(profile.get("output_dir")) or str(default_dir)).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = (
        f"openai-{media_type}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-"
        f"{uuid.uuid4().hex[:8]}.{_extension(mime_type, media_type)}"
    )
    output_path = output_dir / filename
    output_path.write_bytes(media_bytes)
    return media_bytes, mime_type, output_path


def _generate_image(prompt: str, size: str, quality: str, model: str, profile_name: str) -> tuple[bytes, str, Path]:
    config = _read_config()
    resolved_name, profile = _resolve_profile("image_generate", profile_name, config)
    api_key = _api_key(profile, resolved_name)
    resolved_model = _model(profile, model)
    if is_aicopy_profile(profile):
        body, content_type = _aicopy_generate_image(profile, api_key, resolved_model, prompt, size, quality)
        return _save_media_response(body, content_type, profile, "image")
    endpoint = _configured_endpoint(_string(profile.get("base_url")), "images/generations")
    payload = _payload(prompt, resolved_model, size=size, quality=quality)
    payload["n"] = 1
    body, content_type = _request_json_or_media(endpoint, api_key, payload, "image")
    return _save_media_response(body, content_type, profile, "image")


def _multipart_body(fields: dict[str, Any], image_paths: list[Path]) -> tuple[bytes, str]:
    boundary = f"----openai-media-mcp-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        serialized = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            serialized.encode("utf-8"),
            b"\r\n",
        ))
    for image_path in image_paths:
        mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
        filename = image_path.name.replace('"', "")
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image[]"; filename="{filename}"\r\n'.encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            image_path.read_bytes(),
            b"\r\n",
        ))
    chunks.append(f"--{boundary}--\r\n".encode())
    return b"".join(chunks), boundary


def _edit(
    prompt: str,
    image_paths: list[str],
    size: str,
    quality: str,
    model: str,
    profile_name: str,
) -> tuple[bytes, str, Path]:
    paths = [Path(raw_path).expanduser() for raw_path in image_paths]
    if not paths:
        raise ValueError("image_paths must include at least one local image file.")
    if len(paths) > 16:
        raise ValueError("image_paths supports at most 16 reference images.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Reference image does not exist: {path}")

    config = _read_config()
    resolved_name, profile = _resolve_profile("image_edit", profile_name, config)
    api_key = _api_key(profile, resolved_name)
    fields = _payload(prompt, _model(profile, model), size=size, quality=quality)
    fields["n"] = 1
    data, boundary = _multipart_body(fields, paths)
    endpoint = _configured_endpoint(_string(profile.get("base_url")), "images/edits")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            content_type = response.headers.get_content_type()
            raw = response.read()
    except urllib.error.HTTPError as error:
        raise ValueError(_error_message(error, "image")) from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach the image API: {error.reason}") from error
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Image edit API returned invalid JSON.") from error
    return _save_media_response(body, content_type, profile, "image")


def _generate_video(
    prompt: str,
    duration_seconds: int,
    size: str,
    model: str,
    profile_name: str,
) -> tuple[bytes, str, Path]:
    config = _read_config()
    resolved_name, profile = _resolve_profile("video", profile_name, config)
    api_key = _api_key(profile, resolved_name)
    endpoint = _configured_endpoint(_string(profile.get("base_url")), "videos/generations")
    payload = _payload(prompt, _model(profile, model), size=size, duration_seconds=duration_seconds)
    body, content_type = _request_json_or_media(endpoint, api_key, payload, "video")
    if not isinstance(body, bytes):
        body = _poll_video(body, profile, api_key, resolved_name)
    return _save_media_response(body, content_type, profile, "video")


def _profile_summary(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    visible = {"type", "adapter", "base_url", "model", "model_name"}
    summary = {key: profile[key] for key in visible if key in profile}
    summary["name"] = name
    return summary


def _image_model_options(profile_name: str, profile: dict[str, Any]) -> dict[str, Any]:
    if is_aicopy_profile(profile):
        key = _string(profile.get("key"))
        try:
            options = _aicopy_image_pricing_options(profile, key)
        except ValueError:
            if key:
                raise
            return {
                "profile": profile_name,
                "adapter": _string(profile.get("adapter")),
                "models": [],
                "price_note": f"Set the image key for profile {profile_name!r} to refresh the AICopy pricing catalog.",
            }
    else:
        model = _string(profile.get("model")) or _string(profile.get("model_name"))
        options = [{"model": model, "estimated_price": None, "price_note": "configured provider does not expose pricing"}] if model else []
    return {"profile": profile_name, "adapter": _string(profile.get("adapter")), "models": options}


def _image_models_text(config: dict[str, Any], requested_profile: str = "") -> str:
    profiles = _object(config.get("profiles"), "profiles")
    if requested_profile.strip():
        profile_name, profile = _resolve_profile("image_generate", requested_profile, config)
        providers = [_image_model_options(profile_name, profile)]
    else:
        names = _profile_names(config, "image_generate")
        if not names:
            raise ValueError(f"No image_generate profile is configured in {CONFIG_FILE}.")
        providers = [_image_model_options(name, dict(profiles[name])) for name in names]
    return json.dumps(
        {
            "message": "Choose a profile and model, then call generate_image again with both values.",
            "providers": providers,
        },
        ensure_ascii=False,
        indent=2,
    )


def _video_model_text(
    profile_name: str,
    profile: dict[str, Any],
    duration_seconds: int,
    requires_reference_video: bool = False,
    requires_reference_image: bool = False,
    requires_reference_audio: bool = False,
) -> str:
    key = _api_key(profile, profile_name)
    if not is_aicopy_profile(profile):
        model = _string(profile.get("model")) or _string(profile.get("model_name"))
        if not model:
            raise ValueError(f"Profile {profile_name!r} has no model configured.")
        options = [{"model": model, "estimated_price": None, "price_note": "configured provider does not expose pricing"}]
    else:
        options = _aicopy_pricing_options(
            profile,
            key,
            duration_seconds or 5,
            requires_reference_video=requires_reference_video,
            requires_reference_image=requires_reference_image,
            requires_reference_audio=requires_reference_audio,
        )
    return json.dumps(
        {
            "profile": profile_name,
            "duration_seconds": duration_seconds or 5,
            "requires_reference_image": requires_reference_image,
            "requires_reference_video": requires_reference_video,
            "requires_reference_audio": requires_reference_audio,
            "message": "Choose one model and call generate_video again with model set to that exact name.",
            "models": options,
        },
        ensure_ascii=False,
        indent=2,
    )


@mcp.tool(description="List configured image and video profiles, their types, base URLs, and models. Call this before generation when more than one profile may be available.")
def list_media_profiles() -> list[types.TextContent]:
    """Return non-secret provider settings so an AI agent can select the right media tool and profile."""
    config = _read_config()
    profiles = _object(config.get("profiles"), "profiles")
    summaries = [
        _profile_summary(name, profile)
        for name, profile in sorted(profiles.items())
        if isinstance(name, str) and any(_profile_matches(profile, operation) for operation in ("image_generate", "image_edit", "video"))
    ]
    defaults = _object(config.get("defaults"), "defaults")
    return [
        types.TextContent(
            type="text",
            text=json.dumps({"defaults": defaults, "profiles": summaries}, ensure_ascii=False, indent=2),
        )
    ]


@mcp.tool(description="List image-generation providers, selectable models, and estimated prices. AICopy options are read from its cached pricing catalog and refresh after 24 hours.")
def list_image_models(profile: str = "") -> list[types.TextContent]:
    """Return image models across providers, or only for one requested image profile."""
    return [types.TextContent(type="text", text=_image_models_text(_read_config(), profile))]


@mcp.tool(description="List video models and their estimated prices. Set reference_image, reference_video, or reference_audio to list only AICopy models that accept the supplied reference media.")
def list_video_models(
    duration_seconds: int = 5,
    profile: str = "",
    reference_video: bool = False,
    reference_image: bool = False,
    reference_audio: bool = False,
) -> list[types.TextContent]:
    """Fetch the provider's video catalog and prices without generating a video."""
    if not 1 <= duration_seconds <= 3_600:
        raise ValueError("duration_seconds must be between 1 and 3600.")
    config = _read_config()
    resolved_name, resolved_profile = _resolve_profile("video", profile, config)
    return [
        types.TextContent(
            type="text",
            text=_video_model_text(
                resolved_name,
                resolved_profile,
                duration_seconds,
                reference_video,
                reference_image,
                reference_audio,
            ),
        )
    ]


@mcp.tool(description="Generate one image through a named configured image profile. If model is empty, this tool lists providers, selectable models, and AICopy estimated prices instead of generating.")
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "",
    model: str = "",
    profile: str = "",
) -> list[types.TextContent | types.ImageContent]:
    """Generate an image, return it inline, and save a local copy."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")
    if not model.strip():
        return [types.TextContent(type="text", text=_image_models_text(_read_config(), profile))]
    image_bytes, mime_type, output_path = _generate_image(prompt, size.strip() or "1024x1024", quality.strip(), model.strip(), profile)
    return [
        types.TextContent(type="text", text=f"Generated image saved to {output_path}"),
        types.ImageContent(type="image", data=base64.b64encode(image_bytes).decode("ascii"), mimeType=mime_type),
    ]


@mcp.tool(description="Generate an image from one or more local reference images using a named configured image profile. Use for identity-preserved edits, product try-ons, and multi-view boards.")
def edit_images(
    prompt: str,
    image_paths: list[str],
    size: str = "2048x1152",
    quality: str = "high",
    model: str = "",
    profile: str = "",
) -> list[types.TextContent | types.ImageContent]:
    """Generate an image from one or more local reference images and return it inline."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")
    image_bytes, mime_type, output_path = _edit(prompt, image_paths, size.strip() or "2048x1152", quality.strip(), model.strip(), profile)
    return [
        types.TextContent(type="text", text=f"Generated image saved to {output_path}"),
        types.ImageContent(type="image", data=base64.b64encode(image_bytes).decode("ascii"), mimeType=mime_type),
    ]


@mcp.tool(description="Generate one video through a named configured video profile. If model is empty, this tool lists all supported models and estimated prices instead of generating. For AICopy, pass local reference image/video/audio paths and generation_mode when needed.")
def generate_video(
    prompt: str,
    duration_seconds: int = 0,
    size: str = "",
    model: str = "",
    profile: str = "",
    aspect_ratio: str = "16:9",
    generation_mode: str = "",
    image_paths: list[str] | None = None,
    video_paths: list[str] | None = None,
    audio_paths: list[str] | None = None,
) -> list[types.TextContent]:
    """Generate a video, or list model/pricing choices when model is omitted."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")
    if duration_seconds and not 1 <= duration_seconds <= 3_600:
        raise ValueError("duration_seconds must be between 1 and 3600.")
    config = _read_config()
    resolved_name, resolved_profile = _resolve_profile("video", profile, config)
    if not model.strip():
        return [
            types.TextContent(
                type="text",
                text=_video_model_text(
                    resolved_name,
                    resolved_profile,
                    duration_seconds or 5,
                    bool(video_paths),
                    bool(image_paths),
                    bool(audio_paths),
                ),
            )
        ]
    if is_aicopy_profile(resolved_profile):
        api_key = _api_key(resolved_profile, resolved_name)
        body, content_type, _, _ = _aicopy_generate(
            resolved_profile,
            api_key,
            model.strip(),
            prompt,
            duration_seconds,
            aspect_ratio,
            generation_mode,
            image_paths or [],
            video_paths or [],
            audio_paths or [],
        )
        _, mime_type, output_path = _save_media_response(body, content_type, resolved_profile, "video")
        return [types.TextContent(type="text", text=f"Generated video ({mime_type}) saved to {output_path}")]
    duration = duration_seconds or 5
    _, mime_type, output_path = _generate_video(prompt, duration, size.strip(), model.strip(), resolved_name)
    return [types.TextContent(type="text", text=f"Generated video ({mime_type}) saved to {output_path}")]


def _self_check() -> None:
    assert _configured_endpoint("https://api.example.com/v1", "videos/generations") == "https://api.example.com/v1/videos/generations"
    assert _profile_matches({"type": "image"}, "image_generate")
    assert _profile_matches({"type": "image"}, "image_edit")
    assert not _profile_matches({"type": "video"}, "image_edit")
    assert _extension("video/mp4", "video") == "mp4"
    assert _path_value({"data": [{"id": "video-1"}]}, "data.0.id") == "video-1"
    assert _payload("p", "x", duration_seconds=5) == {
        "model": "x",
        "prompt": "p",
        "seconds": 5,
    }
    assert is_aicopy_profile({"adapter": "aicopy", "base_url": "https://example.invalid"})
    assert _aicopy._model_family("开源h3-720p-按次") == "h3"
    assert _aicopy._model_family("开源h3-720p") == "minimax_h3"
    assert _aicopy._model_family("sd-720fast（按秒）") == "sd_rotate"
    assert _aicopy.supports_reference_video("sd-720fast（按秒）")
    assert not _aicopy.supports_reference_video("官方h3-720p")
    assert _aicopy.is_video_model({"model_name": "【官方稳定版】2.5-720p"})

    # Price data is persisted for 24 hours, so listing models does not make a
    # second pricing request while the cache is fresh.
    with tempfile.TemporaryDirectory() as directory:
        cache_path = Path(directory) / "pricing-cache.json"
        profile = {
            "base_url": "https://api.aicopy.top",
            "pricing_cache_file": str(cache_path),
        }
        requests: list[str] = []
        original_request = _aicopy._json_request

        def fake_request(method: str, url: str, key: str, **_: Any) -> tuple[object, str, int]:
            requests.append(f"{method} {url}")
            return (
                {
                    "data": [
                        {
                            "model_name": "sd-2.5-480p不卡脸(按秒)",
                            "model_price": 0.3,
                            "price_unit": "秒",
                            "supported_endpoint_types": ["openai"],
                        }
                    ],
                    "group_ratio": {"default": 1.2},
                },
                "application/json",
                200,
            )

        try:
            _aicopy._json_request = fake_request
            first = _aicopy.pricing_options(profile, "not-used", duration_seconds=5)
            second = _aicopy.pricing_options(profile, "not-used", duration_seconds=5)
            video_reference_only = _aicopy.pricing_options(
                profile,
                "not-used",
                duration_seconds=5,
                requires_reference_video=True,
            )
        finally:
            _aicopy._json_request = original_request
        assert len(requests) == 1
        assert first == second and first[0]["estimated_price"] == 1.8
        assert video_reference_only[0]["supports_reference_video"]

    protocol_profile = {"base_url": "https://api.aicopy.top"}
    protocol_cases = (
        ("grok-imagine-1.0-video", ["https://example.com/image.png"], [], "grok1", "/v1/videos", "image"),
        ("grok-imagine-video-1.5", ["https://example.com/image.png"], [], "grok15", "/v1/videos", "reference_images"),
        ("happyhorse-1.1-i2v-720p", ["https://example.com/image.png"], [], "happyhorse", "/v1/videos", "image_url"),
        ("官方h3-720p", [], [], "h3", "/v1/video/generations", "content"),
        ("【官方稳定版】2.5-720p", [], ["https://example.com/video.mp4"], "official_sd25", "/v1/videos", "video_url"),
        ("【官方稳定版】sd2.0-720p-fast", [], ["https://example.com/video.mp4"], "official_sd20_high", "/v1/video/generations", "video_url"),
        ("开源h3-720p", [], ["https://example.com/video.mp4"], "minimax_h3", "/v1/video/generations", "reference_videos"),
        ("sd-2.5-720p不卡脸(按秒)", [], ["https://example.com/video.mp4"], "sd25_low", "/v1/videos", "videos"),
        ("sd2.0-720fast-不卡脸（按秒）", [], ["https://example.com/video.mp4"], "sd20_low", "/v1/videos", "videos"),
        ("sd2.0-720fast-ad渠道16x9", [], ["https://example.com/video.mp4"], "sd20_ad", "/v1/videos", "media"),
        ("sd-720满血-933（按秒）", [], ["https://example.com/video.mp4"], "sd933", "/v1/video/generations", "video_references"),
        ("sd-720fast（按秒）", [], ["https://example.com/video.mp4"], "sd_rotate", "/v1/videos", "reference_videos"),
        ("sd-720满血-900（不售后）", ["https://example.com/image.png"], [], "sd900", "/v1/videos", "reference_images"),
        ("sd-2.0-720满血（不卡脸）惊喜渠道", [], ["https://example.com/video.mp4"], "surprise", "/v1/video/generations", "video_references"),
        ("omni-fast-视频编辑（无水印）", [], ["https://example.com/video.mp4"], "omni", "/v1/videos", "video_url"),
    )
    for model, images, videos, family, path, payload_key in protocol_cases:
        create_url, payload, poll_url, actual_family, _ = _aicopy.build_request(
            protocol_profile,
            "not-used",
            model,
            "protocol self-check",
            0,
            "16:9",
            "",
            images,
            videos,
            [],
        )
        assert create_url == f"https://api.aicopy.top{path}"
        assert poll_url == f"https://api.aicopy.top{path}/{{id}}"
        assert actual_family == family
        assert payload_key in {key for key, _ in _aicopy._recursive_values(payload)}

    _, official_payload, _, _, _ = _aicopy.build_request(
        protocol_profile,
        "not-used",
        "【官方稳定版】2.5-720p",
        "protocol self-check",
        10,
        "16:9",
        "first_last",
        ["https://example.com/first.png", "https://example.com/last.png"],
        [],
        [],
    )
    assert [item["role"] for item in official_payload["content"][1:]] == ["first_frame", "last_frame"]
    assert "ratio" not in official_payload

    _, high_payload, _, _, _ = _aicopy.build_request(
        protocol_profile,
        "not-used",
        "【官方稳定版】sd2.0-720p-fast",
        "protocol self-check",
        10,
        "16:9",
        "reference",
        ["https://example.com/image.png"],
        ["https://example.com/video.mp4"],
        ["https://example.com/audio.mp3"],
    )
    assert high_payload["content"][1]["role"] == "reference_image"
    assert "role" not in high_payload["content"][2]
    assert "role" not in high_payload["content"][3]

    try:
        _aicopy.build_request(
            protocol_profile,
            "not-used",
            "开源h3-720p",
            "protocol self-check",
            10,
            "16:9",
            "first_last",
            ["https://example.com/first.png", "https://example.com/last.png"],
            ["https://example.com/video.mp4"],
            [],
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Minimax H3 accepted incompatible first_last references")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        mcp.run(transport="stdio")
