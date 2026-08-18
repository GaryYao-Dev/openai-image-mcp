"""AICopy adapter facade.

The public functions stay here for MCP compatibility. Implementation details
are split into model catalog, protocol builders, transport, and pricing modules.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from pathlib import Path
from typing import Any

from aicopy_models import (
    MODEL_CATALOG,
    ModelCapabilities,
    ModelCatalog,
    ModelDescriptor,
    ModelRule,
    is_video_model,
    model_family,
    supports_media,
    supports_reference_video,
)
from aicopy_images import generate as generate_image
from aicopy_pricing import (
    DEFAULT_PRICING_CACHE_FILE,
    PRICING_CACHE_TTL_SECONDS,
    cached_entry,
    fetch_pricing as _fetch_pricing,
    image_pricing_options as _image_pricing_options,
    pricing_cache_path,
    pricing_options as _pricing_options,
    pricing_payload,
    pricing_url,
    write_pricing_cache,
)
from aicopy_protocols import MediaReferences, build_plan, validate_media
from aicopy_transport import (
    as_data_url,
    endpoint,
    find_status,
    find_task_id,
    find_url,
    json_request as _transport_json_request,
    multipart_upload,
    recursive_values,
    resolve_asset,
    root_base,
)


def is_aicopy_profile(profile: dict[str, Any]) -> bool:
    adapter = str(profile.get("adapter", "")).strip().lower()
    base_url = str(profile.get("base_url", "")).lower()
    return adapter == "aicopy" or "api.aicopy.top" in base_url


def _json_request(
    method: str,
    url: str,
    key: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120,
) -> tuple[object, str, int]:
    """Compatibility seam used by tests and callers that inject HTTP responses."""
    return _transport_json_request(method, url, key, body, headers, timeout)


def _recursive_values(value: object):
    return recursive_values(value)


def _model_family(model: str) -> str:
    return model_family(model)


_root_base = root_base
_multipart_upload = multipart_upload
_as_data_url = as_data_url
_pricing_url = pricing_url
_pricing_cache_path = pricing_cache_path
_cached_entry = cached_entry
_write_pricing_cache = write_pricing_cache


def _pricing_payload(
    profile: dict[str, Any],
    key: str,
    cache_file: str | Path | None = None,
    max_age_seconds: float = PRICING_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    return pricing_payload(profile, key, _json_request, cache_file, max_age_seconds)


def fetch_pricing(
    profile: dict[str, Any],
    key: str,
    max_age_seconds: float = PRICING_CACHE_TTL_SECONDS,
    cache_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    return _fetch_pricing(profile, key, _json_request, max_age_seconds, cache_file)


def pricing_options(
    profile: dict[str, Any],
    key: str,
    duration_seconds: int = 5,
    cache_file: str | Path | None = None,
    requires_reference_video: bool = False,
    requires_reference_image: bool = False,
    requires_reference_audio: bool = False,
) -> list[dict[str, Any]]:
    return _pricing_options(
        profile,
        key,
        _json_request,
        duration_seconds,
        cache_file,
        requires_reference_video,
        requires_reference_image,
        requires_reference_audio,
    )


def image_pricing_options(
    profile: dict[str, Any],
    key: str,
    cache_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    return _image_pricing_options(profile, key, _json_request, cache_file)


def build_request(
    profile: dict[str, Any],
    key: str,
    model: str,
    prompt: str,
    duration_seconds: int,
    aspect_ratio: str,
    generation_mode: str,
    image_paths: list[str],
    video_paths: list[str],
    audio_paths: list[str],
) -> tuple[str, dict[str, Any], str, str, dict[str, str]]:
    descriptor = MODEL_CATALOG.resolve(model)
    supplied_media = MediaReferences(tuple(image_paths), tuple(video_paths), tuple(audio_paths))
    validate_media(descriptor.capabilities, model, supplied_media)
    prefer_data = descriptor.asset_encoding == "data"
    resolved_media = MediaReferences(
        images=tuple(resolve_asset(value, key, profile, "image", prefer_data) for value in image_paths),
        videos=tuple(resolve_asset(value, key, profile, "video") for value in video_paths),
        audios=tuple(resolve_asset(value, key, profile, "audio") for value in audio_paths),
    )
    plan = build_plan(
        descriptor,
        model,
        prompt,
        duration_seconds,
        aspect_ratio,
        generation_mode,
        resolved_media,
    )
    base_url = str(profile["base_url"])
    return (
        endpoint(base_url, plan.create_path),
        plan.payload,
        endpoint(base_url, plan.poll_path),
        descriptor.family,
        plan.headers,
    )


def generate(
    profile: dict[str, Any],
    key: str,
    model: str,
    prompt: str,
    duration_seconds: int,
    aspect_ratio: str,
    generation_mode: str,
    image_paths: list[str],
    video_paths: list[str],
    audio_paths: list[str],
) -> tuple[object, str, str, str]:
    create_url, payload, poll_template, family, extra_headers = build_request(
        profile,
        key,
        model,
        prompt,
        duration_seconds,
        aspect_ratio,
        generation_mode,
        image_paths,
        video_paths,
        audio_paths,
    )
    body, content_type, _ = _json_request("POST", create_url, key, payload, extra_headers, timeout=300)
    if isinstance(body, bytes) or find_url(body):
        return body, content_type, poll_template, family
    task_id = find_task_id(body)
    if not task_id:
        raise ValueError("AICopy video creation returned neither a video URL nor a task ID.")
    timeout = float(profile.get("timeout_seconds", 3600))
    interval = float(profile.get("poll_interval_seconds", 5))
    deadline = time.monotonic() + timeout
    poll_url = poll_template.format(id=urllib.parse.quote(task_id, safe=""))
    last_status = ""
    while time.monotonic() < deadline:
        result, result_type, _ = _json_request("GET", poll_url, key, timeout=120)
        last_status = find_status(result)
        if find_url(result):
            return result, result_type, poll_template, family
        if last_status.upper() in {"SUCCESS", "SUCCEEDED", "COMPLETED", "COMPLETE", "DONE", "FINISHED", "OK"}:
            content_url = endpoint(str(profile.get("base_url", "")), f"/v1/videos/{urllib.parse.quote(task_id, safe='')}/content")
            try:
                raw, raw_type, _ = _json_request("GET", content_url, key, timeout=300)
                return raw, raw_type, poll_template, family
            except ValueError as error:
                raise ValueError("AICopy task completed but returned no downloadable video URL.") from error
        if last_status.upper() in {"FAILURE", "FAILED", "ERROR", "CANCELLED", "CANCELED", "TIMEOUT"}:
            raise ValueError(f"AICopy video task failed: {json.dumps(result, ensure_ascii=False)[:800]}")
        time.sleep(interval)
    raise ValueError(f"Timed out after {timeout:g} seconds waiting for AICopy video task (last status: {last_status or 'unknown'}).")
