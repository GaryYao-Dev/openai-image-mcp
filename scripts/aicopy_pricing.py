"""Persistent AICopy pricing-cache access and model price projections."""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from aicopy_models import is_image_model, is_video_model, supports_media, supports_reference_video
from aicopy_transport import root_base


PRICING_CACHE_TTL_SECONDS = 24 * 60 * 60
DEFAULT_PRICING_CACHE_FILE = Path(__file__).resolve().parents[1] / "pricing-cache.json"
RequestFn = Callable[..., tuple[object, str, int]]


def pricing_url(profile: dict[str, Any]) -> str:
    base_url = str(profile.get("base_url", "")).strip()
    return str(profile.get("pricing_url", "")).strip() or f"{root_base(base_url)}/api/pricing"


def pricing_cache_path(profile: dict[str, Any], cache_file: str | Path | None = None) -> Path:
    configured = str(profile.get("pricing_cache_file", "")).strip()
    return Path(configured).expanduser() if configured else Path(cache_file or DEFAULT_PRICING_CACHE_FILE)


def cached_entry(cache_path: Path, source_url: str, max_age_seconds: float) -> dict[str, Any] | None:
    try:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(cache, dict):
        return None
    entries = cache.get("entries")
    entry = entries.get(source_url) if isinstance(entries, dict) else cache if cache.get("pricing_url") == source_url else None
    if not isinstance(entry, dict) or not isinstance(entry.get("fetched_at"), str):
        return None
    try:
        timestamp = datetime.fromisoformat(entry["fetched_at"].replace("Z", "+00:00"))
        timestamp = timestamp.replace(tzinfo=timezone.utc) if timestamp.tzinfo is None else timestamp
        age = time.time() - timestamp.timestamp()
    except ValueError:
        return None
    if age < 0 or age >= max_age_seconds:
        return None
    data = entry.get("data")
    if not isinstance(data, list):
        return None
    response: dict[str, Any] = {"data": data}
    for field in ("group_ratio", "pricing_version"):
        if field in entry:
            response[field] = entry[field]
    return response


def write_pricing_cache(cache_path: Path, source_url: str, response: dict[str, Any]) -> None:
    entry = {
        "pricing_url": source_url,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "data": response.get("data", []),
        "group_ratio": response.get("group_ratio", {}),
        "pricing_version": response.get("pricing_version"),
    }
    temporary: Path | None = None
    try:
        cache: dict[str, Any] = {}
        if cache_path.exists():
            existing = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(existing, dict) and isinstance(existing.get("entries"), dict):
                cache["entries"] = dict(existing["entries"])
        cache.setdefault("entries", {})[source_url] = entry
        cache["version"] = 1
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(cache_path)
    except (OSError, TypeError, ValueError):
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def pricing_payload(
    profile: dict[str, Any],
    key: str,
    request: RequestFn,
    cache_file: str | Path | None = None,
    max_age_seconds: float = PRICING_CACHE_TTL_SECONDS,
) -> dict[str, Any]:
    source_url = pricing_url(profile)
    cache_path = pricing_cache_path(profile, cache_file)
    cached = cached_entry(cache_path, source_url, max_age_seconds)
    if cached is not None:
        return cached
    raw, _, _ = request("GET", source_url, key)
    if not isinstance(raw, dict) or not isinstance(raw.get("data"), list):
        raise ValueError("AICopy pricing response did not contain data[].")
    write_pricing_cache(cache_path, source_url, raw)
    return raw


def fetch_pricing(
    profile: dict[str, Any],
    key: str,
    request: RequestFn,
    max_age_seconds: float = PRICING_CACHE_TTL_SECONDS,
    cache_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    body = pricing_payload(profile, key, request, cache_file, max_age_seconds)
    return [dict(record) for record in body["data"] if isinstance(record, dict) and is_video_model(record)]


def image_pricing_options(
    profile: dict[str, Any],
    key: str,
    request: RequestFn,
    cache_file: str | Path | None = None,
) -> list[dict[str, Any]]:
    body = pricing_payload(profile, key, request, cache_file)
    group = str(profile.get("pricing_group", "default"))
    ratios = body.get("group_ratio", {})
    ratio = float(ratios.get(group, 1)) if isinstance(ratios, dict) else 1.0
    options: list[dict[str, Any]] = []
    for record in body["data"]:
        if not isinstance(record, dict) or not is_image_model(record):
            continue
        price_unit = str(record.get("price_unit") or "次")
        base_price = float(record.get("model_price") or 0)
        options.append(
            {
                "model": str(record.get("model_name", "")),
                "price_unit": price_unit,
                "base_price": base_price,
                "pricing_group": group,
                "group_ratio": ratio,
                "estimated_price": round(base_price * ratio, 6),
                "endpoint_types": record.get("supported_endpoint_types", []),
            }
        )
    return sorted(options, key=lambda item: str(item["model"]))


def pricing_options(
    profile: dict[str, Any],
    key: str,
    request: RequestFn,
    duration_seconds: int = 5,
    cache_file: str | Path | None = None,
    requires_reference_video: bool = False,
    requires_reference_image: bool = False,
    requires_reference_audio: bool = False,
) -> list[dict[str, Any]]:
    body = pricing_payload(profile, key, request, cache_file)
    records = [dict(record) for record in body["data"] if isinstance(record, dict) and is_video_model(record)]
    group = str(profile.get("pricing_group", "default"))
    ratios = body.get("group_ratio", {})
    ratio = float(ratios.get(group, 1)) if isinstance(ratios, dict) else 1.0
    options: list[dict[str, Any]] = []
    for record in records:
        model = str(record.get("model_name", ""))
        accepts_reference_video = supports_reference_video(model)
        accepts_references = supports_media(
            model,
            image_count=1 if requires_reference_image else 0,
            video_count=1 if requires_reference_video else 0,
            audio_count=1 if requires_reference_audio else 0,
        )
        if not accepts_references:
            continue
        base_price = float(record.get("model_price") or 0)
        price_unit = str(record.get("price_unit") or "次")
        quantity = duration_seconds if "秒" in price_unit else 1
        options.append(
            {
                "model": model,
                "price_unit": price_unit,
                "base_price": base_price,
                "pricing_group": group,
                "group_ratio": ratio,
                "estimated_price": round(base_price * ratio * quantity, 6),
                "duration_seconds": duration_seconds if "秒" in price_unit else None,
                "endpoint_types": record.get("supported_endpoint_types", []),
                "supports_reference_image": supports_media(model, image_count=1),
                "supports_reference_video": accepts_reference_video,
                "supports_reference_audio": supports_media(model, audio_count=1),
            }
        )
    return sorted(options, key=lambda item: str(item["model"]))
