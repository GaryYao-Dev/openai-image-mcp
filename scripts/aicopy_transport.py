"""HTTP transport, response extraction, and reference-asset handling."""

from __future__ import annotations

import base64
import json
import mimetypes
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def root_base(base_url: str) -> str:
    value = base_url.strip().rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def endpoint(base_url: str, path: str) -> str:
    base = base_url.strip().rstrip("/")
    normalized = "/" + path.lstrip("/")
    if base.endswith("/v1") and normalized.startswith("/v1/"):
        normalized = normalized[3:]
    return f"{base}{normalized}"


def json_request(
    method: str,
    url: str,
    key: str,
    body: object | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 120,
) -> tuple[object, str, int]:
    request_headers = {"Authorization": f"Bearer {key}"} if key else {}
    request_headers.update(headers or {})
    data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    if body is not None:
        request_headers.setdefault("Content-Type", "application/json")
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get_content_type()
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"AICopy API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach AICopy API: {error.reason}") from error
    if content_type == "application/json" or raw.lstrip().startswith((b"{", b"[")):
        try:
            return json.loads(raw.decode("utf-8")), content_type, status
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("AICopy API returned invalid JSON.") from error
    return raw, content_type, status


def recursive_values(value: object):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key).lower(), child
            yield from recursive_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from recursive_values(child)


def find_url(value: object) -> str:
    preferred = {
        "final_url", "final_urls", "video_url", "result_url", "download_url",
        "content_url", "output_url", "file_url", "video_uri", "media_url", "mp4_url", "url",
    }
    fallback: list[str] = []
    for key, child in recursive_values(value):
        candidates = child if isinstance(child, list) else [child]
        for candidate in candidates:
            if not isinstance(candidate, str) or not candidate.strip():
                continue
            if candidate.startswith(("http://", "https://", "data:", "/")):
                if key in preferred:
                    return candidate
                fallback.append(candidate)
    return fallback[0] if fallback else ""


def find_task_id(value: object) -> str:
    preferred = {"task_id", "taskid", "request_id", "requestid", "video_id", "videoid", "job_id", "jobid", "id"}
    for key, child in recursive_values(value):
        if key in preferred and isinstance(child, (str, int)) and str(child):
            candidate = str(child)
            if candidate.startswith(("task_", "job_", "video_", "request_")) or key != "id":
                return candidate
    return ""


def find_status(value: object) -> str:
    status_keys = {"status", "state", "task_status", "taskstatus", "job_status", "jobstatus"}
    for key, child in recursive_values(value):
        if key in status_keys and child is not None:
            return str(child).lower()
    return ""


def multipart_upload(url: str, key: str, file_path: Path, field_name: str) -> str:
    boundary = f"----aicopy-upload-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name.replace(chr(34), "")}"\r\n'.encode(),
            f"Content-Type: {mime_type}\r\n\r\n".encode(),
            file_path.read_bytes(),
            f"\r\n--{boundary}--\r\n".encode(),
        )
    )
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:500]
        raise ValueError(f"AICopy asset upload returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not upload asset to AICopy: {error.reason}") from error
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("AICopy asset upload returned invalid JSON.") from error
    uploaded_url = find_url(payload)
    if not uploaded_url:
        raise ValueError("AICopy asset upload succeeded but returned no public URL.")
    return uploaded_url


def as_data_url(file_path: Path) -> str:
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_asset(
    raw_value: str,
    key: str,
    profile: dict[str, Any],
    media_kind: str,
    prefer_data: bool = False,
) -> str:
    value = raw_value.strip()
    if value.startswith(("http://", "https://", "data:")):
        return value
    path = Path(value).expanduser()
    if not path.is_file():
        raise ValueError(f"Reference {media_kind} does not exist: {path}")
    if prefer_data:
        return as_data_url(path)
    upload_url = str(profile.get("upload_url", "")).strip() or "https://api.aione.help/v1/uploads"
    return multipart_upload(upload_url, key, path, media_kind)
