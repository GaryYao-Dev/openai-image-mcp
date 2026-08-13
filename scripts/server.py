# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "mcp==1.29.0",
# ]
# ///

"""Small stdio MCP server for an OpenAI-compatible Images API."""

from __future__ import annotations

import base64
import binascii
import json
import mimetypes
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from mcp import types
from mcp.server.fastmcp import FastMCP


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PLUGIN_ROOT / ".env"
DEFAULT_OUTPUT_DIR = Path.home() / "Pictures" / "Codex Generated Images"
mcp = FastMCP("OpenAI Image")


def _read_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.removeprefix("export ").strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _setting(name: str, values: dict[str, str], default: str = "") -> str:
    return os.environ.get(name) or values.get(name, default)


def _endpoint(base_url: str, operation: str = "generations") -> str:
    base_url = base_url.strip().rstrip("/")
    if not base_url:
        raise ValueError("OPENAI_BASE_URL is empty in the plugin .env file.")
    suffix = f"/images/{operation}"
    if base_url.endswith(suffix):
        return base_url
    if base_url.endswith("/images/generations") or base_url.endswith("/images/edits"):
        base_url = base_url.rsplit("/images/", 1)[0]
    return f"{base_url}{suffix}"


def _error_message(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8", errors="replace"))
        detail = payload.get("error", payload)
        if isinstance(detail, dict):
            detail = detail.get("message") or detail.get("code") or "request rejected"
    except Exception:
        detail = "request rejected"
    return f"Image API returned HTTP {error.code}: {str(detail)[:500]}"


def _download(url: str) -> tuple[bytes, str]:
    if urllib.parse.urlparse(url).scheme not in {"http", "https"}:
        raise ValueError("The image API returned an unsupported image URL.")
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read(), response.headers.get_content_type()


def _extension(mime_type: str) -> str:
    return {
        "image/jpeg": "jpg",
        "image/png": "png",
        "image/webp": "webp",
    }.get(mime_type.lower(), "png")


def _api_key(values: dict[str, str]) -> str:
    api_key = _setting("OPENAI_API_KEY", values)
    if not api_key:
        raise ValueError(f"Set OPENAI_API_KEY in {ENV_FILE}; do not paste it into chat.")
    return api_key


def _save_response(body: object, values: dict[str, str]) -> tuple[bytes, str, Path]:
    items = body.get("data") if isinstance(body, dict) else None
    if not isinstance(items, list) or not items or not isinstance(items[0], dict):
        raise ValueError("Image API response did not contain data[0].b64_json or data[0].url.")
    item = items[0]
    if isinstance(item.get("b64_json"), str):
        try:
            image_bytes = base64.b64decode(item["b64_json"], validate=True)
        except (ValueError, binascii.Error) as error:
            raise ValueError("Image API returned invalid b64_json data.") from error
        mime_type = "image/png"
    elif isinstance(item.get("url"), str):
        image_bytes, mime_type = _download(item["url"])
    else:
        raise ValueError("Image API response did not contain b64_json or url image data.")

    output_dir = Path(_setting("OPENAI_IMAGE_OUTPUT_DIR", values, str(DEFAULT_OUTPUT_DIR))).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"openai-image-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}.{_extension(mime_type)}"
    output_path = output_dir / filename
    output_path.write_bytes(image_bytes)
    return image_bytes, mime_type, output_path


def _generate(prompt: str, size: str, quality: str, model: str) -> tuple[bytes, str, Path]:
    values = _read_dotenv(ENV_FILE)
    api_key = _api_key(values)

    payload: dict[str, object] = {
        "model": model or _setting("OPENAI_IMAGE_MODEL", values, "gpt-image-2"),
        "prompt": prompt,
        "size": size,
        "n": 1,
    }
    if quality:
        payload["quality"] = quality

    request = urllib.request.Request(
        _endpoint(_setting("OPENAI_BASE_URL", values), "generations"),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ValueError(_error_message(error)) from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach the image API: {error.reason}") from error

    return _save_response(body, values)


def _multipart_body(fields: dict[str, str], image_paths: list[Path]) -> tuple[bytes, str]:
    boundary = f"----openai-image-mcp-{uuid.uuid4().hex}"
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.extend((
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode(),
            value.encode("utf-8"),
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


def _edit(prompt: str, image_paths: list[str], size: str, quality: str, model: str) -> tuple[bytes, str, Path]:
    paths = [Path(raw_path).expanduser() for raw_path in image_paths]
    if not paths:
        raise ValueError("image_paths must include at least one local image file.")
    if len(paths) > 16:
        raise ValueError("image_paths supports at most 16 reference images.")
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Reference image does not exist: {path}")

    values = _read_dotenv(ENV_FILE)
    api_key = _api_key(values)
    fields = {
        "model": model or _setting("OPENAI_IMAGE_MODEL", values, "gpt-image-2"),
        "prompt": prompt,
        "size": size,
        "n": "1",
    }
    if quality:
        fields["quality"] = quality
    data, boundary = _multipart_body(fields, paths)
    request = urllib.request.Request(
        _endpoint(_setting("OPENAI_BASE_URL", values), "edits"),
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        raise ValueError(_error_message(error)) from error
    except urllib.error.URLError as error:
        raise ValueError(f"Could not reach the image API: {error.reason}") from error
    return _save_response(body, values)


@mcp.tool(description="Generate one image using the configured OpenAI-compatible image API. Use for images, visual references, and multi-view boards.")
def generate_image(
    prompt: str,
    size: str = "1024x1024",
    quality: str = "",
    model: str = "",
) -> list[types.TextContent | types.ImageContent]:
    """Generate an image and return it inline, while also saving a local copy."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")
    image_bytes, mime_type, output_path = _generate(prompt, size.strip() or "1024x1024", quality.strip(), model.strip())
    return [
        types.TextContent(type="text", text=f"Generated image saved to {output_path}"),
        types.ImageContent(type="image", data=base64.b64encode(image_bytes).decode("ascii"), mimeType=mime_type),
    ]


@mcp.tool(description="Generate an image using one or more local reference images. Use for identity-preserved edits, product try-ons, and multi-view boards.")
def edit_images(
    prompt: str,
    image_paths: list[str],
    size: str = "2048x1152",
    quality: str = "high",
    model: str = "",
) -> list[types.TextContent | types.ImageContent]:
    """Generate an image from one or more local reference images and return it inline."""
    prompt = prompt.strip()
    if not prompt:
        raise ValueError("prompt must not be empty.")
    image_bytes, mime_type, output_path = _edit(prompt, image_paths, size.strip() or "2048x1152", quality.strip(), model.strip())
    return [
        types.TextContent(type="text", text=f"Generated image saved to {output_path}"),
        types.ImageContent(type="image", data=base64.b64encode(image_bytes).decode("ascii"), mimeType=mime_type),
    ]


def _self_check() -> None:
    assert _endpoint("https://api.example.com/v1") == "https://api.example.com/v1/images/generations"
    assert _endpoint("https://api.example.com/v1/images/generations/") == "https://api.example.com/v1/images/generations"
    assert _endpoint("https://api.example.com/v1/images/generations", "edits") == "https://api.example.com/v1/images/edits"
    assert _extension("image/jpeg") == "jpg"


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        _self_check()
    else:
        mcp.run(transport="stdio")
