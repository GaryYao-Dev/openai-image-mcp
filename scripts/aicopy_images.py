"""AICopy image request strategies and synchronous response handling."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from aicopy_transport import endpoint, find_url, json_request


@dataclass(frozen=True)
class ImageRequestPlan:
    path: str
    payload: dict[str, Any]


class ImageRequestBuilder(ABC):
    @abstractmethod
    def build(self, model: str, prompt: str, size: str, quality: str) -> ImageRequestPlan:
        raise NotImplementedError


class ChatImageRequestBuilder(ImageRequestBuilder):
    def build(self, model: str, prompt: str, size: str, quality: str) -> ImageRequestPlan:
        return ImageRequestPlan(
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
                "stream": True,
            },
        )


class OpenAIImageRequestBuilder(ImageRequestBuilder):
    def build(self, model: str, prompt: str, size: str, quality: str) -> ImageRequestPlan:
        payload: dict[str, Any] = {"model": model, "prompt": prompt, "n": 1}
        if size:
            payload["size"] = size
        if quality:
            payload["quality"] = quality
        return ImageRequestPlan("/v1/images/generations", payload)


class DoujieImageRequestBuilder(OpenAIImageRequestBuilder):
    def build(self, model: str, prompt: str, size: str, quality: str) -> ImageRequestPlan:
        plan = super().build(model, prompt, size, quality)
        return ImageRequestPlan(
            plan.path,
            {
                **plan.payload,
                "sequential_image_generation": "disabled",
                "response_format": "url",
                "stream": False,
                "watermark": False,
            },
        )


CHAT_IMAGE_PREFIXES = (
    "firefly-nano-banana-pro-",
    "firefly-nano-banana2-",
    "firefly-gpt-image-",
)
CHAT_IMAGE_BUILDER = ChatImageRequestBuilder()
OPENAI_IMAGE_BUILDER = OpenAIImageRequestBuilder()
DOUJIE_IMAGE_BUILDER = DoujieImageRequestBuilder()
URL_PATTERN = re.compile(r"https?://[^\s\]\[\"'<>]+")


def image_builder(model: str) -> ImageRequestBuilder:
    value = model.casefold()
    if value.startswith(CHAT_IMAGE_PREFIXES):
        return CHAT_IMAGE_BUILDER
    if "豆姐图片" in model:
        return DOUJIE_IMAGE_BUILDER
    return OPENAI_IMAGE_BUILDER


def build_plan(model: str, prompt: str, size: str = "", quality: str = "") -> ImageRequestPlan:
    return image_builder(model).build(model, prompt, size.strip(), quality.strip())


def _sse_url(raw: bytes) -> str:
    fragments: list[str] = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if not data or data == "[DONE]":
            continue
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            fragments.append(data)
            continue
        url = find_url(event)
        if url:
            return url
        fragments.append(data)
    match = URL_PATTERN.search("\n".join(fragments))
    return match.group(0) if match else ""


def generate(
    profile: dict[str, Any],
    key: str,
    model: str,
    prompt: str,
    size: str = "",
    quality: str = "",
) -> tuple[object, str]:
    plan = build_plan(model, prompt, size, quality)
    body, content_type, _ = json_request(
        "POST",
        endpoint(str(profile["base_url"]), plan.path),
        key,
        plan.payload,
        timeout=300,
    )
    if isinstance(body, bytes) and content_type == "text/event-stream":
        url = _sse_url(body)
        if not url:
            raise ValueError("AICopy image stream completed without an image URL.")
        return {"data": [{"url": url}]}, "application/json"
    return body, content_type
