"""Request planning strategies for documented AICopy video protocols."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from aicopy_models import ModelCapabilities, ModelDescriptor


def _duration(value: int, default: int, minimum: int, maximum: int, model: str) -> int:
    result = value or default
    if not minimum <= result <= maximum:
        raise ValueError(f"Model {model!r} supports duration {minimum}-{maximum} seconds.")
    return result


def _resolution(model: str) -> str:
    lower = model.lower()
    if "1080" in lower:
        return "1080P" if "happyhorse" in lower else "1080p"
    if "2k" in lower:
        return "2K"
    if "480" in lower:
        return "480p"
    return "720P" if "happyhorse" in lower else "720p"


def _size(aspect_ratio: str) -> str:
    return {
        "16:9": "1280x720",
        "9:16": "720x1280",
        "1:1": "1024x1024",
        "3:2": "1792x1024",
        "2:3": "1024x1792",
    }.get(aspect_ratio, "1280x720")


def _mode(generation_mode: str, images: tuple[str, ...], videos: tuple[str, ...], audios: tuple[str, ...]) -> str:
    aliases = {
        "文生": "text",
        "文生视频": "text",
        "text": "text",
        "text2video": "text",
        "t2v": "text",
        "首帧": "first_frame",
        "首帧生成视频": "first_frame",
        "first_frame": "first_frame",
        "firstframe": "first_frame",
        "image2video": "first_frame",
        "i2v": "first_frame",
        "首尾帧": "first_last",
        "首尾帧生成视频": "first_last",
        "first_last": "first_last",
        "firstlast": "first_last",
        "frames2video": "first_last",
        "多参考": "reference",
        "多参考图": "reference",
        "多参考图生成视频": "reference",
        "reference": "reference",
        "references": "reference",
        "参考": "reference",
        "首帧+参考图": "first_frame_reference",
        "首帧参考图": "first_frame_reference",
        "first_frame_reference": "first_frame_reference",
        "first_frame_references": "first_frame_reference",
    }
    normalized = generation_mode.strip().casefold().replace("-", "_").replace(" ", "_")
    if normalized:
        if normalized not in aliases:
            choices = ", ".join(sorted(set(aliases.values())))
            raise ValueError(f"Unsupported generation_mode {generation_mode!r}. Use one of: {choices}.")
        return aliases[normalized]
    if len(images) == 2 and not videos and not audios:
        return "first_last"
    if images and (videos or audios):
        return "reference"
    if images:
        return "first_frame" if len(images) == 1 else "reference"
    if videos or audios:
        return "reference"
    return "text"


def _image_roles(mode: str, images: tuple[str, ...]) -> tuple[str, ...]:
    if mode == "first_frame":
        return ("first_frame",)
    if mode == "first_last":
        return ("first_frame", "last_frame")
    if mode == "first_frame_reference":
        return ("first_frame",) + ("reference_image",) * (len(images) - 1)
    return ("reference_image",) * len(images)


def _content(
    prompt: str,
    images: tuple[str, ...],
    videos: tuple[str, ...],
    audios: tuple[str, ...],
    *,
    image_roles: tuple[str, ...] | None = None,
    include_reference_media_roles: bool = False,
) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    roles = image_roles or ("reference_image",) * len(images)
    content.extend(
        {"type": "image_url", "image_url": {"url": value}, "role": role}
        for value, role in zip(images, roles, strict=True)
    )
    content.extend(
        {"type": "video_url", "video_url": {"url": value}, **({"role": "reference_video"} if include_reference_media_roles else {})}
        for value in videos
    )
    content.extend(
        {"type": "audio_url", "audio_url": {"url": value}, **({"role": "reference_audio"} if include_reference_media_roles else {})}
        for value in audios
    )
    return content


def _require_mode(context: "BuildContext", *allowed: str) -> None:
    if context.mode not in allowed:
        supported = ", ".join(allowed)
        raise ValueError(f"Model {context.model!r} supports generation modes: {supported}; received {context.mode}.")


def _validate_media_limits(
    model: str,
    media: "MediaReferences",
    max_images: int,
    max_videos: int = 0,
    max_audios: int = 0,
) -> None:
    for kind, values, limit in (
        ("reference images", media.images, max_images),
        ("reference videos", media.videos, max_videos),
        ("reference audio files", media.audios, max_audios),
    ):
        if len(values) > limit:
            raise ValueError(f"Model {model!r} supports at most {limit} {kind}; received {len(values)}.")


def _validate_limits(context: "BuildContext", max_images: int, max_videos: int = 0, max_audios: int = 0) -> None:
    _validate_media_limits(context.model, context.media, max_images, max_videos, max_audios)


def _require_no_media(context: "BuildContext") -> None:
    if context.media.images or context.media.videos or context.media.audios:
        raise ValueError(f"Model {context.model!r} does not accept reference media in text mode.")


def _require_images(context: "BuildContext", minimum: int, maximum: int, mode_name: str) -> None:
    count = len(context.media.images)
    if not minimum <= count <= maximum:
        description = str(minimum) if minimum == maximum else f"{minimum}-{maximum}"
        raise ValueError(f"Model {context.model!r} requires {description} reference image(s) for {mode_name}; received {count}.")


def _require_reference(context: "BuildContext") -> None:
    if not (context.media.images or context.media.videos or context.media.audios):
        raise ValueError(f"Model {context.model!r} requires at least one reference item in reference mode.")


def _require_no_video_or_audio(context: "BuildContext", mode_name: str) -> None:
    if context.media.videos or context.media.audios:
        raise ValueError(f"Model {context.model!r} does not accept reference video or audio in {mode_name} mode.")


@dataclass(frozen=True)
class MediaReferences:
    images: tuple[str, ...] = ()
    videos: tuple[str, ...] = ()
    audios: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildContext:
    descriptor: ModelDescriptor
    model: str
    prompt: str
    duration_seconds: int
    aspect_ratio: str
    mode: str
    media: MediaReferences


@dataclass(frozen=True)
class RequestPlan:
    create_path: str
    poll_path: str
    payload: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class RequestBuilder(ABC):
    @abstractmethod
    def build(self, context: BuildContext) -> RequestPlan:
        raise NotImplementedError


def _plan(create_path: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> RequestPlan:
    return RequestPlan(create_path, f"{create_path}/{{id}}", payload, headers or {})


def _set_list(payload: dict[str, Any], field_name: str, values: tuple[str, ...], limit: int | None = None) -> None:
    if values:
        if limit is not None and len(values) > limit:
            raise ValueError(f"{field_name} supports at most {limit} reference item(s).")
        payload[field_name] = list(values)


def validate_media(
    capabilities: ModelCapabilities,
    model: str,
    media: MediaReferences,
    mode: str | None = None,
) -> None:
    unsupported = [
        label
        for label, values, allowed in (
            ("images", media.images, capabilities.images),
            ("videos", media.videos, capabilities.videos),
            ("audio", media.audios, capabilities.audios),
        )
        if values and not allowed
    ]
    if unsupported:
        raise ValueError(f"Model {model!r} does not support reference {', '.join(unsupported)}.")
    if capabilities.requires_images and not media.images:
        raise ValueError(f"Model {model!r} requires at least one reference image.")
    if capabilities.requires_videos and not media.videos:
        raise ValueError(f"Model {model!r} requires one or two reference videos.")
    _validate_media_limits(
        model,
        media,
        capabilities.max_images if capabilities.max_images is not None else len(media.images),
        capabilities.max_videos if capabilities.max_videos is not None else len(media.videos),
        capabilities.max_audios if capabilities.max_audios is not None else len(media.audios),
    )
    if mode is None:
        return
    if mode not in capabilities.modes:
        raise ValueError(f"Model {model!r} does not support generation_mode {mode!r}.")
    if mode == "text" and (media.images or media.videos or media.audios):
        raise ValueError("generation_mode 'text' cannot include reference media.")
    if mode == "first_frame" and len(media.images) != 1:
        raise ValueError("generation_mode 'first_frame' requires exactly one reference image.")
    if mode == "first_last" and len(media.images) != 2:
        raise ValueError("generation_mode 'first_last' requires exactly two reference images.")
    if mode == "first_frame_reference" and not media.images:
        raise ValueError("generation_mode 'first_frame_reference' requires at least one reference image.")


class Grok1RequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "reference")
        _validate_limits(context, 7)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        else:
            _require_images(context, 1, 7, "reference")
        _require_no_video_or_audio(context, context.mode)
        duration = _duration(context.duration_seconds, 6, 6, 10, context.model)
        payload: dict[str, Any] = {
            "model": context.model,
            "prompt": context.prompt,
            "duration": duration,
            "video_length": duration,
            "aspect_ratio": context.aspect_ratio,
            "resolution": "HD",
            "video_config": {"video_length": duration, "aspect_ratio": context.aspect_ratio, "resolution": "HD", "preset": "normal"},
        }
        if context.mode == "first_frame" and context.media.images:
            payload["image"] = context.media.images[0]
        elif context.media.images:
            _set_list(payload, "reference_images", context.media.images, 7)
        return _plan("/v1/videos", payload)


class Grok15RequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "reference")
        _validate_limits(context, 7)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        else:
            _require_images(context, 1, 7, "reference")
        _require_no_video_or_audio(context, context.mode)
        duration = _duration(context.duration_seconds, 6, 6, 15, context.model)
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "seconds": str(duration), "size": _size(context.aspect_ratio)}
        if context.media.images:
            payload["reference_images"] = [{"url": value} for value in context.media.images[:7]]
        return _plan("/v1/videos", payload)


class HappyHorseRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        expected_mode = (
            "text"
            if "-t2v-" in context.model.casefold()
            else "first_frame"
            if "-i2v-" in context.model.casefold()
            else "reference"
            if "-r2v-" in context.model.casefold()
            else ""
        )
        if not expected_mode:
            raise ValueError(f"HappyHorse model {context.model!r} must contain -t2v-, -i2v-, or -r2v-.")
        if context.mode != expected_mode:
            raise ValueError(f"HappyHorse model {context.model!r} requires generation_mode {expected_mode!r}.")
        _validate_limits(context, 9)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        else:
            _require_images(context, 1, 9, "reference")
        _require_no_video_or_audio(context, context.mode)
        duration = _duration(context.duration_seconds, 4, 4, 15, context.model)
        parameters: dict[str, Any] = {"duration": duration, "resolution": _resolution(context.model), "watermark": False}
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "parameters": parameters}
        if context.mode != "first_frame":
            parameters["ratio"] = context.aspect_ratio
        if context.mode == "first_frame" and context.media.images:
            payload["image_url"] = context.media.images[0]
        elif context.media.images:
            _set_list(payload, "reference_images", context.media.images, 9)
        return _plan("/v1/videos", payload)


class H3RequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "reference")
        _validate_limits(context, 9, 0, 3)
        if context.mode == "text":
            _require_no_media(context)
        else:
            _require_reference(context)
            if context.media.audios and not context.media.images:
                raise ValueError(f"Model {context.model!r} requires a reference image when using reference audio.")
        duration = _duration(context.duration_seconds, 5, 5, 15, context.model)
        payload = {
            "model": context.model,
            "content": _content(context.prompt, context.media.images, (), context.media.audios),
            "duration": duration,
            "aspect_ratio": context.aspect_ratio,
        }
        return _plan("/v1/video/generations", payload)


class OfficialSd25RequestBuilder(RequestBuilder):
    _maximum_duration = {"official_sd25": 30, "official_sd20mini": 15}

    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "first_frame", "first_last", "reference")
        image_limit = 30 if context.descriptor.family == "official_sd25" else 9
        video_limit = 10 if context.descriptor.family == "official_sd25" else 3
        audio_limit = 10 if context.descriptor.family == "official_sd25" else 3
        _validate_limits(context, image_limit, video_limit, audio_limit)
        if context.mode in {"first_frame", "first_last"}:
            _require_no_video_or_audio(context, context.mode)
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, self._maximum_duration[context.descriptor.family], context.model)
        payload: dict[str, Any] = {
            "model": context.model,
            "content": _content(
                context.prompt,
                context.media.images,
                context.media.videos,
                context.media.audios,
                image_roles=_image_roles(context.mode, context.media.images),
                include_reference_media_roles=True,
            ),
            "generate_audio": True,
            "duration": duration,
            "watermark": False,
            "resolution": _resolution(context.model),
        }
        if context.mode == "reference":
            payload["ratio"] = context.aspect_ratio
        return _plan("/v1/videos", payload)


class OfficialSd20HighRequestBuilder(RequestBuilder):
    _mode_values = {"text": "t2v", "first_frame": "first_frame", "first_last": "first_last", "reference": "reference"}

    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference")
        _validate_limits(context, 9, 3, 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode in {"first_frame", "first_last"}:
            _require_no_video_or_audio(context, context.mode)
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, 15, context.model)
        payload = {
            "model": context.model,
            "content": _content(
                context.prompt,
                context.media.images,
                context.media.videos,
                context.media.audios,
                image_roles=_image_roles(context.mode, context.media.images),
                include_reference_media_roles=False,
            ),
            "mode": self._mode_values[context.mode],
            "duration": duration,
            "ratio": context.aspect_ratio,
            "watermark": False,
            "generate_audio": True,
        }
        return _plan("/v1/video/generations", payload)


class MinimaxH3RequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference")
        _validate_limits(context, 9, 3, 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode in {"first_frame", "first_last"}:
            _require_no_video_or_audio(context, context.mode)
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 5, 5, 15, context.model)
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "aspect_ratio": context.aspect_ratio, "duration": duration, "fps": 24}
        if context.media.images:
            roles = _image_roles(context.mode, context.media.images)
            payload["reference_images"] = [{"url": value, "role": role} for value, role in zip(context.media.images, roles, strict=True)]
        if context.media.videos:
            payload["reference_videos"] = [{"url": value} for value in context.media.videos]
        if context.media.audios:
            payload["reference_audios"] = [{"url": value} for value in context.media.audios]
        return _plan("/v1/video/generations", payload)


class Sd25LowRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference", "first_frame_reference")
        _validate_limits(context, 30, 10, 10)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
        elif context.mode == "first_frame_reference":
            _require_images(context, 1, 30, "first_frame_reference")
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, 29, context.model)
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "duration": duration, "aspect_ratio": context.aspect_ratio}
        _set_list(payload, "images", context.media.images)
        _set_list(payload, "videos", context.media.videos)
        _set_list(payload, "audios", context.media.audios)
        return _plan("/v1/videos", payload)


class Sd20LowRequestBuilder(RequestBuilder):
    _mode_values = {"text": "text2video", "first_frame": "image2video", "first_last": "frames2video", "reference": "image2video", "first_frame_reference": "image2video"}

    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference", "first_frame_reference")
        _validate_limits(context, 9, 3, 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
        elif context.mode == "first_frame_reference":
            _require_images(context, 1, 9, "first_frame_reference")
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, 15, context.model)
        payload: dict[str, Any] = {
            "model": context.model,
            "prompt": context.prompt,
            "duration": duration,
            "metadata": {"ratio": context.aspect_ratio, "enableSound": "on", "modeType": self._mode_values[context.mode]},
        }
        _set_list(payload, "images", context.media.images)
        _set_list(payload, "videos", context.media.videos)
        _set_list(payload, "audios", context.media.audios)
        return _plan("/v1/videos", payload)


class Sd20AdRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference")
        _validate_limits(context, 9, 3, 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
        else:
            _require_reference(context)
        media = [
            {"type": media_type, "url": value}
            for media_type, values, limit in (("reference_image", context.media.images, 9), ("reference_video", context.media.videos, 3), ("reference_audio", context.media.audios, 3))
            for value in values
        ]
        payload = {"model": context.model, "prompt": context.prompt, "input": {"prompt": context.prompt, "media": media}, "seconds": "15", "size": _size(context.aspect_ratio)}
        return _plan("/v1/videos", payload)


class SplitReferenceRequestBuilder(RequestBuilder):
    def __init__(self, add_idempotency_key: bool = False) -> None:
        self._add_idempotency_key = add_idempotency_key

    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference")
        _validate_limits(context, 9, 3, 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
            _require_no_video_or_audio(context, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
            _require_no_video_or_audio(context, "first_last")
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, 15, context.model)
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "duration": duration, "resolution": _resolution(context.model), "aspect_ratio": context.aspect_ratio}
        headers = {"Idempotency-Key": f"plugin-{uuid.uuid4()}"} if self._add_idempotency_key else {}
        if context.mode in {"text", "first_frame"}:
            if context.media.images:
                payload["input_reference"] = {"image_url": context.media.images[0]}
            return _plan("/v1/videos", payload, headers)
        _set_list(payload, "image_references", context.media.images)
        _set_list(payload, "video_references", context.media.videos)
        _set_list(payload, "audio_references", context.media.audios)
        return _plan("/v1/video/generations", payload, headers)


class SdRotateRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "text", "first_frame", "first_last", "reference", "first_frame_reference")
        rotating_sd25 = "sd-2.5-轮换" in context.model.casefold() or "sd轮换" in context.model.casefold()
        _validate_limits(context, 30 if rotating_sd25 else 9, 10 if rotating_sd25 else 3, 10 if rotating_sd25 else 3)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
        elif context.mode == "first_frame_reference":
            _require_images(context, 1, 30 if rotating_sd25 else 9, "first_frame_reference")
        else:
            _require_reference(context)
        duration = _duration(context.duration_seconds, 4, 4, 29, context.model)
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "aspect_ratio": context.aspect_ratio, "seconds": str(duration), "resolution": "720p"}
        if context.mode == "first_frame" and context.media.images:
            payload["first_frame_url"] = context.media.images[0]
        elif context.mode == "first_last":
            payload["first_frame_url"] = context.media.images[0]
            payload["last_frame_url"] = context.media.images[1]
        elif context.mode == "first_frame_reference":
            payload["first_frame_url"] = context.media.images[0]
            _set_list(payload, "reference_image_urls", context.media.images[1:])
        elif context.media.images:
            _set_list(payload, "reference_image_urls", context.media.images)
        _set_list(payload, "reference_videos", context.media.videos)
        _set_list(payload, "reference_audios", context.media.audios)
        return _plan("/v1/videos", payload)


class Sd900RequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        _require_mode(context, "reference")
        _validate_limits(context, 9)
        _require_images(context, 1, 9, "reference")
        payload = {
            "model": context.model,
            "prompt": context.prompt,
            "duration": "15",
            "aspect_ratio": context.aspect_ratio,
            "resolution": "720p",
            "reference_images": [{"url": value} for value in context.media.images],
        }
        return _plan("/v1/videos", payload)


class OmniRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        payload: dict[str, Any] = {"model": context.model, "prompt": context.prompt, "aspect_ratio": context.aspect_ratio, "seconds": "10"}
        if context.descriptor.capabilities.requires_videos:
            _require_mode(context, "reference")
            _validate_limits(context, 0, 2)
            if len(context.media.videos) == 1:
                payload["video_url"] = context.media.videos[0]
            elif len(context.media.videos) == 2:
                _set_list(payload, "videos", context.media.videos)
            else:
                raise ValueError("Omni video editing requires one or two reference videos.")
            return _plan("/v1/videos", payload)
        _require_mode(context, "text", "first_frame", "first_last", "reference")
        _validate_limits(context, 5)
        if context.mode == "text":
            _require_no_media(context)
        elif context.mode == "first_frame":
            _require_images(context, 1, 1, "first_frame")
        elif context.mode == "first_last":
            _require_images(context, 2, 2, "first_last")
        else:
            _require_images(context, 1, 5, "reference")
        if context.mode == "first_frame":
            payload["first_image_url"] = context.media.images[0]
        elif context.mode == "first_last":
            payload["first_image_url"] = context.media.images[0]
            payload["last_image_url"] = context.media.images[1]
        elif context.media.images:
            _set_list(payload, "images", context.media.images, 5)
        return _plan("/v1/videos", payload)


class GenericRequestBuilder(RequestBuilder):
    def build(self, context: BuildContext) -> RequestPlan:
        payload = {"model": context.model, "prompt": context.prompt, "duration": context.duration_seconds or 5, "aspect_ratio": context.aspect_ratio}
        return _plan("/v1/videos", payload)


REQUEST_BUILDERS: dict[str, RequestBuilder] = {
    "grok1": Grok1RequestBuilder(), "grok15": Grok15RequestBuilder(), "happyhorse": HappyHorseRequestBuilder(),
    "h3": H3RequestBuilder(), "official_sd25": OfficialSd25RequestBuilder(), "official_sd20_high": OfficialSd20HighRequestBuilder(),
    "minimax_h3": MinimaxH3RequestBuilder(), "sd25_low": Sd25LowRequestBuilder(), "sd20_low": Sd20LowRequestBuilder(),
    "sd20_ad": Sd20AdRequestBuilder(), "sd933": SplitReferenceRequestBuilder(), "sd_rotate": SdRotateRequestBuilder(),
    "sd900": Sd900RequestBuilder(), "surprise": SplitReferenceRequestBuilder(add_idempotency_key=True), "omni": OmniRequestBuilder(),
    "generic": GenericRequestBuilder(),
}


def build_plan(
    descriptor: ModelDescriptor,
    model: str,
    prompt: str,
    duration_seconds: int,
    aspect_ratio: str,
    generation_mode: str,
    media: MediaReferences,
) -> RequestPlan:
    mode = _mode(generation_mode, media.images, media.videos, media.audios)
    if (
        not generation_mode.strip()
        and mode not in descriptor.capabilities.modes
        and "reference" in descriptor.capabilities.modes
        and (media.images or media.videos or media.audios)
    ):
        mode = "reference"
    validate_media(descriptor.capabilities, model, media, mode)
    context = BuildContext(
        descriptor,
        model,
        prompt,
        duration_seconds,
        aspect_ratio.strip() or "16:9",
        mode,
        media,
    )
    plan = REQUEST_BUILDERS[descriptor.builder_id].build(context)
    return plan
