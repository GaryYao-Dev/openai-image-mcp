"""AICopy model catalog and capability contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelCapabilities:
    images: bool = True
    videos: bool = False
    audios: bool = False
    requires_images: bool = False
    requires_videos: bool = False
    modes: frozenset[str] = frozenset({"text", "first_frame", "first_last", "reference"})
    max_images: int | None = None
    max_videos: int | None = None
    max_audios: int | None = None


@dataclass(frozen=True)
class ModelRule:
    any_of: tuple[str, ...] = ()
    all_of: tuple[str, ...] = ()
    none_of: tuple[str, ...] = ()
    prefixes: tuple[str, ...] = ()

    def matches(self, model: str) -> bool:
        value = model.casefold()
        return (
            (not self.any_of or any(marker.casefold() in value for marker in self.any_of))
            and all(marker.casefold() in value for marker in self.all_of)
            and not any(marker.casefold() in value for marker in self.none_of)
            and (not self.prefixes or any(value.startswith(prefix.casefold()) for prefix in self.prefixes))
        )


@dataclass(frozen=True)
class ModelDescriptor:
    family: str
    builder_id: str
    rule: ModelRule
    capabilities: ModelCapabilities
    asset_encoding: str = "upload"


class ModelCatalog:
    def __init__(self, descriptors: tuple[ModelDescriptor, ...]) -> None:
        self._descriptors = descriptors

    def resolve(self, model: str) -> ModelDescriptor:
        for descriptor in self._descriptors:
            if descriptor.rule.matches(model):
                return descriptor
        raise RuntimeError("Model catalog must include a fallback descriptor.")


TEXT_ONLY = frozenset({"text"})
IMAGE_MODES = frozenset({"text", "first_frame", "reference"})
REFERENCE_MODES = frozenset({"text", "first_frame", "first_last", "reference"})
LOW_REFERENCE_MODES = frozenset({"text", "first_frame", "first_last", "reference", "first_frame_reference"})
FRAME_MODES = frozenset({"first_frame", "first_last", "reference"})
NO_MEDIA = ModelCapabilities(images=False, modes=TEXT_ONLY)
IMAGE_ONLY = ModelCapabilities(modes=IMAGE_MODES)
IMAGE_AND_AUDIO = ModelCapabilities(audios=True, modes=frozenset({"text", "reference"}), max_images=9, max_audios=3)
ALL_REFERENCES = ModelCapabilities(videos=True, audios=True, modes=LOW_REFERENCE_MODES)
OFFICIAL_SD25 = ModelCapabilities(
    videos=True,
    audios=True,
    modes=FRAME_MODES,
    max_images=30,
    max_videos=10,
    max_audios=10,
)
OFFICIAL_SD20_MINI = ModelCapabilities(
    videos=True,
    audios=True,
    modes=FRAME_MODES,
    max_images=9,
    max_videos=3,
    max_audios=3,
)
OFFICIAL_SD20_HIGH = ModelCapabilities(
    videos=True,
    audios=True,
    modes=REFERENCE_MODES,
    max_images=9,
    max_videos=3,
    max_audios=3,
)
MINIMAX_H3 = ModelCapabilities(
    videos=True,
    audios=True,
    modes=REFERENCE_MODES,
    max_images=9,
    max_videos=3,
    max_audios=3,
)
VIDEO_EDIT = ModelCapabilities(images=False, videos=True, requires_videos=True, modes=frozenset({"reference"}), max_videos=2)
SD900_REFERENCES = ModelCapabilities(requires_images=True, modes=frozenset({"reference"}), max_images=9)

MODEL_CATALOG = ModelCatalog(
    (
        ModelDescriptor("grok1", "grok1", ModelRule(any_of=("grok-imagine-1.0", "grok-1.0-")), IMAGE_ONLY, "data"),
        ModelDescriptor("grok15", "grok15", ModelRule(any_of=("grok-imagine-video-1.5", "grok-1.5-")), IMAGE_ONLY, "data"),
        ModelDescriptor("happyhorse", "happyhorse", ModelRule(prefixes=("happyhorse-",)), IMAGE_ONLY, "data"),
        ModelDescriptor("h3", "h3", ModelRule(all_of=("开源h3-", "按次")), IMAGE_AND_AUDIO),
        ModelDescriptor("h3", "h3", ModelRule(any_of=("官方h3-",)), IMAGE_AND_AUDIO),
        ModelDescriptor("official_sd25", "official_sd25", ModelRule(any_of=("【官方稳定版】2.5-",)), OFFICIAL_SD25),
        ModelDescriptor(
            "official_sd20mini",
            "official_sd25",
            ModelRule(all_of=("【官方稳定版】2.0-",), none_of=("720p-满血", "720p-fast")),
            OFFICIAL_SD20_MINI,
        ),
        ModelDescriptor("official_sd20_high", "official_sd20_high", ModelRule(any_of=("【官方稳定版】sd2.0-720p",)), OFFICIAL_SD20_HIGH),
        ModelDescriptor("minimax_h3", "minimax_h3", ModelRule(any_of=("开源h3-",)), MINIMAX_H3),
        ModelDescriptor("sd900", "sd900", ModelRule(any_of=("sd-720满血-900",)), SD900_REFERENCES),
        ModelDescriptor("sd933", "sd933", ModelRule(any_of=("933",)), ALL_REFERENCES),
        ModelDescriptor("surprise", "surprise", ModelRule(any_of=("惊喜渠道",)), ALL_REFERENCES),
        ModelDescriptor("omni", "omni", ModelRule(all_of=("omni-fast-视频", "编辑")), VIDEO_EDIT),
        ModelDescriptor("omni", "omni", ModelRule(any_of=("omni-fast-视频",)), IMAGE_ONLY),
        ModelDescriptor("sd20_ad", "sd20_ad", ModelRule(any_of=("ad渠道",)), ALL_REFERENCES),
        ModelDescriptor(
            "sd_rotate",
            "sd_rotate",
            ModelRule(
                any_of=(
                    "sd-2.5-轮换",
                    "sd轮换",
                    "sd-720fast-不卡脸",
                    "sd-720满血-不卡脸",
                    "sd-720满血-较慢",
                    "sd-720fast（按秒）",
                    "sd-720满血（按秒）",
                )
            ),
            ALL_REFERENCES,
        ),
        ModelDescriptor("sd20_low", "sd20_low", ModelRule(any_of=("sd2.0全系列", "sd2.0-")), ALL_REFERENCES),
        ModelDescriptor("sd25_low", "sd25_low", ModelRule(any_of=("sd-2.5-",)), ALL_REFERENCES),
        ModelDescriptor("generic", "generic", ModelRule(), NO_MEDIA),
    )
)


def model_family(model: str) -> str:
    return MODEL_CATALOG.resolve(model).family


def is_video_model(record: dict[str, Any]) -> bool:
    name = str(record.get("model_name", ""))
    endpoint_types = {str(item).casefold() for item in record.get("supported_endpoint_types", []) if isinstance(item, str)}
    return "openai-video" in endpoint_types or MODEL_CATALOG.resolve(name).family != "generic"


def is_image_model(record: dict[str, Any]) -> bool:
    """Identify image-generation entries without treating chat-only OpenAI models as images."""
    if is_video_model(record):
        return False
    name = str(record.get("model_name", "")).casefold()
    endpoint_types = {str(item).casefold() for item in record.get("supported_endpoint_types", []) if isinstance(item, str)}
    if "image-generation" in endpoint_types:
        return True
    return any(marker in name for marker in ("image", "图片", "seedream", "banana", "绘画", "画图"))


def supports_reference_video(model: str) -> bool:
    return MODEL_CATALOG.resolve(model).capabilities.videos


def supports_media(
    model: str,
    *,
    image_count: int = 0,
    video_count: int = 0,
    audio_count: int = 0,
    mode: str = "",
) -> bool:
    capabilities = MODEL_CATALOG.resolve(model).capabilities
    if image_count and not capabilities.images:
        return False
    if video_count and not capabilities.videos:
        return False
    if audio_count and not capabilities.audios:
        return False
    if mode and mode not in capabilities.modes:
        return False
    return all(
        count <= limit
        for count, limit in (
            (image_count, capabilities.max_images),
            (video_count, capabilities.max_videos),
            (audio_count, capabilities.max_audios),
        )
        if limit is not None
    )
