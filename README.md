# OpenAI Media MCP

An stdio MCP server for OpenAI-compatible image and video generation APIs. It
keeps the existing image tools and adds `generate_video`, while allowing
multiple providers and keys through named profiles.

## 配置

使用 `config.json` 配置多个图片/视频接口。每个 profile 只需要声明
`type`、`base_url`、`key`、`model`：

```powershell
Copy-Item config.example.json config.json
```

`config.json` 已加入 `.gitignore`，直接在本地填写 key。服务只从这个文件
读取供应商配置。

普通 OpenAI 兼容 profile 会根据调用类型固定选择 API 路径：

- `generate_image`: `/images/generations`
- `edit_images`: `/images/edits`
- `generate_video`: `/videos/generations`

如果图片生成和编辑需要不同的接口、key 或 model，可以各自建立 profile，
并分别设置 `defaults.image_generate` 与 `defaults.image_edit`。价格、分组、
配额等网站展示字段不参与请求。

`aicopy-image` 与 `aicopy-video` 是两个独立的 AICopy profile，分别填写图片分组和
视频分组的 `key`，不能共用。两者的 `model` 均可留空：模型目录由 `/api/pricing`
获取，实际生成时再传入所选模型。模型价格会写入项目根目录的
`pricing-cache.json`；缓存未满 24 小时直接复用，过期后才会重新获取。缓存不保存
API key，且已加入 `.gitignore`。

## 给 AI agent 的调用方式

1. 先调用 `list_media_profiles`，读取可用 profile、类型和模型。
2. 图片模型未确定时调用 `list_image_models`，或调用 `generate_image` 时不传
   `model`。返回会按供应商列出可选模型；AICopy 还会提供缓存价格和计费方式。选定后
   调用 `generate_image` 并传 `profile` 与 `model`。图片编辑仍使用 `edit_images`。
3. AICopy 视频未确定模型时，调用 `list_video_models`，或调用
   `generate_video` 时不传 `model`。两者都会返回可选模型和按时长估算的价格；
   `list_video_models` 可用 `reference_image`、`reference_video`、
   `reference_audio` 筛选素材能力，`generate_video` 传入对应的本地素材路径后也会
   自动仅保留能接收这些素材的模型。
4. 调用 `generate_video`，传 `prompt`、`model`、`duration_seconds`；可按需传
   `aspect_ratio`、`generation_mode` 和本地 `image_paths`、`video_paths`、
   `audio_paths`。有多个视频 profile 时传 `profile`，否则使用 `defaults.video`。

视频结果会保存到本地 `~/Videos/Codex Generated Videos`，工具返回绝对路径。图片仍会内联返回并
同时保存到本地。视频 API 可以返回直接二进制、`data[0].url`、
`data[0].b64_video` 或常见 base64 字段。

## Portable runtime

The plugin runs with `uv`; it does not distribute or depend on a local
`.venv` or a developer-specific Python installation.

### Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

### macOS and Linux

```sh
sh ./scripts/install.sh
```

Restart Codex after installing `uv`. The first MCP launch needs network access
to obtain Python and the locked dependencies.

## Diagnostics

```sh
uv run --locked --script scripts/server.py --self-check
```

The committed `scripts/server.py.lock` pins the dependency resolution. Update
it deliberately with `uv lock --script scripts/server.py` after changing the
dependency metadata.
