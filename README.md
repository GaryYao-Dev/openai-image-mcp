# OpenAI Image MCP

An stdio MCP server that sends image-generation and image-edit requests to an
OpenAI-compatible Images API.

## Portable runtime

The plugin runs with `uv`; it does not distribute or depend on a local
`.venv`, a fixed Python installation, or a developer-specific path. On the
first launch, `uv` downloads a compatible Python when needed and creates a
cached, isolated environment for `scripts/server.py`.

Install `uv` once before enabling the plugin. The included scripts are safe to
run repeatedly: if `uv` is already available, they only verify it.

### Windows

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\install.ps1
```

### macOS and Linux

```sh
sh ./scripts/install.sh
```

The scripts use uv's official, pinned `0.11.23` installer only when `uv` is
missing. Restart Codex after installation so it inherits the updated `PATH`,
then enable or restart this MCP. The first MCP launch needs network access to
obtain Python and the locked dependencies.

## API configuration

Create a local `.env` file beside this README from `.env.example`, then set
`OPENAI_API_KEY` and any provider-specific values. `.env` is intentionally
ignored by Git and is never needed by the installer.

Images default to `~/Pictures/Codex Generated Images`. Set
`OPENAI_IMAGE_OUTPUT_DIR` in `.env` to use another directory.

## Diagnostics

After `uv` is installed, this command validates the Python runtime and MCP
imports without sending an image request:

```sh
uv run --locked --script scripts/server.py --self-check
```

The committed `scripts/server.py.lock` pins the full dependency resolution.
Update it deliberately with `uv lock --script scripts/server.py` after
changing the dependency metadata.
