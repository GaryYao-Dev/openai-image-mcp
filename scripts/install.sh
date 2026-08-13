#!/bin/sh
set -eu

uv_version="0.11.23"

find_uv_path() {
    if command -v uv >/dev/null 2>&1; then
        command -v uv
        return 0
    fi

    install_directory=${UV_INSTALL_DIR:-"$HOME/.local/bin"}
    installed_path="$install_directory/uv"
    if [ -x "$installed_path" ]; then
        printf '%s\n' "$installed_path"
        return 0
    fi

    return 1
}

uv_path=$(find_uv_path || true)
if [ -n "$uv_path" ]; then
    "$uv_path" --version
    printf 'uv is already available: %s\n' "$uv_path"
    exit 0
fi

temp_base=${TMPDIR:-/tmp}
installer_path=$(mktemp "${temp_base%/}/openai-image-mcp-uv.XXXXXX")
trap 'rm -f "$installer_path"' 0

if command -v curl >/dev/null 2>&1; then
    curl -LsSf "https://astral.sh/uv/$uv_version/install.sh" -o "$installer_path"
elif command -v wget >/dev/null 2>&1; then
    wget -q -O "$installer_path" "https://astral.sh/uv/$uv_version/install.sh"
else
    printf '%s\n' "uv is missing and neither curl nor wget is available to download its official installer." >&2
    exit 1
fi

sh "$installer_path"

uv_path=$(find_uv_path || true)
if [ -z "$uv_path" ]; then
    printf '%s\n' "uv was installed but is not discoverable. Restart your shell, then run this script again." >&2
    exit 1
fi

"$uv_path" --version
printf 'uv installed: %s\n' "$uv_path"
printf '%s\n' "Restart Codex so it inherits the updated PATH, then enable or restart the OpenAI Image MCP."
