# Cross-platform portable uv runtime

## Goal

Make the `openai-image` stdio MCP portable across Windows, macOS, and Linux.
An installed copy must not depend on a developer-specific absolute path, a
checked-in virtual environment, or a pre-installed Python interpreter.

## Chosen approach

Use `uv` as the single cross-platform launcher.  The MCP configuration will
start `uv` from `PATH`, set the plugin root as the working directory, and pass
only relative script paths.  The server script will contain PEP 723 inline
metadata that declares its required Python version and the pinned `mcp`
dependency.  `uv run --script` will create or reuse its per-user cached
environment, install the dependency, and download a compatible Python when
one is absent.

The first bootstrap is explicit: the plugin will ship an idempotent PowerShell
installer for Windows and a POSIX-shell installer for macOS and Linux.  Each
checks whether `uv` is already available, otherwise invokes uv's official
installer and tells the user to restart Codex so its process inherits the
updated `PATH`.  The MCP configuration will not silently download or execute
an installer.

This boundary is intentional.  An MCP command needs an executable before it
can bootstrap anything, and Codex requires users to review and trust plugin
hooks before hooks may run.  A single silent installer would also need
platform- and architecture-specific native bootstrap logic.

## Components

### MCP configuration

`.mcp.json` will replace all absolute paths with the following logical
configuration:

- command: `uv`
- arguments: run the relative `scripts/server.py` as a uv script
- working directory: `.` (the installed plugin root)
- an extended startup timeout for the first download and dependency sync

Codex resolves a plugin-relative working directory against the plugin root,
so the script argument is evaluated from a stable location after installation.
No `cwd`, Python executable, or script path will name a user directory.

### Runtime dependency declaration

`scripts/server.py` will receive PEP 723 metadata at the top of the file.
It will specify a supported Python range and a pinned `mcp` release compatible
with the current server imports.  `uv run --script` uses this metadata and its
normal user cache rather than creating or distributing `.venv` in the plugin
directory.

The existing `PLUGIN_ROOT = Path(__file__).resolve().parents[1]` behavior
remains the source of truth for locating `.env`; it does not rely on the
process current directory.

### Bootstrap installers

`scripts/install.ps1` and `scripts/install.sh` will:

1. Detect a usable `uv` executable and exit successfully when one exists.
2. Download and invoke only uv's official platform installer when it is
   absent and network access is available.
3. Verify `uv --version` after installation.
4. Print the exact next action: restart Codex, then enable or restart the MCP.

They will not create a virtual environment, inspect `.env`, or write absolute
paths into plugin files.  Users explicitly run the appropriate installer once
or use an equivalently trusted organization-managed uv installation.

### Documentation and examples

A concise README will document the Windows and macOS/Linux bootstrap commands,
the first-run network requirement, `.env` setup, and the `--self-check`
command.  `.env.example` will no longer present a machine-specific output
directory; it will describe the portable default based on the user's home
directory.

`.gitignore` will explicitly ignore local virtual environments so generated
runtime state is never published by accident.

## Startup flow

1. The user installs or confirms `uv` once with the operating-system-specific
   script and restarts Codex.
2. Codex launches the plugin's `uv` command from the plugin root.
3. On first use, uv acquires a compatible Python if necessary, resolves the
   declared `mcp` dependency, and creates a cached environment under its
   normal user-managed cache location.
4. uv executes `scripts/server.py`; the server resolves `.env` relative to its
   own file and starts stdio transport.
5. Later launches reuse uv's managed cache.  Plugin updates may create a new
   cache entry without changing any configuration path.

## Error handling

- If `uv` is not on `PATH`, Codex reports that the launcher cannot start; the
  README and installers provide the recovery command.
- If first-run downloads are offline or blocked, uv surfaces the failed
  download.  The plugin does not silently fall back to an unpinned system
  Python.
- If the API configuration is absent or invalid, the existing server validation
  remains responsible for reporting it; bootstrapping never reads or logs the
  API key.
- The installer exits non-zero on a failed uv installation and does not report
  success until `uv --version` is available.

## Validation

Implementation will verify:

1. Tracked launch files contain no developer-specific absolute paths.
2. A clean Windows invocation can run `uv run --script scripts/server.py
   --self-check` after bootstrap.
3. The PowerShell installer parses and is idempotent when `uv` is already
   installed.
4. The POSIX installer passes shell syntax validation and is idempotent when
   `uv` is already installed.
5. The source's home-directory output default and `.env` loading remain
   unchanged in behavior.

macOS and Linux execution cannot be run from this Windows workspace; their
shell script will be written to POSIX syntax and validated statically.  The
same documented `uv run --script` command is the runtime path on all three
operating systems.

## Out of scope

- Bundling uv or Python binaries for every OS and CPU architecture.
- A remote-hosted MCP replacement.
- Silent installation hooks or modification of Codex-wide configuration.
- Changes to image-generation behavior, API providers, or user secrets.
