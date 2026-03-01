# Aurora Grimoire

CLI + MCP server for Aurora OS documentation RAG.

The project covers the full pipeline:
- fetch HTML documentation;
- normalize into Markdown;
- chunk content;
- generate embeddings;
- upload vectors to Qdrant;
- search with sources (`cli search_docs`);
- expose search through MCP tool `search_docs`.

## Features

- Full offline-friendly RAG pipeline via CLI.
- Hybrid retrieval: dense (Qdrant) + BM25.
- Documentation version filtering (`--doc-version`).
- Knowledge sufficiency threshold (`--knowledge-threshold`).
- Artifact transfer between machines via bundle.
- MCP server (`stdio` and HTTP transport).

## Requirements

- Rust/Cargo (to install CLI via `cargo install --path .`).
- For `rag embed`: Ollama with an embedding model.
- For `rag deploy` and search: Qdrant.

For a local dev stack:
- `rag dev up` starts Docker services (Qdrant + Ollama).

## Installation

Install the CLI binary:

```bash
cargo install --path .
```

Check installation:

```bash
aurora-grimoire --version
```

If command is not found, ensure Cargo bin is in `PATH`:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

## Config Is Required

All commands except `rag config init` require an existing `config.toml`.

Create config:

```bash
aurora-grimoire rag config init
```

Overwrite existing config:

```bash
aurora-grimoire rag config init --force
```

Custom config path:

```bash
AURORA_GRIMOIRE_CONFIG=/path/to/config.toml aurora-grimoire rag config init
```

Config template: [`docs/config.example.toml`](docs/config.example.toml)

## Quick Start (end-to-end)

1) Initialize config:

```bash
aurora-grimoire rag config init
```

2) Start local stack:

```bash
aurora-grimoire rag dev up --build --gpu --verbose
```

3) Fetch and prepare data:

```bash
aurora-grimoire rag fetch-web -v
aurora-grimoire rag struct -v
aurora-grimoire rag chunk -v
```

4) Generate embeddings and deploy to Qdrant:

```bash
aurora-grimoire rag embed -v
aurora-grimoire rag deploy -v --recreate
```

5) Validate search:

```bash
aurora-grimoire cli search_docs "how to build project with mb2" --doc-version 5.2.0 --json
```

## Transfer Artifacts To Another User

`rag bundle create` always creates a `dual` bundle (`chunks + vectors`).

Create bundle:

```bash
aurora-grimoire rag bundle create --out ~/.aurora-grimoire/bundles/aurora-dual.tar.zst
```

Inspect bundle:

```bash
aurora-grimoire rag bundle inspect --file ~/.aurora-grimoire/bundles/aurora-dual.tar.zst
```

On another machine:

```bash
aurora-grimoire rag config init
aurora-grimoire rag bundle extract --file /path/aurora-dual.tar.zst
aurora-grimoire rag deploy --input ~/.aurora-grimoire/vectors_data --recreate -v
```

## MCP

Start MCP server:

```bash
aurora-grimoire mcp start --stdio
```

or HTTP:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
```

MCP smoke test:

```bash
aurora-grimoire mcp smoke --doc-version 5.2.0 --query "how to build project with mb2"
```

MCP exposes one tool:
- `search_docs`

## Agent Integrations

Ensure `aurora-grimoire` is installed and available in `PATH`.
If you need absolute binary path:

```bash
AURORA_GRIMOIRE_BIN="$(command -v aurora-grimoire)"
```

Use that absolute path in examples below.

### Claude Code

Stdio transport:

```bash
claude mcp add aurora-grimoire -- $AURORA_GRIMOIRE_BIN mcp start --stdio
```

HTTP transport:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
claude mcp add --transport http aurora-grimoire-http http://127.0.0.1:8080/mcp
```

### Codex

Stdio transport:

```bash
codex mcp add aurora-grimoire -- $AURORA_GRIMOIRE_BIN mcp start --stdio
```

HTTP transport:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
codex mcp add aurora-grimoire-http --url http://127.0.0.1:8080/mcp
```

Check:

```bash
codex mcp list
```

### OpenCode

OpenCode adds MCP server via interactive wizard:

```bash
opencode mcp add
```

For local mode set:
- `name`: `aurora-grimoire`
- `transport/type`: `local`/`stdio`
- `command`: `$AURORA_GRIMOIRE_BIN`
- `args`: `mcp start --stdio`

For remote mode set:
- `transport/type`: `remote`/`http`
- `url`: `http://127.0.0.1:8080/mcp`

Check:

```bash
opencode mcp list
```

### Other MCP Clients

If your client supports stdio `mcpServers` config, use:

```json
{
  "mcpServers": {
    "aurora-grimoire": {
      "command": "$AURORA_GRIMOIRE_BIN",
      "args": ["mcp", "start", "--stdio"]
    }
  }
}
```

If your client supports streamable HTTP, start server:

```bash
aurora-grimoire mcp start --http --host 127.0.0.1 --port 8080
```

and connect URL `http://127.0.0.1:8080/mcp`.

## Install Agent Skill/Command Templates

CLI includes installer command:

```bash
aurora-grimoire agents install --runtime all --scope global --verbose
```

Flags:
- `--runtime <claude|opencode|codex|all>` (default: `all`)
- `--scope <global|local>` (default: `local`)
- `--config-dir <PATH>` (single runtime only)
- `--force` (overwrite existing files)

Installed targets:
- Claude Code: `commands/aurora/search-docs.md` (invoke: `/aurora:search-docs ...`)
- OpenCode: `command/aurora-search-docs.md` (invoke: `/aurora-search-docs ...`)
- Codex: `skills/aurora-search-docs/SKILL.md` (invoke: `$aurora-search-docs ...`)

## Main Commands

```text
aurora-grimoire
├─ agents
│  └─ install
├─ rag
│  ├─ config init
│  ├─ fetch-web
│  ├─ struct
│  ├─ chunk
│  ├─ embed
│  ├─ deploy
│  ├─ bundle (create/inspect/extract)
│  ├─ dev (up/down/status/logs)
│  ├─ test-e2e
│  └─ clear
├─ cli
│  └─ search_docs
└─ mcp
   ├─ start
   └─ smoke
```

Full and current command tree with flags: [`docs/architecture.md`](docs/architecture.md)

## Useful Commands

Check dev stack status:

```bash
aurora-grimoire rag dev status
```

Logs:

```bash
aurora-grimoire rag dev logs -f
```

Clear artifacts:

```bash
aurora-grimoire rag clear --all
```
