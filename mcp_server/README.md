# bike-planner-mcp

A stdio [Model Context Protocol](https://modelcontextprotocol.io/) server
exposing the bike-planner's tools to any MCP client — Claude Desktop,
Claude Code, Cursor, Continue, or anything else that speaks the spec.

The tools are the same ones the web chat uses (`route`,
`stations_along_route`, `pois_near_anchor`, `split_into_stages`, …),
imported from `api.app.tools`. **One tool implementation, two
frontends** — that's the point of the split.

## What MCP is (and why this project has one)

MCP is Anthropic's spec for how an LLM's host application (Claude
Desktop, an IDE plugin, …) discovers and calls **external tools** that
live outside the LLM provider's servers. The host launches an MCP
"server" — a subprocess speaking JSON-RPC over stdio — and asks it
"what tools do you have?" and "run this one with these args."

For this project it means Claude Desktop can plan a route by calling
the same `route(from_ref, to_ref, via_refs)` tool the web chat uses,
without the web UI or the FastAPI service needing to know Claude
Desktop exists.

## Architecture at a glance

```
Claude Desktop / Claude Code / Cursor
              │
              │  stdio JSON-RPC (MCP protocol)
              ▼
     ┌──────────────────────────┐
     │  bike-planner-mcp        │
     │  ─────────────────       │
     │  · MCP Server (stdio)    │
     │  · Dispatcher            │  ← routes each tool call to
     │  · Backend registry      │    the right backend based on
     └──────────────────────────┘    region config
              │
      ┌───────┴────────┐
      ▼                ▼
  LocalPython      HttpForward
  (in-process:     (POST /tools/<name>
   api.app.tools)   on a running api
                    container, one per
                    region)
```

The dispatcher is region-aware: for tools like `route` and
`stations_along_route`, it looks at the args' target coord and picks a
backend whose bbox covers it. Today the default region is a single
`local` backend running all four ingested countries (AT/DE/CZ/DK). To
add a second regional shard — say, US-southwest — you edit
`regions.toml`:

```toml
[backends.us_southwest]
kind = "http_forward"
url  = "http://us-southwest-api:8000"
bbox = [-114.0, 31.3, -102.0, 42.0]
```

Nothing else changes. Tool schemas stay the same, the LLM's system
prompt is unchanged, the web chat's code path doesn't move. The
dispatcher pattern makes tool interface independent of tool
implementation.

## Install

```bash
# From the repo root:
pip install -e mcp_server
```

The local-python backend reaches into `api/app/tools.py` at import
time, so make sure the `api/` package's deps are also installed:

```bash
pip install -r api/requirements.txt
```

The `bike-planner-mcp` command is now on your PATH.

## Wire it into Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "bike-planner": {
      "command": "bike-planner-mcp"
    }
  }
}
```

Restart Claude Desktop. In a new conversation, the bike-planner tools
appear in the tool picker. Type:

> Route Graz to Vienna, ~80 km/day, rail-accessible overnights.

Claude will call `search_anchors`, `route`, `split_into_stages`, and
`stations_along_route` end-to-end — same tools the web chat uses.

### With Claude Code

Same config file location differs but the schema is identical. Point
the `command` at wherever you installed the console script.

## Wire it into Claude Code (from this repo)

Claude Code reads `.claude/mcp.json` at the repo root. Add:

```json
{
  "mcpServers": {
    "bike-planner": {
      "command": "bike-planner-mcp"
    }
  }
}
```

## Environment

- `MCP_LOG_LEVEL` (default `INFO`) — set to `DEBUG` for verbose
  logging on stderr. Stdout is reserved for MCP JSON-RPC frames.
- `DEFAULT_PROFILE` (default `views`) — routing cost profile the
  LocalPython backend uses.
- `ANTHROPIC_API_KEY` — **not needed here**. The MCP server never
  talks to Anthropic directly; it only responds to a host that does.

## First tool call is slow (lazy import + lazy trunk load)

The LocalPython backend imports `api.app.tools` lazily on the first
call. That import wires up `trunk_router`, which now uses an
LRU-bounded lazy cache (`TrunkStore`) instead of preloading every
trunk. Boot cost is ~15 s; the FIRST tool call for a novel corridor
pays another 3–5 s while SQLite pages the needed blobs into memory.
Steady-state RSS is ~300 MB (was ~5 GB before the lazy-load
refactor — see commit `f2fa513`).

Tune with `TRUNK_CACHE_MAX_ENTRIES` (default 512) — higher keeps
more corridors hot in RAM at the cost of steady-state memory.

## Tests

```bash
cd mcp_server
pytest -q
```

Tests use fake in-memory backends — they exercise the dispatcher's
region routing logic without needing the api/ package's data loaded.

## Layout

```
mcp_server/
├── mcp_bike_planner/
│   ├── __init__.py
│   ├── server.py           # stdio entry, wires tools to dispatcher
│   ├── dispatcher.py       # Backend protocol + LocalPython + HttpForward
│   └── regions.toml        # per-region backend config
├── tests/
│   └── test_dispatcher.py  # unit tests for region routing
├── pyproject.toml
└── README.md
```
