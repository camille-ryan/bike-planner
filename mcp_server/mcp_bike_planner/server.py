"""Stdio MCP server entry point.

Wires the shared bike-planner tools (schemas from
``api.app.tools.TOOLS``) into the ``mcp`` SDK's stdio server, and
delegates every ``call_tool`` to the Dispatcher configured from
``regions.toml``.

Run this via the console script installed by the package:

    bike-planner-mcp

or directly:

    python -m mcp_bike_planner.server

Add it to Claude Desktop's ``claude_desktop_config.json``:

.. code-block:: json

    {
      "mcpServers": {
        "bike-planner": {
          "command": "bike-planner-mcp"
        }
      }
    }

Then restart Claude Desktop. The tools appear as bike-planner_route,
bike-planner_search_anchors, etc.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from .dispatcher import load_dispatcher


def _load_tool_schemas() -> list[dict]:
    """Import the shared tool schemas lazily so this module is
    still importable in test contexts where api/ isn't installed
    (e.g. CI running just the dispatcher tests)."""
    try:
        from api.app.tools import TOOLS
    except ImportError as exc:  # pragma: no cover
        print(
            "[bike-planner-mcp] FATAL: could not import api.app.tools — "
            "make sure the repo root is on PYTHONPATH.\n"
            f"  reason: {exc}",
            file=sys.stderr,
        )
        raise
    return list(TOOLS)


log = logging.getLogger("bike-planner-mcp")


def _mcp_tool_of(schema: dict) -> Tool:
    """Convert one of our JSON-schema tool descriptors into MCP's Tool
    shape. MCP wants `inputSchema` (camelCase) while ours ships as
    `input_schema` (snake) to match the Anthropic messages API."""
    return Tool(
        name=schema["name"],
        description=schema["description"],
        inputSchema=schema["input_schema"],
    )


def build_server(config_path: Path | None = None) -> Server:
    """Construct and wire the MCP Server. Kept separate from `main`
    so tests can build one without spinning up stdio."""
    server: Server = Server("bike-planner")
    dispatcher = load_dispatcher(config_path)
    tool_schemas = _load_tool_schemas()

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [_mcp_tool_of(t) for t in tool_schemas]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        log.info("tool call: %s(%s)", name, arguments)
        result = await dispatcher.call(name, arguments or {})
        # MCP encodes tool results as content blocks. TextContent with
        # a JSON body is the maximally-portable option and matches the
        # shape our other frontends already handle.
        return [TextContent(type="text", text=json.dumps(result))]

    # Attach the dispatcher so main() can close it cleanly on shutdown.
    server._bike_planner_dispatcher = dispatcher  # type: ignore[attr-defined]
    return server


async def _run(server: Server) -> None:
    async with stdio_server() as (read, write):
        await server.run(
            read, write,
            server.create_initialization_options(),
        )


def main() -> None:
    """Console-script entry. Sets up minimal logging to stderr — stdout
    is reserved for the MCP JSON-RPC frames the client parses."""
    logging.basicConfig(
        level=os.environ.get("MCP_LOG_LEVEL", "INFO"),
        stream=sys.stderr,
        format="[%(name)s] %(levelname)s %(message)s",
    )
    server = build_server()
    try:
        asyncio.run(_run(server))
    except KeyboardInterrupt:
        pass
    finally:
        dispatcher = getattr(server, "_bike_planner_dispatcher", None)
        if dispatcher is not None:
            asyncio.run(dispatcher.aclose())


if __name__ == "__main__":
    main()
