"""bike-planner-mcp — stdio MCP server exposing bike-tour planning
tools to Claude Desktop / Claude Code / any MCP client.

Entry point: `mcp_bike_planner.server:main` (also installed as the
`bike-planner-mcp` console script).

Architecture in one sentence: the same tool schemas the FastAPI web
chat uses are re-exported via MCP's stdio transport; a small
Dispatcher decides for each call whether to run the tool in-process
(LocalPython backend) or forward to a running api container (HttpForward
backend), based on region config in regions.toml. Today the default
region is `local`; adding a second regional shard later is a five-line
TOML edit — the LLM's view of the tools never changes.
"""

from .dispatcher import Backend, Dispatcher, HttpForward, LocalPython

__all__ = ["Backend", "Dispatcher", "HttpForward", "LocalPython"]
