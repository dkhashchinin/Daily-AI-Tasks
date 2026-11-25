"""Sync facade over a long-lived MCP stdio session (mcp-cbr-rates).

The Day 6 agent is fully synchronous (Flask + sync OpenAI client) while the
MCP SDK is async, so the session lives in a daemon thread with its own event
loop. Public methods are plain sync calls.
"""
import asyncio
import threading
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

_SERVER_EXE = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "atomno-mcp-cbr-rates.exe"
_VENV_PYTHON = Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "python.exe"
# The uv trampoline exe can fail to canonicalize its script path (e.g. after the
# venv is moved), so prefer launching the installed module via the venv python.
SERVER_COMMAND = (
    [str(_VENV_PYTHON), "-m", "mcp_cbr_rates"]
    if _VENV_PYTHON.exists()
    else [str(_SERVER_EXE)]
)

_CALL_TIMEOUT = 60.0
_START_TIMEOUT = 30.0


def _clean_schema_node(node):
    """Drop keys an OpenAI-compatible endpoint may reject and unwrap `anyOf` nulls."""
    if isinstance(node, list):
        return [_clean_schema_node(item) for item in node]
    if not isinstance(node, dict):
        return node
    if "anyOf" in node:
        variants = [v for v in node["anyOf"] if v.get("type") != "null"]
        merged = {k: v for k, v in node.items() if k not in ("anyOf", "title", "default")}
        if len(variants) == 1:
            merged.update(_clean_schema_node(variants[0]))
            return merged
        merged["anyOf"] = _clean_schema_node(variants) or [{"type": "string"}]
        return merged
    return {k: _clean_schema_node(v) for k, v in node.items() if k not in ("title", "default")}


def _result_to_text(result) -> str:
    """Flatten an MCP CallToolResult into the string the model receives."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if result.isError:
        return f"Ошибка MCP-инструмента: {text}"
    return text


class McpBridge:
    """Keeps one MCP session alive in a background thread and exposes sync calls."""

    def __init__(self, command: list[str] | None = None):
        self._command = list(command) if command else list(SERVER_COMMAND)
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._error: Exception | None = None
        self._tools: list = []
        self._server_info = None
        self._stdio_cm = None
        self._session_cm = None
        self._session: ClientSession | None = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="mcp-bridge")
        self._thread.start()
        if not self._ready.wait(_START_TIMEOUT):
            self.close()
            raise RuntimeError("MCP connection timed out")
        if self._error:
            raise RuntimeError(f"MCP connection failed: {self._error}") from self._error

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._connect())
        if self._session is not None:
            self._loop.run_forever()

    async def _connect(self) -> None:
        try:
            self._stdio_cm = stdio_client(StdioServerParameters(
                command=self._command[0], args=self._command[1:],
            ))
            read, write = await self._stdio_cm.__aenter__()
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            info = await self._session.initialize()
            self._server_info = info
            self._tools = (await self._session.list_tools()).tools
        except Exception as exc:
            self._error = exc
        finally:
            self._ready.set()

    def _submit(self, coro):
        if self._session is None:
            raise RuntimeError("MCP session is not running")
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(_CALL_TIMEOUT)

    @property
    def tool_names(self) -> list[str]:
        return [t.name for t in self._tools]

    def has_tool(self, name: str) -> bool:
        return any(t.name == name for t in self._tools)

    def get_tool_schemas(self) -> list[dict]:
        """MCP tools -> OpenAI function-calling `tools` format."""
        schemas = []
        for tool in self._tools:
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": _clean_schema_node(dict(tool.inputSchema or {})),
                },
            })
        return schemas

    def call_tool(self, name: str, args: dict) -> str:
        result = self._submit(self._session.call_tool(name, args or {}))
        return _result_to_text(result)

    def get_status(self) -> dict:
        """Connection status, server metadata and tool list for the web UI."""
        info = self._server_info
        return {
            "connected": self._session is not None,
            "server_name": getattr(getattr(info, "serverInfo", None), "name", None),
            "server_version": getattr(getattr(info, "serverInfo", None), "version", None),
            "protocol_version": getattr(info, "protocolVersion", None),
            "command": " ".join(self._command),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": _clean_schema_node(dict(t.inputSchema or {})),
                }
                for t in self._tools
            ],
        }

    def close(self) -> None:
        """Shut the session and the server subprocess down."""
        session, session_cm, stdio_cm, loop = (
            self._session, self._session_cm, self._stdio_cm, self._loop,
        )
        self._session = None
        if loop is None or loop.is_closed():
            return

        async def _shutdown():
            for cm in (session_cm, stdio_cm):
                if cm is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass

        if loop.is_running():
            asyncio.run_coroutine_threadsafe(_shutdown(), loop).result(10.0)
            loop.call_soon_threadsafe(loop.stop)
        else:
            loop.run_until_complete(_shutdown())
        loop.close()

    def __enter__(self) -> "McpBridge":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


_bridge: McpBridge | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> McpBridge:
    """Process-wide singleton; spawns the MCP server on first use."""
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = McpBridge()
        return _bridge
