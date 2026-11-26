"""Sync facade over long-lived MCP stdio sessions (one per server).

The Day 6 agent is fully synchronous (Flask + sync OpenAI client) while the
MCP SDK is async, so each session lives in a daemon thread with its own event
loop. Public methods are plain sync calls. McpHub aggregates all configured
servers (SERVER_REGISTRY) and routes tool calls to the right session.
"""
import asyncio
import threading
import time
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


def _server_command(module: str) -> list[str]:
    """Launch command for an MCP server module installed in the project venv."""
    if _VENV_PYTHON.exists():
        return [str(_VENV_PYTHON), "-m", module]
    return list(SERVER_COMMAND)


# All MCP servers the agent connects to, name -> launch command.
SERVER_REGISTRY: dict[str, list[str]] = {
    "cbr-rates": SERVER_COMMAND,
    "tbank-invest": _server_command("mcp_tbank_invest"),
    # rates-store: данные -> сохранение -> чтение за период;
    # его дёргает планировщик (app/scheduler.py), агенту он тоже доступен.
    "rates-store": _server_command("mcp_rates_store"),
}

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


def _result_is_error(result) -> bool:
    """Флаг ошибки результата: `isError` в mcp 1.x, `is_error` в mcp 2.x."""
    return bool(getattr(result, "isError", None) or getattr(result, "is_error", False))


def _result_to_text(result) -> str:
    """Flatten an MCP CallToolResult into the string the model receives."""
    parts = []
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    text = "\n".join(parts).strip()
    if _result_is_error(result):
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
            # The loop thread needs a moment to process the stop callback.
            deadline = time.monotonic() + 5.0
            while loop.is_running() and time.monotonic() < deadline:
                time.sleep(0.05)
        else:
            loop.run_until_complete(_shutdown())
        loop.close()

    def __enter__(self) -> "McpBridge":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class McpHub:
    """Owns one McpBridge per registered server and routes tools by name.

    A server that fails to start does not kill the hub: it is recorded in
    _errors, its tools are simply missing from the merged list.
    """

    def __init__(self, registry: dict[str, list[str]] | None = None):
        self._registry = dict(registry or SERVER_REGISTRY)
        self._bridges: dict[str, McpBridge] = {}
        self._errors: dict[str, str] = {}
        for name, command in self._registry.items():
            try:
                self._bridges[name] = McpBridge(command)
            except Exception as exc:
                self._errors[name] = str(exc)
                print(f"[mcp] server {name} unavailable: {exc}")

    def _owner(self, name: str) -> McpBridge | None:
        for bridge in self._bridges.values():
            if bridge.has_tool(name):
                return bridge
        return None

    @property
    def tool_names(self) -> list[str]:
        seen: set[str] = set()
        names: list[str] = []
        for bridge in self._bridges.values():
            for name in bridge.tool_names:
                if name not in seen:
                    seen.add(name)
                    names.append(name)
        return names

    def has_tool(self, name: str) -> bool:
        return self._owner(name) is not None

    def get_tool_schemas(self) -> list[dict]:
        schemas: list[dict] = []
        seen: set[str] = set()
        for bridge in self._bridges.values():
            for schema in bridge.get_tool_schemas():
                fname = schema["function"]["name"]
                if fname in seen:
                    print(f"[mcp] duplicate tool {fname!r} ignored")
                    continue
                seen.add(fname)
                schemas.append(schema)
        return schemas

    def call_tool(self, name: str, args: dict) -> str:
        bridge = self._owner(name)
        if bridge is None:
            raise KeyError(f"no MCP server provides tool {name!r}")
        return bridge.call_tool(name, args)

    def get_statuses(self) -> list[dict]:
        statuses = [b.get_status() for b in self._bridges.values()]
        for name, error in self._errors.items():
            statuses.append({
                "connected": False,
                "server_name": name,
                "server_version": None,
                "protocol_version": None,
                "command": " ".join(self._registry.get(name, [])),
                "tools": [],
                "error": error,
            })
        return statuses

    def close(self) -> None:
        for bridge in self._bridges.values():
            bridge.close()
        self._bridges.clear()


_bridge: McpHub | None = None
_bridge_lock = threading.Lock()


def get_bridge() -> McpHub:
    """Process-wide singleton; spawns all MCP servers on first use."""
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = McpHub()
        return _bridge


def close_bridge() -> None:
    """Shut every MCP session down (used on Flask shutdown)."""
    global _bridge
    with _bridge_lock:
        if _bridge is not None:
            _bridge.close()
            _bridge = None
