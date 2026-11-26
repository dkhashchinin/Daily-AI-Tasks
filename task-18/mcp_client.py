"""Minimal MCP client: connects to mcp-cbr-rates over stdio and lists its tools."""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

VENV_PYTHON = Path(__file__).parent / ".venv" / "Scripts" / "python.exe"
SERVER_COMMAND = [str(VENV_PYTHON), "-m", "mcp_cbr_rates"]


async def main() -> None:
    server_params = StdioServerParameters(command=SERVER_COMMAND[0], args=SERVER_COMMAND[1:])

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            print(f"Connected to: {info.serverInfo.name} v{info.serverInfo.version}")
            print(f"Protocol version: {info.protocolVersion}")

            tools = await session.list_tools()
            print(f"\nAvailable tools ({len(tools.tools)}):")
            for tool in tools.tools:
                print(f"\n- {tool.name}: {tool.description}")
                print(f"  input schema: {tool.inputSchema}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
