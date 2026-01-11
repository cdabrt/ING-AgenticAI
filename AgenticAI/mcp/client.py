import json
import logging
import sys
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


logger = logging.getLogger(__name__)


class MCPToolClient:
    """Async context manager for interacting with MCP servers over stdio."""

    def __init__(self, server_script: str, env: Optional[Dict[str, str]] = None):
        self.server_script = server_script
        self.env = env
        self.exit_stack = AsyncExitStack()
        self.session: Optional[ClientSession] = None
        self._transport = None

    async def __aenter__(self) -> "MCPToolClient":
        if not self.server_script.endswith(".py"):
            raise ValueError("Only python-based MCP servers are currently supported")

        logger.info("Launching MCP server %s", self.server_script)
        params = StdioServerParameters(command=sys.executable, args=[self.server_script], env=self.env)
        self._transport = await self.exit_stack.enter_async_context(stdio_client(params))
        stdio, write = self._transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(stdio, write))
        await self.session.initialize()
        logger.info("MCP session initialized for %s", self.server_script)
        return self

    async def __aexit__(self, exc_type, exc, tb):
        logger.info("Shutting down MCP server %s", self.server_script)
        await self.exit_stack.aclose()
        self.session = None
        self._transport = None

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        if not self.session:
            raise RuntimeError("MCP session has not been initialized")

        logger.info("Calling MCP tool=%s args=%s", tool_name, list(arguments.keys()))
        result = await self.session.call_tool(tool_name, arguments)
        return self._extract_text(result.content)

    async def list_tools(self) -> List[str]:
        if not self.session:
            raise RuntimeError("MCP session has not been initialized")

        response = await self.session.list_tools()
        logger.info("Discovered MCP tools: %s", ", ".join(tool.name for tool in response.tools))
        return [tool.name for tool in response.tools]

    @staticmethod
    def _extract_text(content_list: List[Any]) -> str:
        texts: List[str] = []
        for content in content_list:
            content_type = getattr(content, "type", None)
            if content_type == "text":
                texts.append(getattr(content, "text", ""))
            elif content_type == "json":
                texts.append(json.dumps(getattr(content, "json", {})))
            elif isinstance(content, str):
                texts.append(content)
        return "\n".join(filter(None, texts))
