"""
MCP integration for xVerify.

This module provides tools for integrating Model Context Protocol (MCP) servers with xVerify.
"""

import asyncio
import os
import warnings
from contextlib import AsyncExitStack
from functools import lru_cache
from typing import Any, Type

from mcp import ClientSession, StdioServerParameters, stdio_client
from pydantic import BaseModel, create_model

from .convert import jsonschema2model
from .tool_use import BaseTool

__all__ = ["MCPClient"]

"""
Client for Model Context Protocol (MCP) servers.

Example:
```python
# Create a client (connects automatically)
client = MCPClient("npx", ["-y", "@modelcontextprotocol/server-calculator"])

# Use tools in your models
class Math(BaseModel):
    expression: str
    result: XMLToolUse[client["calculate"]]

# Use multiple tools
class MultiTool(BaseModel):
    query: str
    tool: XMLToolUse[client["search", "summarize"]]
```
"""


class MCPClient:
    _event_loop: asyncio.AbstractEventLoop | None
    _exit_stack: AsyncExitStack
    _session: ClientSession

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] = dict(os.environ),
    ):
        """
        Initialize a synchronous MCP connection manager.

        Args:
            command: The command to execute (e.g., "python")
            args: Optional command line arguments
            env: Optional environment variables
        """

        async def _connect(server_params: StdioServerParameters):
            _exit_stack = AsyncExitStack()

            # Set up the stdio client
            read_stream, write_stream = await _exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            _session = await _exit_stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )

            # Initialize the session
            await _session.initialize()
            return _exit_stack, _session

        self._event_loop = None
        self._exit_stack, self._session = self._run_async(
            _connect(
                server_params=StdioServerParameters(
                    command=command,
                    args=args or [],
                    env=env,
                )
            )
        )

    def _run_async(self, coro):
        """Run an async coroutine in the event loop and return the result synchronously."""
        if self._event_loop is None or self._event_loop.is_closed():
            self._event_loop = asyncio.new_event_loop()
        return self._event_loop.run_until_complete(coro)

    def __del__(self):
        """Clean up resources when the object is garbage collected."""

        async def _close_exit_stack():
            await self._exit_stack.aclose()

        try:
            self._run_async(_close_exit_stack())
        except Exception as e:
            warnings.warn(f"Error closing exit stack: {e}")

        if self._event_loop is not None and not self._event_loop.is_closed():
            self._event_loop.close()

    # === Tool models ===

    def __getitem__(
        self,
        tool_names,  # : str | tuple[str, ...]
    ) -> Type[BaseModel] | list[Type[BaseModel]]:
        """
        Get tool model(s) by name.

        Args:
            tool_names: The name of the tool or tuple of tool names

        Returns:
            A single model or list of models

        """
        return (
            [self.get_tool(name) for name in tool_names]
            if isinstance(tool_names, tuple)
            else self.get_tool(tool_names)
        )

    @lru_cache(maxsize=None)
    def get_tool(self, tool_name: str) -> Type[BaseModel]:
        """Create a Pydantic model for an MCP tool."""

        tools = self.list_tools().tools
        try:
            tool = next(tool for tool in tools if tool.name == tool_name)
        except StopIteration as e:
            raise KeyError(f"Tool '{tool_name}' not found. Available: {tools}") from e

        fields: dict = jsonschema2model(tool.inputSchema, return_fields=True)  # type: ignore

        def _tool_func(client):
            def _f(**kwargs):
                return client.call_tool(name=tool_name, arguments=kwargs)

            return _f

        tool_func = _tool_func(client=self)
        tool_func.__name__ = tool_name  # type: ignore

        return create_model(
            tool_name,
            __doc__=tool.description,
            __base__=BaseTool,
            __cls_kwargs__=dict(_tool_func=tool_func),
            **fields,
        )

    # === Synchronous session method wrappers ===

    def list_prompts(self):
        """List available prompts from the server (blocking)."""

        async def _list_prompts():
            return await self._session.list_prompts()

        return self._run_async(_list_prompts())

    def list_tools(self):
        """List available tools from the server (blocking)."""

        async def _list_tools():
            return await self._session.list_tools()

        return self._run_async(_list_tools())

    def list_resources(self):
        """List available resources from the server (blocking)."""

        async def _list_resources():
            return await self._session.list_resources()

        return self._run_async(_list_resources())

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None):
        """Call a tool on the server (blocking)."""

        async def _call_tool():
            return await self._session.call_tool(name, arguments)

        return self._run_async(_call_tool())

    def read_resource(self, uri: str):
        """Read a resource from the server (blocking)."""

        async def _read_resource():
            return await self._session.read_resource(uri)  # type: ignore

        return self._run_async(_read_resource())

    def get_prompt(self, name: str, arguments: dict[str, Any] | None = None):
        """Get a prompt from the server (blocking)."""

        async def _get_prompt():
            return await self._session.get_prompt(name, arguments)

        return self._run_async(_get_prompt())
