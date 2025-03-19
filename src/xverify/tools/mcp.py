"""
MCP integration for xVerify.

This module provides tools for integrating Model Context Protocol (MCP) servers with xVerify.
"""

import os
import asyncio
from typing import List, Optional, Type

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from pydantic import BaseModel, create_model

from .convert import jsonschema2model
from .tool_use import BaseTool

__all__ = ["MCPClient"]


class MCPClient:
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

    def __init__(
        self,
        command: str,
        args: Optional[List[str]] = None,
        env: dict[str, str] = dict(os.environ),
    ):
        """
        Initialize and connect to an MCP server.

        Args:
            command: The command to run the MCP server
            args: Arguments for the command
        """

        async def _connect(command: str, args: List[str], env: dict | None):
            """Connect to an MCP server."""
            # Set up connection
            params = StdioServerParameters(command=command, args=args, env=env)
            async with stdio_client(params) as streams:
                read, write = streams
                self.session = ClientSession(read, write)
                print("before initialize")
                await self.session.initialize()
                print("after initialize")

                # Discover tools and resources
                result = await self.session.list_tools()
                self.tools = {tool.name: tool for tool in result.tools}
                result = await self.session.list_resources()
                self.resources = {
                    resource.uri: resource for resource in result.resources
                }

        self.session = None
        self.tools = {}
        self.resources = {}
        self.tool_models = {}

        # Connect immediately
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_connect(command, args or [], env))

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

    def get_tool(self, tool_name: str) -> Type[BaseModel]:
        """Get a tool model by name."""
        # Return cached model if available
        if tool_name in self.tool_models:
            return self.tool_models[tool_name]

        # Check if the tool exists
        if tool_name not in self.tools:
            available = ", ".join(self.tools.keys())
            raise KeyError(f"Tool '{tool_name}' not found. Available: {available}")

        # Create and cache the model
        model = self._create_tool_model(tool_name)
        self.tool_models[tool_name] = model
        return model

    def list_tools(self) -> List[str]:
        """List all available tool names."""
        return list(self.tools.keys())

    def list_resources(self) -> List[str]:
        """List all available resource URIs."""
        return list(self.resources.keys())

    def read_resource(self, uri: str) -> str:
        """Read a resource synchronously."""

        async def _read_resource_async(uri: str) -> str:
            """Read a resource asynchronously."""
            if self.session is None:
                raise RuntimeError("Not connected to an MCP server")

            result = await self.session.read_resource(uri=uri)  # type: ignore

            # Extract text content
            for content in result.contents:
                if hasattr(content, "text") and content.text:
                    return content.text

            return ""

        if not self.session:
            raise RuntimeError("Not connected to an MCP server")

        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_read_resource_async(uri))

    def close(self):
        """Close the connection."""
        if self.session:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.session.close())
            self.session = None

    def __del__(self):
        """Clean up when the client is garbage collected."""
        self.close()

    def _create_tool_model(self, tool_name: str) -> Type[BaseModel]:
        """Create a Pydantic model for an MCP tool."""

        def _create_tool_executor(tool_name: str):
            """Create a function that executes an MCP tool."""
            client = self  # Capture client reference

            # Create a function that will execute the tool
            async def _execute_tool_async(**kwargs):
                if client.session is None:
                    raise RuntimeError("Not connected to an MCP server")

                result = await client.session.call_tool(tool_name, kwargs)  # type: ignore

                # Extract text content
                text_parts = []
                for content in result.content:
                    if hasattr(content, "text") and content.text:
                        text_parts.append(content.text)

                return "\n".join(text_parts)

            # synchronous wrapper
            def execute_tool(**kwargs):
                loop = asyncio.get_event_loop()
                return loop.run_until_complete(_execute_tool_async(**kwargs))

            # Set name to match tool name for proper identification
            execute_tool.__name__ = tool_name

            return execute_tool

        tool = self.tools[tool_name]
        fields: dict = jsonschema2model(tool.inputSchema, return_fields=True)  # type: ignore
        model = create_model(
            tool_name,
            __doc__=tool.description or f"MCP tool: {tool_name}",
            __base__=BaseTool,
            _tool=_create_tool_executor(tool_name),
            _tool_name=tool_name,
            **fields,
        )

        return model
