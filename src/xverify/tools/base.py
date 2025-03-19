from abc import abstractmethod

from mcp.types import CallToolResult
from pydantic import BaseModel

__all__ = ["ToolUse", "BaseTool"]


class ToolUse:
    """
    Use tools in a pydantic model.

    Examples:
    ```python
    class ReasoiningToolResult(BaseModel):
        reasoning: str
        tool_use: ToolUse[calculator, search] # pick one of the tools
        tool_use2: ToolUse[calculator] # only one tool
        tool_use3: ToolUse[calculator] | None # optional
        tool_use4: list[ToolUse[calculator, search]] # list of tools
    ```
    """

    @abstractmethod
    def run_tool(self): ...  # see BaseTool


class BaseTool(BaseModel):
    """
    Base class for tools. After creating a tool model, you can call it with `run`.
    """

    # model_config = {
    #     "extra": "forbid"  # no extra fields
    # }

    @classmethod
    def __init_subclass__(cls, **kwargs):
        cls._tool_func = kwargs.pop("_tool_func")  # type: ignore
        cls._tool_name = cls._tool_func.__name__
        super().__init_subclass__(**kwargs)

    def run_tool(self, unwrap_content: bool = True):
        args = self.model_dump()
        # tool_name is used as discriminator, not passed as arg
        if "tool_name" in args:
            del args["tool_name"]
        res = self.__class__._tool_func(**args)
        if unwrap_content and isinstance(res, CallToolResult):
            # assumes TextContent
            return res.content[0].text  # type: ignore
        return res
