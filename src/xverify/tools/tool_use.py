from typing import Annotated, Union

from pydantic import BaseModel, Field

from .base import ToolUse
from .convert import BaseTool, tool2model

__all__ = ["run_tools", "JSONToolUse", "XMLToolUse"]


def run_tools(model: BaseModel, unwrap_content: bool = True) -> dict | None:
    """
    Run all tools in the model. Return a dictionary of nested results.
    """
    return _run_nested_tools(model, unwrap_content)  # type: ignore


class XMLToolUse(ToolUse):
    @classmethod
    def __class_getitem__(cls, tools) -> type:
        models = _mk_models(tools, discriminator=None)
        return models[0] if len(models) == 1 else Union[*models]  # type: ignore


class JSONToolUse(ToolUse):
    @classmethod
    def __class_getitem__(cls, tools) -> type:
        models = _mk_models(tools, discriminator="tool_name")
        return (
            models[0]
            if len(models) == 1
            else Annotated[
                Union[*models],  # type: ignore
                Field(..., discriminator="tool_name"),
            ]
        )


def _mk_models(tools, discriminator: str | None) -> list[type[BaseModel]]:
    # unpack tools
    if not isinstance(tools, tuple):
        tools = (tools,)
    tools = tuple(
        [
            item
            for sublist in tools
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
    )

    models = []
    for tool in tools:
        if isinstance(tool, type) and issubclass(tool, BaseModel):
            models.append(tool)
        elif callable(tool):
            models.append(tool2model(tool, discriminator=discriminator))
        else:
            raise TypeError(
                f"Expected callable or BaseModel subclass, got {type(tool)}"
            )
    return models


def _run_nested_tools(item, unwrap_content):
    match item:
        # If the item is a tool, run it and return its output keyed by the tool name.
        case BaseTool() as tool:
            return {tool._tool_name: tool.run_tool(unwrap_content=unwrap_content)}  # type: ignore

        # For any other BaseModel, process its fields recursively and filter out empty results.
        case BaseModel():
            subitems = {
                field: _run_nested_tools(value, unwrap_content) for field, value in item
            }
            subitems = {k: v for k, v in subitems.items() if v is not None}
            return subitems or None

        # Iterable containers
        case list() | tuple() | set() as container:
            processed = [_run_nested_tools(i, unwrap_content) for i in container]
            processed = type(container)(x for x in processed if x is not None)
            return processed or None

        # Mapping containers
        case dict():
            new_dict = {
                k: _run_nested_tools(v, unwrap_content) for k, v in item.items()
            }
            new_dict = {k: v for k, v in new_dict.items() if v is not None}
            return new_dict or None

        # For any other type, ignore it.
        case _:
            return None
