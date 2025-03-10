import inspect
from abc import abstractmethod
from typing import Annotated, Callable, Literal, Tuple, Type, Union

from fastcore.docments import docments
from pydantic import BaseModel, Field, create_model

__all__ = ["ToolUse", "run_tools"]


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

    @classmethod
    def __class_getitem__(cls, tools: tuple[Callable] | Callable):
        # multiple funcs
        if isinstance(tools, Tuple):
            return Annotated[
                Union[*map(_tool2model, tools)],  # type: ignore
                Field(discriminator="tool_name"),
            ]
        # single func
        return _tool2model(tools)

    @abstractmethod
    def run_tool(self): ...  # see BaseTool


def run_tools(model: BaseModel) -> dict | None:
    """
    Run all tools in the model. Return a dictionary of nested results.
    """
    return _run_nested_tools(model)  # type: ignore


class BaseTool(BaseModel):
    """
    Base class for tools. After creating a tool model, you can call it with `run`.
    """

    @classmethod
    def __init_subclass__(cls, **kwargs):
        cls._tool_func = kwargs.pop("_tool_func")  # type: ignore
        super().__init_subclass__(**kwargs)

    def run_tool(self):
        args = self.model_dump()
        del args["tool_name"]  # tool_name is used as discriminator, not passed as arg
        return self.__class__._tool_func(**args)


def _tool2model(tool: Callable) -> Type[BaseTool]:
    """Convert a function signature to a Pydantic model with documentation"""
    assert isinstance(tool, Callable), "tool must be a callable"
    name = tool.__name__
    docs = docments(tool, full=True)
    assert "tool_name" not in docs, "tool_name is reserved for the discriminator"

    # Create docstring -- add return info if exists
    docstring = tool.__doc__ or ""
    return_info = docs.pop("return", {})
    return_type = return_info.get("anno")
    if return_type is not inspect._empty:
        docstring += f"\nReturns: {return_type.__name__}"
        if return_doc_str := return_info.get("docment"):
            docstring += f" - {return_doc_str}"

    # Build input parameters
    fields: dict = {
        name: (
            info["anno"],  # parameter type
            Field(
                default=(... if info["default"] is inspect._empty else info["default"]),
                description=info.get("docment", None),  # doc for parameter
            ),
        )
        for name, info in docs.items()
    }
    fields["tool_name"] = (
        Literal[name],  # type: ignore
        Field(..., description="Function to call"),
    )
    return create_model(
        name,
        __doc__=docstring,
        __base__=BaseTool,
        __cls_kwargs__=dict(_tool_func=tool),
        **fields,
    )


def _run_nested_tools(item):
    match item:
        # If the item is a tool, run it and return its output keyed by the tool name.
        case BaseTool() as tool:
            return {tool.tool_name: tool.run_tool()}  # type: ignore

        # For any other BaseModel, process its fields recursively and filter out empty results.
        case BaseModel():
            subitems = {field: _run_nested_tools(value) for field, value in item}
            subitems = {k: v for k, v in subitems.items() if v is not None}
            return subitems or None

        # Process list, tuple, and set in the same branch.
        case list() | tuple() | set() as container:
            processed = [_run_nested_tools(i) for i in container]
            processed = type(container)(x for x in processed if x is not None)
            return processed or None

        # Process dictionaries recursively.
        case dict():
            new_dict = {k: _run_nested_tools(v) for k, v in item.items()}
            new_dict = {k: v for k, v in new_dict.items() if v is not None}
            return new_dict or None

        # For any other type, ignore it.
        case _:
            return None
