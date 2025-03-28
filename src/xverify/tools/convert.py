from enum import Enum
from typing import Any, Callable, Literal, Optional, Type

from docstring_parser import parse
from pydantic import BaseModel, Field, create_model

from .base import BaseTool


def tool2model(tool: Callable, discriminator: str | None) -> Type[BaseTool]:
    """Convert a function signature to a Pydantic model with documentation"""
    assert isinstance(tool, Callable), "tool must be a callable"
    assert tool.__doc__ is not None, "tool must have a docstring"

    name = tool.__name__
    docs = parse(tool.__doc__)
    assert discriminator not in {
        p.arg_name for p in docs.params
    }, f"{discriminator} is reserved for the discriminator"

    fields: dict = {}
    if discriminator is not None:
        fields[discriminator] = (
            Literal[name],  # type: ignore
            Field(..., description="Function to call"),
        )
    fields.update(
        {
            p.arg_name: (
                p.type_name,  # parameter type
                Field(
                    default=p.default if p.is_optional else ...,
                    description=p.description,
                ),
            )
            for p in docs.params
        }
    )
    return create_model(
        name,
        __doc__=tool.__doc__,
        __base__=BaseTool,
        __cls_kwargs__=dict(_tool_func=tool),
        **fields,
    )


def jsonschema2model(
    schema: dict[str, Any], return_fields: bool
) -> type[BaseModel] | dict:
    type_mapping: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    model_fields = {}

    def process_field(field_name: str, field_props: dict[str, Any]) -> tuple:
        """Recursively processes a field and returns its type and Field instance."""
        json_type = field_props.get("type", "string")
        enum_values = field_props.get("enum")
        field_type: Any

        # Handle Enums
        if enum_values:
            enum_name: str = f"{field_name.capitalize()}Enum"
            field_type = Enum(enum_name, {v: v for v in enum_values})
        # Handle Nested Objects
        elif json_type == "object" and "properties" in field_props:
            # Recursively create submodel
            field_type = jsonschema2model(  # type: ignore
                field_props, return_fields=False
            )
        # Handle Arrays with Nested Objects
        elif json_type == "array" and "items" in field_props:
            item_props = field_props["items"]
            if item_props.get("type") == "object":
                item_type = jsonschema2model(  # type: ignore
                    item_props, return_fields=False
                )
            else:
                item_type = type_mapping.get(item_props.get("type"), Any)
            field_type = list[item_type]
        else:
            field_type = type_mapping.get(json_type, Any)

        # Handle default values and optionality
        default_value = field_props.get("default", ...)
        nullable = field_props.get("nullable", False)
        description = field_props.get("description", "")

        if nullable:
            field_type = Optional[field_type]

        if field_name not in required_fields:
            default_value = field_props.get("default", ...)

        return field_type, Field(default_value, description=description)

    for field_name, field_props in properties.items():
        model_fields[field_name] = process_field(field_name, field_props)

    if return_fields:
        return model_fields
    else:
        return create_model(schema["title"].replace(" ", "_"), **model_fields)
