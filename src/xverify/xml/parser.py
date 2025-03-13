from typing import Generator, Type, get_args

import xmltodict
from pydantic import BaseModel

__all__ = ["parse_xml_to_model"]


def parse_xml_to_model(model: Type[BaseModel], xml_text: str) -> BaseModel:
    # Force various container tags to be parsed as lists
    parsed = xmltodict.parse(
        xml_text, force_list=("list-item", "set-item", "key-item", "value-item")
    )

    # Include all container tags in model names for handling
    model_names = {
        "list",
        "set",
        "dict",
        "list-item",
        "set-item",
        "key-item",
        "value-item",
        *(_get_model_names(model)),
    }

    # First process dictionaries to handle alternating key-value pairs
    # processed = _process_dicts(parsed)

    # Then squeeze model keys
    squeezed = _squeeze_model_keys(parsed, model_names)
    return model.model_validate(squeezed)


def _iter_nested_models(
    item, seen_models=None
) -> Generator[type[BaseModel], None, None]:
    """
    Recursively iterate through nested models, avoiding infinite recursion.

    Args:
        item: The type or model to process
        seen_models: Set of model types already processed to avoid recursion
    """
    if seen_models is None:
        seen_models = set()

    # For models, process once and skip if already seen
    if isinstance(item, type) and issubclass(item, BaseModel):
        # Skip if we've already seen this model to avoid recursion
        if item in seen_models:
            return

        # Mark as seen before processing fields to prevent recursion
        seen_models.add(item)

        # Yield the model itself
        yield item

        # Process each field
        for sub_field in item.model_fields.values():
            yield from _iter_nested_models(sub_field.annotation, seen_models)
    else:
        # Process type arguments (for containers, unions, etc.)
        for arg in get_args(item):
            yield from _iter_nested_models(arg, seen_models)


def _get_model_names(model: Type[BaseModel]) -> set[str]:
    """
    Collects and returns a list of all nested BaseModel subclasses within the given model.
    """
    return {model.__name__ for model in _iter_nested_models(model)}


def _squeeze_model_keys(data, model_names: set[str]):
    """
    Recursively traverse the data structure. If a dict has exactly one key
    and that key is a BaseModel name, return its value (after processing).
    """
    if isinstance(data, dict):
        # Squeeze if the dict has a single key that is a BaseModel name.
        if "dict" in data:
            assert len(data) == 1
            # dicts are represented as {"key": ["key1", "key2"], "value": ["value1", "value2"]}
            return dict(zip(*data["dict"].values()))

        key, value = next(iter(data.items()))
        if key in model_names:
            assert len(data) == 1
            return _squeeze_model_keys(value, model_names)

        return {k: _squeeze_model_keys(v, model_names) for k, v in data.items()}

    elif isinstance(data, list):
        if data == [None]:
            return []
        return [_squeeze_model_keys(item, model_names) for item in data]
    return data
