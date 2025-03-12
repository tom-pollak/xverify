import xmltodict
from pydantic import BaseModel
from typing import Generator, Type, get_args

__all__ = ["parse_xml_to_model"]


def parse_xml_to_model(model: Type[BaseModel], xml_text: str) -> BaseModel:
    parsed = xmltodict.parse(xml_text, force_list=("items",))
    model_names = _get_model_names(model)
    squeezed = _squeeze_model_keys(parsed, model_names)
    return model.model_validate(squeezed)


def _iter_nested_models(item) -> Generator[type[BaseModel], None, None]:
    if isinstance(item, type) and issubclass(item, BaseModel):
        yield item  # Yield the model itself.
        for sub_field in item.model_fields.values():
            yield from _iter_nested_models(sub_field.annotation)
    else:
        for arg in get_args(item):
            yield from _iter_nested_models(arg)


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
        if len(data) == 1:
            key, value = next(iter(data.items()))
            if key in model_names:
                return _squeeze_model_keys(value, model_names)
        # Otherwise, process each key-value pair.
        return {k: _squeeze_model_keys(v, model_names) for k, v in data.items()}
    elif isinstance(data, list):
        return [_squeeze_model_keys(item, model_names) for item in data]
    return data
