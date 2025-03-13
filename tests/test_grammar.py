"""Tests for the GBNF grammar generation from Pydantic models."""

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Set, Union

from pydantic import BaseModel, Field
from xverify.xml import generate_gbnf_grammar_and_documentation


def assert_valid_grammar(model_classes):
    """Helper to test grammar generation and assert it produces valid output."""
    grammar, doc = generate_gbnf_grammar_and_documentation(model_classes)
    assert grammar is not None
    assert doc is not None
    assert isinstance(grammar, str)
    assert isinstance(doc, str)
    return grammar, doc


# Test Group 1: Basic Type Fields
def test_basic_types():
    """Test grammar generation for basic primitive types."""

    class BasicTypes(BaseModel):
        int_field: int
        str_field: str
        bool_field: bool
        float_field: float

    assert_valid_grammar([BasicTypes])


def test_optional_fields():
    """Test grammar generation for optional fields."""

    class OptionalFields(BaseModel):
        required_field: str
        optional_field: Optional[str] = None
        optional_int: Optional[int] = None

    assert_valid_grammar([OptionalFields])


def test_default_values():
    """Test grammar generation for fields with default values."""

    class DefaultValues(BaseModel):
        str_field: str = "default"
        int_field: int = 42
        bool_field: bool = True

    assert_valid_grammar([DefaultValues])


# Test Group 2: Collection Types
def test_list_type():
    """Test grammar generation for List type."""

    class ListModel(BaseModel):
        items: List[str]
        items_native: list[str]

    assert_valid_grammar([ListModel])


def test_dict_type():
    """Test grammar generation for Dict type."""

    class DictModel(BaseModel):
        properties: Dict[str, Any]
        mappings: Dict[str, int]
        mappings_native: dict[str, int]

    assert_valid_grammar([DictModel])


def test_set_type():
    """Test grammar generation for Set type."""

    class SetModel(BaseModel):
        unique_items: Set[int]
        unique_items_native: set[int]

    assert_valid_grammar([SetModel])


# Test Group 3: Nested Models
def test_nested_model():
    """Test grammar generation for nested models."""

    class Child(BaseModel):
        name: str
        age: int

    class Parent(BaseModel):
        name: str
        child: Child

    assert_valid_grammar([Parent])


def test_recursive_model():
    """Test grammar generation for recursive models."""

    class RecursiveModel(BaseModel):
        name: str
        children: Optional[List["RecursiveModel"]] = None

    RecursiveModel.model_rebuild()  # Using model_rebuild instead of update_forward_refs
    assert_valid_grammar([RecursiveModel])


# Test Group 4: Enums and Literals


class Status(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


def test_enum_field():
    """Test grammar generation for Enum fields."""

    class EnumModel(BaseModel):
        status: Status

    assert_valid_grammar([EnumModel])


def test_literal_field():
    """Test grammar generation for Literal fields."""

    class LiteralModel(BaseModel):
        mode: Literal["read", "write", "append"]

    assert_valid_grammar([LiteralModel])


# Test Group 5: Union Types
def test_simple_union():
    """Test grammar generation for simple Union types."""

    class UnionModel(BaseModel):
        value: Union[int, str]

    assert_valid_grammar([UnionModel])


def test_optional_union():
    """Test grammar generation for Optional Union types."""

    class OptionalUnionModel(BaseModel):
        value: Optional[Union[int, str]] = None

    assert_valid_grammar([OptionalUnionModel])


def test_native_union():
    """Test grammar generation for native Union types."""

    class NativeUnionModel(BaseModel):
        value: str | int

    assert_valid_grammar([NativeUnionModel])


class A(BaseModel):
    type: Literal["a"]
    a_value: int


class B(BaseModel):
    type: Literal["b"]
    b_value: str


class C(BaseModel):
    type: Literal["c"]
    c_value: bool
    optional: Optional[str] = None


def test_nested_union():
    """Test grammar generation for nested unions."""

    class NestedUnion(BaseModel):
        value: Union[A, B, C]

    assert_valid_grammar([NestedUnion])


# Test Group 6: Discriminated Unions
def test_discriminated_union():
    """Test grammar generation for discriminated unions."""

    class DiscriminatedModel(BaseModel):
        content: Annotated[Union[A, B, C], Field(discriminator="type")]

    assert_valid_grammar([DiscriminatedModel])


def test_list_of_discriminated_unions():
    """Test grammar generation for lists of discriminated unions."""

    class ListOfDiscriminated(BaseModel):
        items: List[Annotated[Union[A, B], Field(discriminator="type")]]

    assert_valid_grammar([ListOfDiscriminated])


def test_nested_discriminated_union():
    """Test grammar generation for nested discriminated unions."""

    class NestedItem(BaseModel):
        item: Annotated[Union[A, B], Field(discriminator="type")]

    class NestedDiscriminated(BaseModel):
        wrapper: NestedItem

    assert_valid_grammar([NestedDiscriminated])


class ComplexChild(BaseModel):
    name: str
    tags: List[str]
    properties: Dict[str, Any]


class ComplexModel(BaseModel):
    id: int
    name: str
    status: Status
    children: List[ComplexChild]
    config: Dict[str, Any] = {}
    content: Optional[Annotated[Union[A, B, C], Field(discriminator="type")]] = None
    values: List[Union[int, str]] = []
    mode: Literal["simple", "advanced"] = "simple"


# Test Group 7: Integration Test
def test_complex_integration():
    """Integration test with multiple features combined."""

    assert_valid_grammar([ComplexModel])


def test_multiple_models():
    """Test grammar generation for multiple models."""
    assert_valid_grammar([A, B, C])


def test_multiple_complex_models():
    """Test grammar generation for multiple complex models."""
    assert_valid_grammar([ComplexModel, ComplexChild])
