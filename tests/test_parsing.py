"""
Comprehensive end-to-end testing of xVerify's grammar generation,
prompt creation, LLM output, and XML parsing.

This test suite validates that the grammar correctly guides LLMs
to generate parseable XML outputs for a wide variety of model structures.
"""

from enum import Enum
from typing import Annotated, Any, Dict, List, Literal, Optional, Union

import pytest
from pydantic import BaseModel, Field

from xverify import GuidedSchema, XMLToolUse
from xverify.tools import calculator, search


# Test utility function
def run_model_parsing_test(model_cls, xml_response, should_parse=True):
    """
    Test that a model can be properly processed through the entire pipeline.

    Args:
        model_cls: Pydantic model class to test
        xml_response: Optional mock XML response
        should_parse: Whether parsing should succeed
    """
    env = GuidedSchema(model_cls)
    try:
        parsed = env.parse(xml_response)
        if should_parse:
            assert (
                parsed is not None
            ), f"Failed to parse valid output for {model_cls.__name__}"
            # Validate the parsed model matches the expected class
            assert isinstance(
                parsed, model_cls
            ), f"Parsed result is not an instance of {model_cls.__name__}"
        else:
            assert (
                parsed is None
            ), f"Incorrectly parsed invalid output for {model_cls.__name__}"
    except Exception as e:
        if should_parse:
            pytest.fail(f"Error parsing output for {model_cls.__name__}: {str(e)}")


@pytest.mark.skip(reason="Requires actual LLM connection")
def test_with_real_llm():
    """Test with actual LLM (skipped by default)."""
    from vllm import LLM

    # Initialize LLM
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", max_model_len=2000)

    # Test with simple model
    env = GuidedSchema(SimpleModel)
    prompt = f"You are a structured output testing engineer. Fuzzily test the structured output given:\n{env.doc}"

    sampling_params = env.sampling_params(max_tokens=500, temperature=1.0)
    output = llm.generate(prompt, sampling_params=sampling_params)

    text = output[0].outputs[0].text
    try:
        parsed = env.parse(text)
        assert parsed is not None, "Failed to parse LLM output"
        assert isinstance(parsed, SimpleModel), "Parsed result has wrong type"
    except Exception as e:
        if "ParseError" in str(e.__class__):
            pytest.skip("LLM output was incomplete")
        else:
            pytest.fail(f"Error parsing LLM output: {str(e)}")


class A(BaseModel):
    type: Literal["a"]
    a_value: int


class B(BaseModel):
    type: Literal["b"]
    b_value: str


# Test Cases - Basic Types
class SimpleModel(BaseModel):
    """A simple model with basic field types."""

    text: str
    number: int
    flag: bool


class OptionalFieldsModel(BaseModel):
    """Model with optional fields."""

    required: str
    optional: Optional[str] = None
    with_default: str = "default value"


class NestedModel(BaseModel):
    """Model with a nested model field."""

    name: str
    child: SimpleModel


# Test Cases - Container Types
class ListModel(BaseModel):
    """Model with list fields."""

    items: List[str]
    numbers: List[int]


class DictModel(BaseModel):
    """Model with dictionary fields."""

    properties: Dict[str, str]
    metadata: Dict[str, Any]


# Test Cases - Union Types
class Status(str, Enum):
    """Simple enum for status values."""

    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"


class UnionModel(BaseModel):
    """Model with union field."""

    value: Union[str, int]
    status: Status
    mode: Literal["simple", "advanced"]


class NativeUnionModel(BaseModel):
    """Model with native union field."""

    value: str | int


class NestedUnionModel(BaseModel):
    """Model with nested union field."""

    value: Union[A, B]


class NestedDiscriminatedUnionModel(BaseModel):
    """Model with nested discriminated union field."""

    value: Annotated[Union[A, B], Field(discriminator="type")]


# Test Cases - Tool Use
class ToolUseModel(BaseModel):
    """Model with tool use."""

    task: str
    tool: XMLToolUse[calculator, search]


class MultiToolUseModel(BaseModel):
    """Model with multiple tool uses."""

    task: str
    tools: List[XMLToolUse[calculator, search]]


# Test Cases - Complex Types
class RecursiveItem(BaseModel):
    """Item with recursive structure."""

    name: str
    children: Optional[List["RecursiveItem"]] = None


RecursiveItem.model_rebuild()  # Finalize the recursive model


class ComplexModel(BaseModel):
    """Complex model with multiple nested structures."""

    id: int
    name: str
    status: Status
    items: List[RecursiveItem]
    metadata: Dict[str, Any] = {}
    tools: Optional[List[XMLToolUse[calculator, search]]] = None


# Tests for basic models
def test_simple_model():
    """Test simple model with basic fields."""
    xml = """
    <SimpleModel>
    <text>Hello World</text>
    <number>42</number>
    <flag>true</flag>
    </SimpleModel>
    """
    run_model_parsing_test(SimpleModel, xml)


def test_optional_fields_model():
    """Test model with optional fields."""
    xml = """
    <OptionalFieldsModel>
    <required>required value</required>
    <optional>optional value</optional>
    <with_default>custom default</with_default>
    </OptionalFieldsModel>
    """
    run_model_parsing_test(OptionalFieldsModel, xml)


def test_nested_model():
    """Test model with nested structure."""
    xml = """
    <NestedModel>
    <name>Parent Name</name>
    <child>
    <SimpleModel>
    <text>Child Text</text>
    <number>42</number>
    <flag>true</flag>
    </SimpleModel>
    </child>
    </NestedModel>
    """
    run_model_parsing_test(NestedModel, xml)


# Tests for container models
def test_list_model():
    """Test model with list fields."""
    xml = """
    <ListModel>
    <items>
    <list>
    <list-item>Item 1</list-item>
    <list-item>Item 2</list-item>
    <list-item>Item 3</list-item>
    </list>
    </items>
    <numbers>
    <list>
    <list-item>42</list-item>
    <list-item>43</list-item>
    </list>
    </numbers>
    </ListModel>
    """
    run_model_parsing_test(ListModel, xml)

    xml2 = """
    <ListModel>
    <items>
    <list>
    <list-item>Item 1</list-item>
    </list>
    </items>
    <numbers>
    <list>
    <list-item>43</list-item>
    </list>
    </numbers>
    </ListModel>
    """
    run_model_parsing_test(ListModel, xml2)


def test_dict_model():
    """Test model with dictionary fields."""
    xml = """
    <DictModel>
    <properties>
    <dict>
    <key>key1</key>
    <value>value1</value>
    <key>key2</key>
    <value>value2</value>
    </dict>
    </properties>
    <metadata>
    <dict>
    <key>count</key>
    <value>42</value>
    <key>active</key>
    <value>true</value>
    </dict>
    </metadata>
    </DictModel>
    """
    run_model_parsing_test(DictModel, xml)


# Tests for union models
def test_union_model():
    """Test model with union field."""
    xml = """
    <UnionModel>
    <value>string value</value>
    <status>active</status>
    <mode>simple</mode>
    </UnionModel>
    """
    run_model_parsing_test(UnionModel, xml)


def test_native_union_model():
    """Test model with native union field."""
    xml = """
    <NativeUnionModel>
    <value>42</value>
    </NativeUnionModel>
    """
    run_model_parsing_test(NativeUnionModel, xml)

    xml = """
    <NativeUnionModel>
    <value>string value</value>
    </NativeUnionModel>
    """
    run_model_parsing_test(NativeUnionModel, xml)


def test_nested_union_model():
    """Test model with nested union field."""
    xml = """
    <NestedUnionModel>
    <value>
    <A>
    <type>a</type>
    <a_value>42</a_value>
    </A>
    </value>
    </NestedUnionModel>
    """
    run_model_parsing_test(NestedUnionModel, xml)


def test_nested_discriminated_union_model():
    """Test model with nested discriminated union field."""
    xml = """
    <NestedDiscriminatedUnionModel>
    <value>
    <A>
    <type>a</type>
    <a_value>42</a_value>
    </A>
    </value>
    </NestedDiscriminatedUnionModel>
    """
    run_model_parsing_test(NestedDiscriminatedUnionModel, xml)


# Tests for tool use models
def test_tool_use_model():
    """Test model with tool use."""
    xml = """
    <ToolUseModel>
    <task>Calculate the result</task>
    <tool>
    <calculator>
    <expression>2 + 2</expression>
    </calculator>
    </tool>
    </ToolUseModel>
    """
    run_model_parsing_test(ToolUseModel, xml)


def test_multi_tool_use_model():
    """Test model with multiple tool uses."""
    xml = """
    <MultiToolUseModel>
    <task>Perform multiple operations</task>
    <tools>
    <list>
    <list-item>
    <calculator>
    <expression>2 + 2</expression>
    </calculator>
    </list-item>
    <list-item>
    <search>
    <query>python programming</query>
    <num_results>3</num_results>
    </search>
    </list-item>
    </list>
    </tools>
    </MultiToolUseModel>
    """
    run_model_parsing_test(MultiToolUseModel, xml)


# Tests for complex models
def test_recursive_model():
    """Test model with recursive structure."""
    xml = """
    <RecursiveItem>
    <name>Root Item</name>
    <children>
    <list>
    <list-item>
    <RecursiveItem>
    <name>Child 1</name>
    <children>
    <list>
    <list-item>
    <RecursiveItem>
    <name>Grandchild</name>
    </RecursiveItem>
    </list-item>
    </list>
    </children>
    </RecursiveItem>
    </list-item>
    <list-item>
    <RecursiveItem>
    <name>Child 2</name>
    </RecursiveItem>
    </list-item>
    </list>
    </children>
    </RecursiveItem>
    """
    run_model_parsing_test(RecursiveItem, xml)


def test_complex_model():
    """Test complex model with multiple nested structures."""
    xml = """
    <ComplexModel>
    <id>1234</id>
    <name>Complex Example</name>
    <status>active</status>
    <items>
    <list>
    <list-item>
    <RecursiveItem>
    <name>Item 1</name>
    <children>
    <list>
    <list-item>
    <RecursiveItem>
    <name>Subitem 1</name>
    </RecursiveItem>
    </list-item>
    </list>
    </children>
    </RecursiveItem>
    </list-item>
    </list>
    </items>
    <metadata>
    <dict>
    <key>created_at</key>
    <value>2023-01-01</value>
    </dict>
    </metadata>
    <tools>
    <list>
    <list-item>
    <calculator>
    <expression>2 * 3.14</expression>
    </calculator>
    </list-item>
    </list>
    </tools>
    </ComplexModel>
    """
    run_model_parsing_test(ComplexModel, xml)


# Tests for invalid XML
def test_invalid_xml():
    """Test handling of invalid XML."""
    xml = """
    <SimpleModel>
    <text>Incomplete model
    """
    run_model_parsing_test(SimpleModel, xml, should_parse=False)


def test_wrong_model_xml():
    """Test handling of XML for a different model."""
    xml = """
    <WrongModel>
    <field>value</field>
    </WrongModel>
    """
    run_model_parsing_test(SimpleModel, xml, should_parse=False)
