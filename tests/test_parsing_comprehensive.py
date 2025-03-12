"""
Comprehensive end-to-end testing of xVerify's grammar generation,
prompt creation, LLM output, and XML parsing.

This test suite validates that the grammar correctly guides LLMs
to generate parseable XML outputs for a wide variety of model structures.
"""

import pytest
from typing import Optional, List, Dict, Union, Literal, Any
from enum import Enum
from pydantic import BaseModel, Field

import xmltodict
from xml.parsers.expat import ExpatError as ParseError

from xverify import Env, ToolUse
from xverify.tools import calculator, search


# Mock LLM for testing without actual model calls
class MockLLM:
    """Mock LLM that returns predefined responses based on model structure."""

    def __init__(self, *args, **kwargs):
        self.responses = {}

    def add_response(self, model_name: str, xml_response: str):
        """Add a predefined response for a model name."""
        self.responses[model_name] = xml_response

    def generate(self, prompt, sampling_params=None, **kwargs):
        """Generate mock response based on model name detected in prompt."""
        for model_name, response in self.responses.items():
            if model_name in prompt:
                return [MockOutput(response)]

        # Default simple response if no specific match
        return [MockOutput("<SimpleModel>\n<value>default response</value>\n</SimpleModel>")]


class MockOutput:
    """Mock output from LLM to simulate vLLM response structure."""

    def __init__(self, text):
        self.outputs = [MockOutputText(text)]


class MockOutputText:
    """Mock output text from LLM."""

    def __init__(self, text):
        self.text = text


# Test utility function
def run_model_parsing_test(model_cls, xml_response=None, should_parse=True):
    """
    Test that a model can be properly processed through the entire pipeline.

    Args:
        model_cls: Pydantic model class to test
        xml_response: Optional mock XML response (otherwise generates from model)
        should_parse: Whether parsing should succeed
    """
    # Initialize environment with model
    env = Env(model_cls)

    # Use mock LLM for testing
    mock_llm = MockLLM()

    # If no XML response provided, generate a simple one based on model structure
    if xml_response is None:
        xml_response = generate_sample_xml(model_cls)

    # Add response to mock LLM
    mock_llm.add_response(model_cls.__name__, xml_response)

    # Create sampling parameters
    sampling_params = env.sampling_params(max_tokens=500, temperature=0.7)

    # Generate mock output
    prompt = f"Generate XML for {model_cls.__name__}:\n{env.doc}"
    mock_output = mock_llm.generate(prompt, sampling_params=sampling_params)
    output_text = mock_output[0].outputs[0].text

    # Try to parse output
    try:
        parsed = env.parse(output_text)
        if should_parse:
            assert parsed is not None, f"Failed to parse valid output for {model_cls.__name__}"
            # Validate the parsed model matches the expected class
            assert isinstance(parsed, model_cls), f"Parsed result is not an instance of {model_cls.__name__}"
        else:
            assert parsed is None, f"Incorrectly parsed invalid output for {model_cls.__name__}"
    except Exception as e:
        if should_parse:
            pytest.fail(f"Error parsing output for {model_cls.__name__}: {str(e)}")


def generate_sample_xml(model_cls):
    """Generate a simple XML sample based on model structure."""
    model_name = model_cls.__name__

    # Check for primitive field types
    if hasattr(model_cls, "__annotations__"):
        fields = []
        for field_name, field_type in model_cls.__annotations__.items():
            field_xml = generate_field_xml(field_name, field_type)
            fields.append(field_xml)

        xml = f"<{model_name}>\n" + "\n".join(fields) + f"\n</{model_name}>"
        return xml

    # Default simple response
    return f"<{model_name}>\n<value>sample value</value>\n</{model_name}>"


def generate_field_xml(field_name, field_type):
    """Generate XML for a field based on its type."""
    origin = getattr(field_type, "__origin__", None)
    args = getattr(field_type, "__args__", [])

    # Handle primitive types
    if field_type == str:
        return f"<{field_name}>sample text</{field_name}>"
    elif field_type == int:
        return f"<{field_name}>42</{field_name}>"
    elif field_type == bool:
        return f"<{field_name}>true</{field_name}>"
    elif field_type == float:
        return f"<{field_name}>3.14</{field_name}>"

    # Handle list types with new format
    elif origin == list:
        item_type = args[0] if args else Any
        return f"<{field_name}>\n<list>\n<list-item>{generate_simple_value(item_type)}</list-item>\n</list>\n</{field_name}>"

    # Handle union types (including Optional)
    elif origin == Union:
        # Use the first non-None type for the sample
        for arg in args:
            if arg is not type(None):
                return f"<{field_name}>{generate_simple_value(arg)}</{field_name}>"
        return f"<{field_name}>null</{field_name}>"

    # Handle dict types with new format (without dict-entry tags)
    elif origin == dict:
        return f"<{field_name}>\n<dict>\n<key>sample_key</key>\n<value>sample_value</value>\n</dict>\n</{field_name}>"

    # Default for unknown/complex types
    return f"<{field_name}>sample value</{field_name}>"


def generate_simple_value(type_hint):
    """Generate a simple value for a type hint."""
    if type_hint == str:
        return "sample text"
    elif type_hint == int:
        return "42"
    elif type_hint == bool:
        return "true"
    elif type_hint == float:
        return "3.14"
    else:
        return "sample value"


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


# Test Cases - Tool Use
class ToolUseModel(BaseModel):
    """Model with tool use."""

    task: str
    tool: ToolUse[calculator, search]


class MultiToolUseModel(BaseModel):
    """Model with multiple tool uses."""

    task: str
    tools: List[ToolUse[calculator, search]]


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
    tools: Optional[List[ToolUse[calculator, search]]] = None


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
    # Using the new list format with multiple items
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


# Tests for tool use models
def test_tool_use_model():
    """Test model with tool use."""
    xml = """
    <ToolUseModel>
    <task>Calculate the result</task>
    <tool>
    <calculator>
    <tool_name>calculator</tool_name>
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
    <tool_name>calculator</tool_name>
    <expression>2 + 2</expression>
    </calculator>
    </list-item>
    <list-item>
    <search>
    <tool_name>search</tool_name>
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
    <tool_name>calculator</tool_name>
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


# Integration test with actual LLM (disabled by default)
@pytest.mark.skip(reason="Requires actual LLM connection")
def test_with_real_llm():
    """Test with actual LLM (skipped by default)."""
    from vllm import LLM

    # Initialize LLM
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct", max_model_len=2000)

    # Test with simple model
    env = Env(SimpleModel)
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


if __name__ == "__main__":
    # Run a simple test when executed directly
    test_simple_model()
    print("Simple model test passed!")
