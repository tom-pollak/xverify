"""Tests for Literal type handling, especially with discriminated unions."""

import pytest
from typing import Literal, Union
from pydantic import BaseModel, Field

from xverify.xml import generate_gbnf_grammar_and_documentation
from xverify.tools import calculator, search
from xverify import XMLToolUse, Env


def test_basic_literal_fields():
    """Test that basic Literal fields generate correct grammar rules."""

    class BasicLiteralModel(BaseModel):
        mode: Literal["read", "write", "append"]
        level: Literal[1, 2, 3]
        flag: Literal[True, False]

    grammar, _ = generate_gbnf_grammar_and_documentation([BasicLiteralModel])

    # Check that each literal field has its own rule with all options
    assert "mode-literal ::=" in grammar
    assert (
        'nl "read"' in grammar and 'nl "write"' in grammar and 'nl "append"' in grammar
    )

    assert "level-literal ::=" in grammar
    assert 'nl "1"' in grammar and 'nl "2"' in grammar and 'nl "3"' in grammar

    assert "flag-literal ::=" in grammar
    assert 'nl "true"' in grammar and 'nl "false"' in grammar


def test_literal_fields_with_similar_names():
    """Test that Literal fields with the same name in different models get unique rules."""

    class ModelA(BaseModel):
        type: Literal["a", "b"]

    class ModelB(BaseModel):
        type: Literal["x", "y", "z"]

    grammar, _ = generate_gbnf_grammar_and_documentation([ModelA, ModelB])

    # Check that each model's literal field has its own rule with appropriate values
    assert "ModelA-type-literal ::=" in grammar
    assert 'nl "a"' in grammar and 'nl "b"' in grammar

    assert "ModelB-type-literal ::=" in grammar
    assert 'nl "x"' in grammar and 'nl "y"' in grammar and 'nl "z"' in grammar


def test_discriminated_union_with_literals():
    """Test discriminated unions with Literal fields that determine the discriminator."""

    class ToolAction(BaseModel):
        action: Literal["tool_use"] = Field(..., description="Action type")
        tool_name: str = Field(..., description="Name of the tool")

    class TextAction(BaseModel):
        action: Literal["text"] = Field(..., description="Action type")
        content: str = Field(..., description="Text content")

    class ActionModel(BaseModel):
        task: str = Field(..., description="Task description")
        action_to_take: Union[ToolAction, TextAction] = Field(
            ..., description="The action to perform", discriminator="action"
        )

    grammar, _ = generate_gbnf_grammar_and_documentation([ActionModel])

    # Each model should have its own literal definition for the action field
    assert "ToolAction-action-literal ::=" in grammar
    assert 'nl "tool_use"' in grammar

    assert "TextAction-action-literal ::=" in grammar
    assert 'nl "text"' in grammar

    # The discriminated union should use these specific literals
    assert "ToolAction ::=" in grammar
    assert "TextAction ::=" in grammar
    assert "action_to_take-union ::= ToolAction | TextAction" in grammar


def test_example_from_literal_tests():
    """Test the specific example from literal_tests.py."""

    # Define models exactly as in literal_tests.py
    class Tools(BaseModel):
        action: Literal["tool_use"] = Field(..., description="Action discriminator")
        tool_use: XMLToolUse[calculator, search] = Field(
            ..., description="The tool call to use"
        )

    class FinalAnswer(BaseModel):
        action: Literal["final_answer"] = Field(..., description="Action discriminator")
        answer: str = Field(..., description="The final answer to the question")

    class Reason_and_Act(BaseModel):
        scratchpad: str = Field(
            ...,
            description="Information from the Observation useful to answer the question",
        )
        reasoning: str = Field(
            ...,
            description="It describes your thoughts about the question you have been asked",
        )
        response: Union[Tools, FinalAnswer] = Field(
            ...,
            description="Final output: choose between the tool call or the final answer",
            discriminator="action",
        )

    # Generate the grammar
    env = Env(Reason_and_Act)
    grammar = env.gbnf

    # Verify literal fields have separate rules
    assert "Tools-action-literal ::=" in grammar
    assert 'nl "tool_use"' in grammar

    assert "FinalAnswer-action-literal ::=" in grammar
    assert 'nl "final_answer"' in grammar

    # Check that these are used in their respective model rules
    assert "<action>" in grammar and "Tools-action-literal" in grammar
    assert "<action>" in grammar and "FinalAnswer-action-literal" in grammar


def test_multiple_literal_fields_same_model():
    """Test multiple Literal fields in the same model."""

    class ComplexModel(BaseModel):
        action_type: Literal["create", "update", "delete"]
        visibility: Literal["public", "private", "restricted"]
        priority: Literal[1, 2, 3, 4, 5]

    grammar, _ = generate_gbnf_grammar_and_documentation([ComplexModel])

    # Each literal field should have its own rule
    assert "action_type-literal ::=" in grammar
    assert (
        'nl "create"' in grammar
        and 'nl "update"' in grammar
        and 'nl "delete"' in grammar
    )

    assert "visibility-literal ::=" in grammar
    assert (
        'nl "public"' in grammar
        and 'nl "private"' in grammar
        and 'nl "restricted"' in grammar
    )

    assert "priority-literal ::=" in grammar
    assert 'nl "1"' in grammar and 'nl "5"' in grammar  # Just checking a couple


def test_end_to_end_literal_parsing():
    """Test that the generated grammar correctly parses literals in XML."""

    class MenuAction(BaseModel):
        action_type: Literal["menu"] = Field(..., description="Action type")
        menu_item: str = Field(..., description="Selected menu item")

    class ButtonAction(BaseModel):
        action_type: Literal["button"] = Field(..., description="Action type")
        button_id: int = Field(..., description="Button identifier")

    class UIAction(BaseModel):
        description: str = Field(..., description="Description of the action")
        user_action: Union[MenuAction, ButtonAction] = Field(
            ..., description="The UI action performed", discriminator="action_type"
        )

    # Create test XML
    menu_xml = """
    <UIAction>
    <description>User selected from menu</description>
    <user_action>
    <MenuAction>
    <action_type>menu</action_type>
    <menu_item>File > Save</menu_item>
    </MenuAction>
    </user_action>
    </UIAction>
    """

    button_xml = """
    <UIAction>
    <description>User clicked a button</description>
    <user_action>
    <ButtonAction>
    <action_type>button</action_type>
    <button_id>42</button_id>
    </ButtonAction>
    </user_action>
    </UIAction>
    """

    # Test parsing with the environment
    env = Env(UIAction)

    # Parse menu action
    parsed_menu = env.parse(menu_xml)
    assert parsed_menu is not None
    assert parsed_menu.user_action.action_type == "menu"
    assert parsed_menu.user_action.menu_item == "File > Save"

    # Parse button action
    parsed_button = env.parse(button_xml)
    assert parsed_button is not None
    assert parsed_button.user_action.action_type == "button"
    assert parsed_button.user_action.button_id == 42
