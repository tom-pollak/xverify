"""
Grammar and documentation generator for Pydantic models.

This module generates both:
1. XML-based GBNF grammar for structured output
2. Human-readable documentation of models and their fields

Key features:
- Recursive type inspection for deeply nested structures
- Proper handling of containers (lists, sets, dicts)
- Support for union types and optional fields
- Clear documentation of complex nested structures
"""

from __future__ import annotations

from inspect import getdoc, isclass
import json
from enum import Enum
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    get_args,
    get_origin,
    Annotated,
)

from pydantic import BaseModel

__all__ = ["generate_gbnf_grammar_and_documentation"]


def generate_gbnf_grammar_and_documentation(
    pydantic_model_list: List[Type[BaseModel]],
    model_prefix: str = "Output Model",
    fields_prefix: str = "Output Fields",
    include_descriptions: bool = True,
) -> Tuple[str, str]:
    """
    Generate both GBNF grammar and documentation for a list of Pydantic models.

    Args:
        pydantic_model_list: List of Pydantic model classes
        model_prefix: Prefix for model sections in documentation
        fields_prefix: Prefix for field sections in documentation
        include_descriptions: Whether to include field descriptions

    Returns:
        Tuple containing the grammar string and documentation string
    """
    # Generate documentation
    doc_generator = DocumentationGenerator(
        pydantic_model_list,
        model_prefix=model_prefix,
        fields_prefix=fields_prefix,
        include_descriptions=include_descriptions,
    )
    documentation = doc_generator.generate()

    # Generate grammar
    grammar_generator = XMLGrammarGenerator(pydantic_model_list)
    grammar = grammar_generator.generate()

    return grammar, documentation


class TypeKind(str, Enum):
    """Classification of type structures for grammar and documentation generation."""

    PRIMITIVE = "primitive"
    LIST = "list"
    SET = "set"
    DICT = "dict"
    UNION = "union"
    ENUM = "enum"
    MODEL = "model"
    LITERAL = "literal"
    CUSTOM = "custom"
    ANY = "any"
    NONE = "none"


def safe_type_name(typ: Any) -> str:
    """Safely get a type name even for objects without __name__ attribute."""
    return getattr(typ, "__name__", str(typ))


class TypeInfo:
    """
    Core type analysis class that extracts and organizes type metadata.

    This class handles the recursive inspection of types, tracking their
    structure, nested elements, and other relevant metadata for grammar
    and documentation generation.
    """

    def __init__(
        self,
        python_type: Type[Any],
        field_name: str = "",
        parent: Optional[TypeInfo] = None,
        registry: Optional[Dict[str, TypeInfo]] = None,
    ):
        self.python_type = python_type
        self.field_name = field_name
        self.parent = parent
        self.registry = registry or {}

        # Extract core type information
        self.origin = get_origin(python_type)
        self.args = get_args(python_type)

        # Determine the kind of type
        self.kind = self._determine_kind()

        # For container types, analyze their element types
        self.element_types: List[TypeInfo] = []
        self._process_element_types()

    def _determine_kind(self) -> TypeKind:
        """Determine the kind of type structure."""
        # Handle Annotated types by unwrapping
        if self.origin is Annotated:
            if not self.args:
                raise ValueError(f"Annotated type {self.python_type} has no arguments")
            # Create a new TypeInfo for the underlying type
            return TypeInfo(
                self.args[0], self.field_name, self.parent, self.registry
            ).kind

        # Handle primitive types
        if isclass(self.python_type):
            if issubclass(self.python_type, bool):
                return TypeKind.PRIMITIVE
            elif issubclass(self.python_type, int):
                return TypeKind.PRIMITIVE
            elif issubclass(self.python_type, float):
                return TypeKind.PRIMITIVE
            elif issubclass(self.python_type, str):
                return TypeKind.PRIMITIVE
            elif issubclass(self.python_type, Enum):
                return TypeKind.ENUM
            elif issubclass(self.python_type, BaseModel):
                return TypeKind.MODEL
            elif self.python_type is type(None):
                return TypeKind.NONE

        # Handle container types
        if self.origin is list:
            return TypeKind.LIST
        elif self.origin is set:
            return TypeKind.SET
        elif self.origin is dict:
            return TypeKind.DICT

        # Handle union types (includes Optional)
        if self.origin is Union or str(type(self.python_type)).endswith("UnionType"):
            return TypeKind.UNION

        # Handle Literal types
        if self.origin is Literal:
            return TypeKind.LITERAL

        # Default for unknown/custom types
        return TypeKind.CUSTOM

    def _process_element_types(self):
        """Process element types for container and union types."""
        # Handle Annotated types specially by extracting the base type
        if self.origin is Annotated and self.args:
            base_type = self.args[0]
            # Create TypeInfo for the base type, which might itself be a container
            base_info = TypeInfo(base_type, self.field_name, self.parent, self.registry)
            self.element_types = base_info.element_types
            return

        if self.kind == TypeKind.LIST or self.kind == TypeKind.SET:
            if self.args:
                element_type = self.args[0]
                self.element_types = [
                    TypeInfo(
                        element_type, f"{self.field_name}Element", self, self.registry
                    )
                ]

        elif self.kind == TypeKind.DICT:
            if len(self.args) >= 2:
                key_type, value_type = self.args[0], self.args[1]
                self.element_types = [
                    TypeInfo(key_type, f"{self.field_name}Key", self, self.registry),
                    TypeInfo(
                        value_type, f"{self.field_name}Value", self, self.registry
                    ),
                ]

        elif self.kind == TypeKind.UNION:
            self.element_types = [
                TypeInfo(arg, f"{self.field_name}Option{i}", self, self.registry)
                for i, arg in enumerate(self.args)
                if arg is not type(None)
            ]

    def get_field_types(self) -> Dict[str, TypeInfo]:
        """Get field types for model types."""
        if self.kind != TypeKind.MODEL:
            return {}

        result = {}
        if not hasattr(self.python_type, "__annotations__"):
            return result

        for field_name, field_type in self.python_type.__annotations__.items():
            # Skip special attributes
            if field_name.startswith("__") and field_name.endswith("__"):
                continue

            # Create TypeInfo for this field
            field_info = TypeInfo(field_type, field_name, self, self.registry)
            result[field_name] = field_info

        return result

    def is_optional(self) -> bool:
        """Check if this type is optional (Union with None)."""
        if self.kind != TypeKind.UNION:
            return False

        return any(arg is type(None) for arg in self.args)

    def get_type_name(self) -> str:
        """Get a descriptive string for this type."""
        # Handle Annotated types by extracting the base type
        if self.origin is Annotated and self.args:
            base_type = self.args[0]
            # Use the base type's name directly
            return TypeInfo(base_type, self.field_name).get_type_name()

        if self.kind == TypeKind.PRIMITIVE:
            return safe_type_name(self.python_type)

        elif self.kind == TypeKind.LIST:
            element_type = (
                self.element_types[0] if self.element_types else TypeInfo(Any)
            )
            return f"list of {element_type.get_type_name()}"

        elif self.kind == TypeKind.SET:
            element_type = (
                self.element_types[0] if self.element_types else TypeInfo(Any)
            )
            return f"set of {element_type.get_type_name()}"

        elif self.kind == TypeKind.DICT:
            if len(self.element_types) >= 2:
                key_type, value_type = self.element_types[0], self.element_types[1]
                return f"dict mapping {key_type.get_type_name()} to {value_type.get_type_name()}"
            return "dict"

        elif self.kind == TypeKind.UNION:
            if self.is_optional():
                # For optional types, show the type without the None option
                non_none_types = [arg for arg in self.args if arg is not type(None)]
                if len(non_none_types) == 1:
                    return f"optional {TypeInfo(non_none_types[0], self.field_name).get_type_name()}"

            return " or ".join(
                element.get_type_name() for element in self.element_types
            )

        elif self.kind == TypeKind.ENUM:
            return f"enum {safe_type_name(self.python_type)}"

        elif self.kind == TypeKind.MODEL:
            return safe_type_name(self.python_type)

        elif self.kind == TypeKind.LITERAL:
            literal_values = ", ".join(repr(arg) for arg in self.args)
            return f"Literal[{literal_values}]"

        elif self.kind == TypeKind.NONE:
            return "None"

        elif self.kind == TypeKind.ANY:
            return "Any"

        return f"custom-{safe_type_name(self.python_type)}"


class DocumentationGenerator:
    """
    Generates human-readable documentation for Pydantic models.

    This class traverses models recursively, creating detailed
    documentation that properly shows nested structures.
    """

    def __init__(
        self,
        models: List[Type[BaseModel]],
        model_prefix: str = "Output Model",
        fields_prefix: str = "Output Fields",
        include_descriptions: bool = True,
    ):
        self.models = models
        self.model_prefix = model_prefix
        self.fields_prefix = fields_prefix
        self.include_descriptions = include_descriptions
        self.type_registry: Dict[str, TypeInfo] = {}
        self.processed_models: Set[Type[BaseModel]] = set()

    def generate(self) -> str:
        """Generate the complete documentation."""
        documentation = []

        # Process primary models with their prefixes
        for model in self.models:
            model_doc = self._document_model(model, add_prefix=True)
            documentation.append(model_doc)

        return "\n".join(documentation)

    def _document_model(self, model: Type[BaseModel], add_prefix: bool = False) -> str:
        """Generate documentation for a single model."""
        if model in self.processed_models:
            return ""  # Skip already processed models

        self.processed_models.add(model)

        lines = []

        # Add model header
        if add_prefix:
            lines.append(f"{self.model_prefix}: {model.__name__}")
        else:
            lines.append(f"Model: {model.__name__}")

        # Add model description if available
        model_doc = getdoc(model)
        base_doc = getdoc(BaseModel)

        if model_doc and model_doc != base_doc:
            description = model_doc.strip()
            if description:
                # Format multi-line descriptions with proper indentation
                desc_lines = description.split("\n")
                lines.append("  Description: " + desc_lines[0])
                for line in desc_lines[1:]:
                    lines.append("    " + line)

        # Document fields
        if hasattr(model, "__annotations__") and model.__annotations__:
            if add_prefix:
                lines.append(f"  {self.fields_prefix}:")
            else:
                lines.append("  Fields:")

            # Process each field
            model_type_info = TypeInfo(model, registry=self.type_registry)
            field_types = model_type_info.get_field_types()

            for field_name, field_type_info in field_types.items():
                field_doc = self._document_field(
                    field_name, field_type_info, model, depth=1
                )
                lines.append(field_doc)

        # Add examples if available
        if (
            hasattr(model, "Config")
            and hasattr(model.Config, "json_schema_extra")
            and "example" in model.Config.json_schema_extra
        ):
            lines.append(f"  Expected Example Output for {model.__name__}:")
            example_json = json.dumps(
                model.Config.json_schema_extra["example"], indent=2
            )
            example_lines = example_json.split("\n")
            for line in example_lines:
                lines.append("    " + line)

        # Process additional models found while documenting this one
        additional_models_docs = []
        # Make a copy of the registry to avoid "dictionary changed during iteration" error
        registry_copy = list(self.type_registry.values())
        for model_type in registry_copy:
            if (
                model_type.kind == TypeKind.MODEL
                and model_type.python_type not in self.processed_models
            ):
                additional_doc = self._document_model(model_type.python_type)
                if additional_doc:
                    additional_models_docs.append(additional_doc)

        if additional_models_docs:
            lines.append("")  # Add spacing between models
            lines.extend(additional_models_docs)

        return "\n".join(lines)

    def _document_field(
        self,
        field_name: str,
        type_info: TypeInfo,
        model: Type[BaseModel],
        depth: int = 1,
    ) -> str:
        """Document a single field with proper handling of nested types."""
        indent = "    " * depth
        lines = []

        # Get field metadata
        field_info = model.model_fields.get(field_name, None)
        field_description = (
            field_info.description if field_info and field_info.description else ""
        )

        # Add field type information
        type_name = type_info.get_type_name()
        field_header = f"{indent}{field_name} ({type_name})"

        if field_description and self.include_descriptions:
            field_header += ":"

        lines.append(field_header)

        # Add field description if available
        if field_description and self.include_descriptions:
            lines.append(f"{indent}    Description: {field_description}")

        # Document nested elements for containers
        if type_info.kind == TypeKind.LIST or type_info.kind == TypeKind.SET:
            if type_info.element_types:
                element_type = type_info.element_types[0]
                # For model elements, register them for top-level documentation
                if element_type.kind == TypeKind.MODEL:
                    # Just register for top-level documentation, no need for redundant info
                    self.type_registry[element_type.python_type.__name__] = element_type

                # For union elements (like ToolUse[a, b]), describe the options
                elif element_type.kind == TypeKind.UNION:
                    lines.append(f"{indent}    Items can be one of:")
                    for option in element_type.element_types:
                        if option.kind == TypeKind.MODEL:
                            lines.append(
                                f"{indent}        - {option.python_type.__name__}"
                            )
                            # Register this model for later documentation
                            self.type_registry[option.python_type.__name__] = option
                        else:
                            lines.append(f"{indent}        - {option.get_type_name()}")

        # Document union types directly in the field
        elif type_info.kind == TypeKind.UNION:
            for option in type_info.element_types:
                if option.kind == TypeKind.MODEL:
                    # Register this model for later documentation
                    self.type_registry[option.python_type.__name__] = option

        # Document model fields directly
        elif type_info.kind == TypeKind.MODEL:
            # Don't include details here, just register for top-level docs
            # This ensures that models are documented at the top level
            nested_model = type_info.python_type
            self.type_registry[nested_model.__name__] = type_info
            # No need to add redundant type information here

        return "\n".join(lines)


class GrammarRule:
    """
    Represents a single GBNF grammar rule.

    This class encapsulates a named production rule in the grammar,
    handling its formatting and dependencies.
    """

    def __init__(self, name: str, pattern: str):
        self.name = name
        self.pattern = pattern
        self.dependencies: Set[str] = set()

    def __str__(self) -> str:
        return f"{self.name} ::= {self.pattern}"

    def add_dependency(self, rule_name: str):
        """Add a dependency on another rule."""
        self.dependencies.add(rule_name)


class GrammarRuleSet:
    """
    Manages a collection of grammar rules with dependency tracking.

    This class handles rule organization, deduplication, and proper
    sequencing based on dependencies.
    """

    def __init__(self):
        self.rules: Dict[str, GrammarRule] = {}
        self.root_rule: Optional[GrammarRule] = None

    def add_rule(self, rule: GrammarRule) -> None:
        """Add a rule to the set if it doesn't already exist."""
        if rule.name not in self.rules:
            self.rules[rule.name] = rule

    def set_root_rule(self, rule: GrammarRule) -> None:
        """Set the root rule for the grammar."""
        self.root_rule = rule
        self.add_rule(rule)

    def get_ordered_rules(self) -> List[GrammarRule]:
        """Get rules ordered by dependencies."""
        if not self.root_rule:
            return list(self.rules.values())

        # Start with root and traverse dependencies
        visited: Set[str] = set()
        ordered_rules: List[GrammarRule] = []

        def visit(rule_name: str):
            if rule_name in visited:
                return

            visited.add(rule_name)
            rule = self.rules.get(rule_name)

            if not rule:
                return

            # Visit dependencies first
            for dep in rule.dependencies:
                visit(dep)

            ordered_rules.append(rule)

        # Start traversal from root rule
        visit(self.root_rule.name)

        # Add any rules not reachable from root
        for rule in self.rules.values():
            if rule.name not in visited:
                ordered_rules.append(rule)

        return ordered_rules

    def __str__(self) -> str:
        """Convert the entire rule set to a string."""
        ordered_rules = self.get_ordered_rules()
        return "\n".join(str(rule) for rule in ordered_rules)


class XMLGrammarGenerator:
    """
    Generates XML-based GBNF grammar for Pydantic models.

    This class creates a grammar that can parse structured XML output
    matching the structure of the provided Pydantic models.
    """

    def __init__(self, models: List[Type[BaseModel]]):
        self.models = models
        self.rule_set = GrammarRuleSet()
        self.processed_types: Set[Type[Any]] = set()

        # Add primitive rules
        self._add_primitive_rules()

    def generate(self) -> str:
        """Generate the complete grammar."""
        # Create root rule
        root_rule = GrammarRule("root", "grammar-models")
        self.rule_set.set_root_rule(root_rule)

        # Create grammar-models rule
        models_pattern = " | ".join(model.__name__ for model in self.models)
        models_rule = GrammarRule("grammar-models", models_pattern)

        # Add dependencies
        for model in self.models:
            models_rule.add_dependency(model.__name__)
            self._process_model(model)

        self.rule_set.add_rule(models_rule)

        # Return the complete grammar
        return str(self.rule_set)

    def _process_model(self, model: Type[BaseModel]) -> None:
        """Process a single model, generating rules for it and its fields."""
        if model in self.processed_types:
            return

        self.processed_types.add(model)

        # Create TypeInfo for this model
        model_info = TypeInfo(model)
        field_types = model_info.get_field_types()

        # Create the model rule pattern
        pattern_parts = [rf'nl "<{model.__name__}>" nl']

        for field_name, field_type in field_types.items():
            # Add this field to the model pattern
            field_rule_name = self._process_field_type(field_name, field_type)
            pattern_parts.append(
                f'nl "<{field_name}>" {field_rule_name} nl "</{field_name}>"'
            )

        pattern_parts.append(f'nl "</{model.__name__}>"')
        pattern = " ".join(pattern_parts)

        # Create and add the model rule
        model_rule = GrammarRule(model.__name__, pattern)
        self.rule_set.add_rule(model_rule)

    def _process_field_type(self, field_name: str, type_info: TypeInfo) -> str:
        """Process a field type, creating rules as needed."""
        # Handle primitives
        if type_info.kind == TypeKind.PRIMITIVE:
            if issubclass(type_info.python_type, bool):
                return "boolean"
            elif issubclass(type_info.python_type, int):
                return "integer"
            elif issubclass(type_info.python_type, float):
                return "float"
            elif issubclass(type_info.python_type, str):
                return "string"

        # Handle lists
        elif type_info.kind == TypeKind.LIST:
            if type_info.element_types:
                element_type = type_info.element_types[0]
                element_rule_name = self._process_field_type(
                    f"{field_name}Element", element_type
                )
                rule_name = f"{field_name}-list"

                # Create list rule with better tag names
                list_pattern = rf'nl "<list>" ("<list-item>" {element_rule_name} nl "</list-item>")* nl "</list>"'
                list_rule = GrammarRule(rule_name, list_pattern)
                list_rule.add_dependency(element_rule_name)
                self.rule_set.add_rule(list_rule)

                return rule_name
            return "string-list"  # Default for empty lists

        # Handle sets
        elif type_info.kind == TypeKind.SET:
            if type_info.element_types:
                element_type = type_info.element_types[0]
                element_rule_name = self._process_field_type(
                    f"{field_name}Element", element_type
                )
                rule_name = f"{field_name}-set"

                # Create set rule with better tag names
                set_pattern = rf'nl "<set>" ("<set-item>" {element_rule_name} nl "</set-item>")* nl "</set>"'
                set_rule = GrammarRule(rule_name, set_pattern)
                set_rule.add_dependency(element_rule_name)
                self.rule_set.add_rule(set_rule)

                return rule_name
            return "string-set"  # Default for empty sets

        # Handle dictionaries
        elif type_info.kind == TypeKind.DICT:
            if len(type_info.element_types) >= 2:
                key_type, value_type = (
                    type_info.element_types[0],
                    type_info.element_types[1],
                )
                key_rule_name = self._process_field_type(f"{field_name}Key", key_type)
                value_rule_name = self._process_field_type(
                    f"{field_name}Value", value_type
                )
                rule_name = f"{field_name}-dict"

                # Create dict rule with better tag names
                dict_pattern = rf'nl "<dict>" ("<dict-entry>" nl "<key>" {key_rule_name} nl "</key>" nl "<value>" {value_rule_name} nl "</value>" nl "</dict-entry>")* nl "</dict>"'
                dict_rule = GrammarRule(rule_name, dict_pattern)
                dict_rule.add_dependency(key_rule_name)
                dict_rule.add_dependency(value_rule_name)
                self.rule_set.add_rule(dict_rule)

                return rule_name
            return "string-dict"  # Default for untyped dicts

        # Handle unions
        elif type_info.kind == TypeKind.UNION:
            # Create a union rule
            union_parts = []
            rule_name = f"{field_name}-union"

            for i, element_type in enumerate(type_info.element_types):
                element_rule_name = self._process_field_type(
                    f"{field_name}Option{i}", element_type
                )
                union_parts.append(element_rule_name)

            # Create union rule
            union_pattern = " | ".join(union_parts)
            union_rule = GrammarRule(rule_name, union_pattern)

            # Add dependencies
            for part in union_parts:
                union_rule.add_dependency(part)

            self.rule_set.add_rule(union_rule)
            return rule_name

        # Handle enums
        elif type_info.kind == TypeKind.ENUM:
            rule_name = f"{field_name}-enum"
            enum_values = [f'nl "{e.value}"' for e in type_info.python_type]
            enum_pattern = " | ".join(enum_values)
            enum_rule = GrammarRule(rule_name, enum_pattern)
            self.rule_set.add_rule(enum_rule)
            return rule_name

        # Handle Literal types
        elif type_info.kind == TypeKind.LITERAL:
            rule_name = f"{field_name}-literal"
            literal_values = [f'nl "{value}"' for value in type_info.args]
            literal_pattern = " | ".join(literal_values)
            literal_rule = GrammarRule(rule_name, literal_pattern)
            self.rule_set.add_rule(literal_rule)
            return rule_name

        # Handle models (recursive)
        elif type_info.kind == TypeKind.MODEL:
            model_class = type_info.python_type
            self._process_model(model_class)
            return model_class.__name__

        # Default for unknown types
        return "string"

    def _add_primitive_rules(self) -> None:
        """Add rules for primitive types."""
        # String rule
        string_rule = GrammarRule("string", r"nl [^\n<] ([^<])*")
        self.rule_set.add_rule(string_rule)

        # Boolean rule
        boolean_rule = GrammarRule("boolean", r'nl "true" | nl "false"')
        self.rule_set.add_rule(boolean_rule)

        # Integer rule
        integer_rule = GrammarRule("integer", r"nl [0-9]+")
        self.rule_set.add_rule(integer_rule)

        # Float rule
        float_rule = GrammarRule("float", r'nl "-"? [0-9]+ ("." [0-9]+)?')
        self.rule_set.add_rule(float_rule)

        # Null rule
        null_rule = GrammarRule("null", r'nl "null"')
        self.rule_set.add_rule(null_rule)

        # Newline rule
        nl_rule = GrammarRule("nl", r'"\n"')
        self.rule_set.add_rule(nl_rule)
