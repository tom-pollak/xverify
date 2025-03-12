# xVerify Project Guidelines

## Development Commands
- Setup environment: `uv sync && uv pip install -r requirements.txt && uv venv && source .venv/bin/activate`
- Run example: `python examples/example_tool_use.py`
- Install linting tools: `pip install black ruff mypy`
- Format code: `black src/`
- Lint code: `ruff src/`
- Type check: `mypy src/`

## Code Style Guidelines
- Use Python 3.11+ features (match-case statements, typing)
- Add type annotations everywhere, use Pydantic for data validation
- Add docstrings with examples and typing information
- Organize imports: standard library, then third-party, then local
- Use BaseTool for tool implementations
- Follow functional approach with clear separation of concerns
- Handle errors with try/except blocks with specific error messages
- Use snake_case for variables and functions, PascalCase for classes
- Support static type checking (py.typed file is present)
- Prefix private functions with underscore

## XML Grammar and Documentation System

### Grammar Generation
- The system generates XML-based GBNF grammar for Pydantic models
- Grammar consists of named rules for each model and field type
- Container types (lists, sets) are handled with nested item tags
- Union types support multiple alternative structures

### Type System
- Uses recursive type analysis to handle nested structures
- Supports: primitives, containers (list/set/dict), unions, models, literals, enums
- Correctly unwraps Annotated types (used in ToolUse implementation)
- Dependency tracking ensures proper rule ordering

### Documentation Generation
- Generates clear, hierarchical documentation of models and fields
- Nested models are documented at top level with proper indentation
- Container items are properly referenced with their types
- Union options are clearly listed
- Preserves model descriptions and field metadata
