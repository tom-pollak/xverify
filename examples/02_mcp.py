# %%

"""Example of using the unified MCP integration with xVerify."""

from pydantic import BaseModel, Field

import xverify as xv

client = xv.MCPClient(
    "uv",
    ["run", "-m", "server"],
)

print(f"Available tools: {client.list_tools()}")

# Ways to use MCP tools:


# 1. Direct usage with XMLToolUse
class SimpleSearch(BaseModel):
    """Model for performing a web search."""

    query: str = Field(..., description="The search query")
    tool: xv.XMLToolUse[client["search"]]  # Works exactly like function-based tools


# 2. Multiple tools with unpacking
class AdvancedSearch(BaseModel):
    """Model for advanced search using multiple tools."""

    query: str = Field(..., description="The search query")
    # Multiple tools (uses the list returned by client["x", "y"])
    tool: xv.XMLToolUse[client["search", "brave_summarize"]]


# 3. JSON format with discriminator
class DiscriminatedSearch(BaseModel):
    """Model using JSON discriminator for tool selection."""

    query: str = Field(..., description="The search query")
    tool: xv.JSONToolUse[client["search", "brave_summarize"]]


# 4. Mixing MCP tools with regular function-based tools
class HybridTools(BaseModel):
    """Model mixing MCP tools with traditional function-based tools."""

    query: str = Field(..., description="The query or calculation")
    # Mix both types of tools
    tool: xv.XMLToolUse[client["search"], xv.calculator]


# Create a GuidedSchema with the model
schema = xv.GuidedSchema(SimpleSearch)

# Example XML response (in practice, would come from an LLM)
example_response = """
<SimpleSearch>
<query>What is the Model Context Protocol?</query>
<tool>
<search>
<query>What is the Model Context Protocol?</query>
<num_results>3</num_results>
</search>
</tool>
</SimpleSearch>
"""

# Parse the response
parsed = schema.parse(example_response)
print(f"\nParsed query: {parsed.query}")
print(f"Tool type: {type(parsed.tool).__name__}")

# You can run the tool directly (using the unified BaseTool implementation)
if parsed:
    print("\nTool execution would normally run here")
    # In a real application:
    # result = parsed.tool.run_tool()
    # print(f"Result: {result}")

# Check GBNF grammar and documentation
print(f"\nGrammar snippet (first 150 chars):\n{schema.gbnf[:150]}...")
print(f"\nDocumentation snippet:\n{schema.doc[:200]}...")

# Clean up
client.close()
# %%
