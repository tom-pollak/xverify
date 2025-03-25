"""Example of using the unified MCP integration with xVerify."""

from pydantic import BaseModel

from xverify import GuidedSchema, MCPClient, XMLToolUse, run_tools

client = MCPClient("python", ["server.py"])

print(f"Available tools: {client.list_tools()}")


class A(BaseModel):
    tool: XMLToolUse[client["add", "greeting"]]


schema = GuidedSchema(A)
print(schema.gbnf)
print("-" * 100)
print(schema.doc)

res = """
<A>
<tool>
<add>
<a>1</a>
<b>2</b>
</add>
</tool>
</A>
"""

# Parse the response
parsed = schema.parse(res)
assert parsed
print(run_tools(parsed))

res = """
<A>
<tool>
<greeting>
<name>John</name>
</greeting>
</tool>
</A>
"""

parsed = schema.parse(res)
assert parsed
print(run_tools(parsed))


res = """
<A>
<tool>
<add>
<b>2</b>
</add>
</tool>
</A>
"""

parsed = schema.parse(res)
assert not parsed


# class B(BaseModel):
#     tool: XMLToolUse[client["add2"]]



client = MCPClient("python", ["-m", "mcp_server_time", "--local-timezone=America/New_York"])

class Time(BaseModel):
    time: XMLToolUse[client["get_current_time"]]

schema = GuidedSchema(Time)
print(schema.gbnf)
print("-" * 100)
print(schema.doc)

res = """
<Time>
<time>
<get_current_time>
<timezone>
America/New_York
</timezone>
</get_current_time>
</time>
</Time>
"""

parsed = schema.parse(res)
assert parsed
print(run_tools(parsed))

