# %%
from pydantic import BaseModel

from xverify import ToolUse, run_tools
from xverify.src.xverify.tools import calculator, search


class ReasoiningToolResult(BaseModel):
    """The result of a reasoning tool"""

    reasoning: str
    tool_use: list[ToolUse[calculator, search]]


res = ReasoiningToolResult.model_validate(
    {
        "reasoning": "I need to add three numbers",
        "tool_use": [
            {
                "tool_name": "calculator",
                "expression": "2 + 2",
            },
            {
                "tool_name": "search",
                "query": "What is the capital of France?",
            },
        ],
    }
)


class B(BaseModel):
    a: int
    b: int


b = B(a=1, b=2)


run_tools(res)

# res.tool_use.run_tool()

# %%


res2 = ReasoiningToolResult.model_validate(
    {
        "reasoning": "I need to add three numbers",
        "tool_use": {
            "tool_name": "calculator",
            "expression": "2 + 4",
        },
    }
)
res2.tool_use.run_tool()
# %%

from pydantic import Field


class A(BaseModel):
    a: int = Field(..., description="a is a number")
    b: int


A.model_json_schema()


# %%
from fastcore.docments import docments

docs = docments(calculator, full=True)
docs

# %%
