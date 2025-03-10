# %%
from pydantic import BaseModel
from xverify import Env, ToolUse, run_tools


class ReasoiningToolResult(BaseModel):
    """The result of a reasoning tool"""

    reasoning: str
    tool_use: list[ToolUse[calculator, search]]


env = Env(model=ReasoiningToolResult)


# %%
