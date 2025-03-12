from pydantic import BaseModel
from xverify import ToolUse
from xverify.tools import calculator, search
from xverify.xml import generate_gbnf_grammar_and_documentation


class Tools(BaseModel):
    tool_use: ToolUse[calculator, search]


class MultiTool(BaseModel):
    tool_use: list[ToolUse[calculator, search]]


grammar, doc = generate_gbnf_grammar_and_documentation([Tools])
