import xmltodict
from vllm import LLM
from pydantic import BaseModel

from xverify import Env, ToolUse
from xverify.tools import calculator, search

if "llm" not in globals():  # interactive use
    llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct", max_model_len=2000)


class MultiToolUse(BaseModel):
    tool_use: list[ToolUse[calculator, search]]


env = Env(MultiToolUse)
print(env.doc)

print(env.gbnf)

print("*" * 80)


outp = llm.generate(  # type: ignore
    f"Use 2 tools! first tool is calculator, calculate 2 + 2, second tool is search, search for the capital of France\n{env.doc}",
    sampling_params=env.sampling_params(),
    use_tqdm=False,
)
text = outp[0].outputs[0].text
print(text)
print()
assert env.parse(outp[0].outputs[0].text) is not None
