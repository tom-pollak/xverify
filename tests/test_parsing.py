# %%
import xmltodict
from vllm import LLM
from pydantic import BaseModel

from xverify import Env, ToolUse
from xverify.tools import calculator, search


class MultiToolUse(BaseModel):
    tool_use: list[ToolUse[calculator, search]]


# %%

if "llm" not in globals():  # interactive use
    llm = LLM(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        max_model_len=2000,
        gpu_memory_utilization=0.5,
        enforce_eager=True,
    )


env = Env(MultiToolUse)
print(env.doc)

print(env.gbnf)

print("*" * 80)


outp = llm.generate(  # type: ignore
    f"Use 2 tools! first tool is calculator, calculate 2 + 2, second tool is search, search for the capital of France\n{env.doc}",
    sampling_params=env.sampling_params(temperature=2.),
    use_tqdm=False,
)
text = outp[0].outputs[0].text
print(text)
print()
assert env.parse(outp[0].outputs[0].text) is not None
