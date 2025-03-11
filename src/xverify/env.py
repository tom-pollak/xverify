from typing import Callable, Literal
from pydantic import BaseModel, ValidationError
from vllm.sampling_params import GuidedDecodingParams, SamplingParams
from xverify.tool_use import run_tools

from pydantic_gbnf_grammar_generator import generate_gbnf_grammar_and_documentation

from .xml_parse import parse_xml_to_model
from xml.etree.ElementTree import ParseError


class Env:
    """
    A trajectory is completed when either:
    - `env_response` returns None -- no tool is used
    - `is_completed` returns True -- max steps reached

    Notation:
    - completions: batch of chat messages list[list[dict]]
    - trajectory: list of chat messages list[dict]
    - messages: list of messages list[str]

    """

    model: type[BaseModel]
    max_steps: int
    gbnf: str
    doc: str

    def __init__(
        self,
        model: type[BaseModel],
        schema: Literal["json", "xml"] = "xml",
        max_steps: int = 10,
    ):
        self.model = model
        self.schema = schema
        self.max_steps = max_steps
        self.gbnf, self.doc = generate_gbnf_grammar_and_documentation([self.model])

    @staticmethod
    def contents(trajectory: list[dict], role: str = "assistant") -> list[str]:
        return [c["content"] for c in trajectory if c["role"] == role]

    def parse(self, message: str) -> BaseModel | None:
        try:
            match self.schema:
                case "json":
                    return self.model.model_validate_json(message)
                case "xml":
                    return parse_xml_to_model(self.model, message)
                case _:
                    raise ValueError(f"Invalid schema: {self.schema}")
        except ValidationError as e:
            print(f"Validation error: {e}")
            return None
        except ParseError as e:
            print(f"XML parsing error: {e}")
            return None

    def env_response(self, trajectory: list[dict]) -> dict | None:
        last_message = trajectory[-1]
        assert last_message["role"] == "assistant", "should be assistant"
        parsed = self.parse(last_message["content"])
        return run_tools(parsed) if parsed is not None else None

    def is_completed(self, trajectory: list[dict]) -> bool:
        return len(self.contents(trajectory, role="assistant")) >= self.max_steps

    def parse_reward_func(self, reward_weight: float = 1.0) -> Callable:
        """
        Reward function that checks if the output is a valid structured output.
        """

        def reward_func(completions, **kwargs) -> list[float]:
            def parse_trajectory(trajectory) -> float:
                assistant_messages = self.contents(trajectory, role="assistant")
                return sum(
                    reward_weight if self.parse(m) is not None else 0.0
                    for m in assistant_messages
                ) / len(assistant_messages)

            return [parse_trajectory(c) for c in completions]

        return reward_func

    def guided_decoding_args(self, **kwargs):
        assert "json" not in kwargs, "Parser handles json"
        assert "grammar" not in kwargs, "Parser handles grammar"
        return GuidedDecodingParams.from_optional(
            json=self.model.model_json_schema() if self.schema == "json" else None,
            grammar=self.gbnf if self.schema == "xml" else None,
            **kwargs,
        )

    def sampling_params(self, **kwargs):
        guided_decoding = self.guided_decoding_args(**kwargs.pop("guided_decoding", {}))
        return SamplingParams(guided_decoding=guided_decoding, **kwargs)
