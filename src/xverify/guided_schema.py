from pyexpat import ExpatError
from typing import Callable, Literal
import warnings
from pydantic import BaseModel, ValidationError
from xml.etree.ElementTree import ParseError
from vllm.sampling_params import GuidedDecodingParams, SamplingParams, RequestOutputKind

from .tool_use import run_tools
from .xml import generate_gbnf_grammar_and_documentation, parse_xml_to_model


class GuidedSchema:
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
        tool_response_func: Callable = run_tools,
        max_steps: int = 10,
    ):
        self.model = model
        self.schema = schema
        self.tool_response_func = tool_response_func
        self.max_steps = max_steps
        self.gbnf, self.doc = generate_gbnf_grammar_and_documentation([self.model])

    @staticmethod
    def contents(trajectory: list[dict], role: str = "assistant") -> list[str]:
        return [c["content"] for c in trajectory if c["role"] == role]

    def parse(self, message: str, raise_error: bool = False) -> BaseModel | None:
        try:
            match self.schema:
                case "json":
                    return self.model.model_validate_json(message)
                case "xml":
                    return parse_xml_to_model(self.model, message)
                case _:
                    raise ValueError(f"Invalid schema: {self.schema}")
        except (ValidationError, ParseError, ExpatError) as e:
            if raise_error:
                raise e
            return None

    def is_completed(self, trajectory: list[dict]) -> bool:
        return len(self.contents(trajectory, role="assistant")) >= self.max_steps

    def parse_reward_func(self, reward_weight: float = 1.0) -> Callable:
        """
        Reward function that checks if the output is a valid structured output.

        The only way the model does not generate structured output (and fails the check)
        is if it generates a string longer than the max number of tokens.

        I don't think this is strictly necessary, since if we base all other rewards on
        the structured output, then the model will generate valid outputs to get any reward
        at all. And it has to try quite hard _not_ to generate valid outputs.
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

    @staticmethod
    def _warn_and_remove_keys(kwargs: dict, keys: set[str]) -> dict:
        for key in keys:
            if key in kwargs:
                warnings.warn(f"GuidedSchema sets {key}")
                del kwargs[key]
        return kwargs

    def guided_decoding_params(self, **kwargs) -> GuidedDecodingParams:
        """
        Wraps GuidedDecodingParams

        - Sets json and grammar based on schema
        """
        return GuidedDecodingParams(
            json=self.model.model_json_schema() if self.schema == "json" else None,
            grammar=self.gbnf if self.schema == "xml" else None,
            backend="xgrammar",
            **self._warn_and_remove_keys(kwargs, {"json", "grammar", "backend"}),
        )

    def sampling_params(
        self,
        max_tokens: int = 512,
        guided_decoding: dict = {},
        **kwargs,
    ) -> SamplingParams:
        """
        Wraps SamplingParams
        """
        return SamplingParams(
            max_tokens=max_tokens,
            guided_decoding=self.guided_decoding_params(**guided_decoding),
            output_kind=RequestOutputKind.FINAL_ONLY,
            **self._warn_and_remove_keys(kwargs, {"output_kind"}),
        )
