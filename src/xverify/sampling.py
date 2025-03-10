from typing import Callable
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel

from xverify import run_tools


class Structured:
    """
    Notation:
    - completions: batch of chat messages list[list[dict]]
    - completion: list of chat messages list[dict]
    - messages: list of messages list[str]

    A completion is completed when either:
    - `env_response` returns None -- no tool is used
    - `is_completed` returns True -- max steps reached
    """

    def __init__(self, model: type[BaseModel], max_steps: int = 10):
        self.model = model
        self.max_steps = max_steps

    @staticmethod
    def contents(completion: list[dict], role: str = "assistant") -> list[str]:
        return [c["content"] for c in completion if c["role"] == role]

    def parse(self, message: str) -> BaseModel:
        return self.model.model_validate_json(message)

    def env_response(self, completion: list[dict]) -> dict | None:
        last_message = completion[-1]
        assert last_message["role"] == "assistant", "should be assistant"
        parsed = self.parse(last_message["content"])
        return run_tools(parsed)

    def is_completed(self, completion: list[dict]) -> bool:
        return len(self.contents(completion, role="assistant")) >= self.max_steps

    def parse_reward_func(self, reward_weight: float = 1.0) -> Callable:
        """
        Reward function that checks if the output is a valid structured output.
        """

        def reward_func(completions, **kwargs) -> list[float]:
            def parse_completion(completion) -> float:
                assistant_messages = self.contents(completion, role="assistant")
                return sum(
                    reward_weight if self.parse(m) is not None else 0.0
                    for m in assistant_messages
                ) / len(assistant_messages)

            return [parse_completion(c) for c in completions]

        return reward_func

    def guided_decoding_args(self, **kwargs):
        assert "json" not in kwargs, "Parser handles json"
        return GuidedDecodingParams.from_optional(
            json=self.model.model_json_schema(),
            # whitespace_pattern=r"[\n ]?",
            # backend="outlines",
            **kwargs,
        )

    def sampling_args(self, **kwargs):
        return {
            "guided_decoding": self.guided_decoding_args(
                **kwargs.pop("guided_decoding", {})
            ),
            "n": 1,
            "include_stop_str_in_output": False,
            **kwargs,
        }
