from typing import Optional, Union
import msgspec

from datasets import Dataset, IterableDataset
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl.trainer.grpo_trainer import GRPOTrainer, GRPOConfig, RewardFunc
from vllm import RequestOutput

from .guided_schema import GuidedSchema


class GRPOGuidedTrainer(GRPOTrainer):
    def __init__(
        self,
        /,
        guided_schema: GuidedSchema,
        ### GRPOTrainer args ###
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,  # noqa: F821 # type: ignore
    ):
        if args:
            if not args.use_vllm:
                raise ValueError("vLLM must be enabled for GRPOEnvTrainer")
            if args.vllm_guided_decoding_regex is not None:
                raise ValueError("Guided decoding is set by GRPOEnvTrainer")

        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        self.guided_schema = guided_schema
        # update sampling params to use our guided decoding params
        self.sampling_params = self.guided_schema.sampling_params(
            **msgspec.json.decode(msgspec.json.encode(self.sampling_params))
        )
        # patch vLLM to use our generate method -- which is used by GRPOTrainer to generate trajectories
        # we can still use the original generate method under _generate
        self.llm._generate, self.llm.generate = self.llm.generate, self.llm._generate  # type: ignore

    def generate(self, prompts, sampling_params, **kwargs) -> list[RequestOutput]:
        return self.llm._generate(prompts, sampling_params=sampling_params, **kwargs)  # type: ignore

    def tool_response(self, trajectory: list[dict]) -> dict | None:
        last_message = trajectory[-1]
        if last_message["role"] != "assistant":
            # could return None here, but this is UB
            raise ValueError("Last message should be assistant")
        parsed = self.guided_schema.parse(last_message["content"])
        return (
            self.guided_schema.tool_response_func(parsed)
            if parsed is not None
            else None
        )
