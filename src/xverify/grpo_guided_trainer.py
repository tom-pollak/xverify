from typing import Optional, Union, Any

from datasets import Dataset, IterableDataset
import torch
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from trl.trainer.grpo_trainer import GRPOTrainer, GRPOConfig, RewardFunc
from vllm.sampling_params import RequestOutputKind
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
        self.sampling_params.guided_decoding = (
            self.guided_schema.guided_decoding_params()
        )
        self.sampling_params.output_kind = RequestOutputKind.FINAL_ONLY
        self.llm._generate = self.llm.generate  # type: ignore
        self.llm.generate = self.generate  # type: ignore

    def generate(self, prompts, sampling_params, **kwargs) -> list[RequestOutput]:
        return self.llm._generate(prompts, sampling_params=sampling_params, **kwargs)  # type: ignore
