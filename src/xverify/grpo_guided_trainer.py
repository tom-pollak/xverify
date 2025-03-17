import copy
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import msgspec
import torch
import wandb
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from trl import GRPOTrainer
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
from vllm import LLM, SamplingParams

from .grpo_guided_config import GuidedGRPOConfig
from .guided_schema import GuidedSchema
from .utils.logging_utils import print_prompt_completions_sample

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class GRPOGuidedTrainer(GRPOTrainer):
    args: GuidedGRPOConfig

    def __init__(
        self,
        /,
        guided_schema: GuidedSchema,
        ### GRPOTrainer args ###
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GuidedGRPOConfig] = None,
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

    def step(
        self, states: List[Dict[str, Any]], llm: LLM, sampling_params: SamplingParams
    ) -> List[Dict[str, Any]]:
        live_indices = [i for i, s in enumerate(states) if not s["completed"]]
        messages_to_step = [states[i]["messages"] for i in live_indices]
        llm_responses = llm.chat(
            messages_to_step, sampling_params=sampling_params, use_tqdm=False
        )  # type: ignore

        # for i, j in enumerate(live_indices):
        def update_state(j, llm_response):
            # sleep for 0-1 seconds to avoid rate limiting
            time.sleep(self.args.sleep_time * random.random())

            state = copy.deepcopy(states[j])  # TODO: do we deepcopy
            if len(state["prompt_ids"]) == 0:
                state["prompt_ids"] = llm_response.prompt_token_ids
            state["messages"].append(
                {"role": "assistant", "content": llm_response.outputs[0].text}
            )

            # get token lengths of env response and new completion
            total_prev_len = len(state["prompt_ids"]) + len(state["completion_ids"])
            env_response_len = len(list(llm_response.prompt_token_ids)) - total_prev_len  # type: ignore
            new_completion_len = len(llm_response.outputs[0].token_ids)

            # update completion masks
            state["completion_mask"].extend([self.args.env_mask] * env_response_len)
            state["completion_mask"].extend([1] * new_completion_len)

            # update completion ids
            state["completion_ids"] = list(llm_response.prompt_token_ids)  # type: ignore
            state["completion_ids"].extend(list(llm_response.outputs[0].token_ids))
            state["completion_ids"] = state["completion_ids"][
                len(state["prompt_ids"]) :
            ]

            env_res = self.tool_response(state["messages"])
            if (
                self.reached_max_steps(state["messages"])
                or len(state["completion_ids"]) > sampling_params.max_tokens  # type: ignore
                or env_res is None
            ):
                state["completed"] = True
                state["completion_ids"] = state["completion_ids"][
                    : sampling_params.max_tokens
                ]
                state["completion_mask"] = state["completion_mask"][
                    : len(state["completion_ids"])
                ]
            else:
                state["messages"].append(env_res)

            if not len(state["completion_mask"]) == len(state["completion_ids"]):
                print(f"{state['completed']=}")
                print(f"{state['messages']=}")
                print(f"{state['completion_mask']=}")
                print(f"{state['completion_ids']=}")
                raise ValueError(
                    f"Completion mask and completion ids are not the same length for state {j}"
                )

            return j, state

        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            results = list(
                executor.map(
                    lambda args: update_state(*args),
                    [(j, llm_responses[i]) for i, j in enumerate(live_indices)],
                )
            )

        for j, state in results:
            states[j] = state

        return states

    def generate(
        self,
        prompts: List[List[Dict[str, Any]]],
        llm: LLM,
        sampling_params: SamplingParams,
        **kwargs: Any,
    ) -> Dict[str, List[Sequence[int]] | List[str] | List[List[Dict[str, Any]]]]:
        # initialize state variables
        all_completed = False
        states = [
            {
                "messages": m,
                "prompt_messages": len(m),
                "prompt_ids": [],
                "completed": False,
                "completion_ids": [],
                "completion_mask": [],
            }
            for m in prompts
        ]

        # main loop
        while not all_completed:
            states = self.step(states, llm, sampling_params)
            all_completed = all(state["completed"] for state in states)

        completion_messages = [s["messages"][s["prompt_messages"] :] for s in states]
        completion_ids = [s["completion_ids"] for s in states]
        completion_mask = [s["completion_mask"] for s in states]
        output = {
            "ids": completion_ids,
            "messages": completion_messages,
            "mask": completion_mask,
        }
        return output

    def tool_response(self, trajectory: list[dict]) -> dict | None:
        last_message = trajectory[-1]
        if last_message["role"] != "assistant":
            # could return None here, but this is UB
            raise ValueError("Last message should be assistant")
        parsed = self.guided_schema.parse(last_message["content"])
        if parsed is None:
            return None
        tool_res = self.guided_schema.tool_response_func(parsed)
        if tool_res is None:
            return None
        return dict(role="user", content=json.dumps(tool_res, indent=2))

    def reached_max_steps(self, trajectory: list[dict]) -> bool:
        return (
            len(self.guided_schema.contents(trajectory, role="assistant"))
            >= self.args.max_steps
        )

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]  # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,  # type: ignore
        )  # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)  # type: ignore
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result["ids"]
            completion_messages = env_result["messages"]
            completion_mask = env_result["mask"]

        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id
        )  # type: ignore

        completion_mask = [
            torch.tensor(mask, device=device) for mask in completion_mask
        ]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            output_reward_func = reward_func(
                prompts=prompts, completions=completions, **reward_kwargs
            )  # type: ignore
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func, dtype=torch.float32, device=device
            )

        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(
            dim=1
        )

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  # type: ignore
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)  # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )  # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )  # type: ignore
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )  # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0)  # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__  # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )

        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):  # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }
