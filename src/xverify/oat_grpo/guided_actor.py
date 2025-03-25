import logging
import time
from dataclasses import dataclass

import vllm
from oat.algorithms.ppo import PPOActor, PPOArgs

from xverify.guided_schema import GuidedSchema


class MultiStepPPOActor(PPOActor):
    def __init__(
        self, schema: GuidedSchema, ipc_server, vllm_args, args: PPOArgs
    ) -> None:
        super().__init__(ipc_server, vllm_args, args)
        self.sampling_params = schema.sampling_params(
            **schema.serialize_sampling_params(self.sampling_params)
        )
        # TODO: add self.oracle

    @dataclass
    class BatchCandidate:
        candidates: list[list[str]]
        win_probs: list[list[float]]

    def extract_multi_candidates_from_output(
        self,
        outputs: list[vllm.RequestOutput],
        n_output_seqs: int,
        strip: bool = True,
        last_only: bool = True,
    ) -> list[list[str]]:
        candidates = []
        for i in range(len(outputs)):
            # for each prompt
            candidates.append([])
            for k in range(n_output_seqs):
                # for each response
                text = outputs[i].outputs[k].text
                if strip:
                    text = text.strip()
                candidates[i].append(text)
        return candidates

    def generate_and_maybe_eval(
        self,
        prompts: list[str],
        formatted_prompts: list[str],
        references: list[str] | None = None,
    ):
        """
        1) Generate responses for given prompts;
        2) Optionally evaluate the win rate over references based on the oracle reward model.
        """
        assert self.eval_mode
        outputs = self.generate(formatted_prompts, self.eval_sampling_params)
        candidates = self.extract_multi_candidates_from_output(
            outputs, self.eval_sampling_params.n
        )
        responses = []
        for j in range(self.eval_sampling_params.n):
            responses.extend([candidates[i][j] for i in range(len(prompts))])

        win_probs = None
        if references:
            logging.debug(f"Evaluating using oracle {self.oracle}")
            st = time.time()
            win_probs, _ = self.oracle.compare(
                prompts * self.eval_sampling_params.n,
                responses,
                references * self.eval_sampling_params.n,
                batch_size=self.oracle_batch_size,
                return_probs=True,
                disable_tqdm=True,
            )
            logging.debug(f"Time elapse {time.time() - st}")
        reshaped_responses = []
        for x_i in range(len(prompts)):
            reshaped_responses.append(
                [responses[y_i] for y_i in range(x_i, len(responses), len(prompts))]
            )
        reshaped_win_probs = win_probs.reshape(
            self.eval_sampling_params.n, len(prompts)
        ).transpose(1, 0)
        return reshaped_responses, reshaped_win_probs

    def generate(self, prompts: list[str], sampling_params: vllm.SamplingParams):
        def _strip_bos(prompts: list[str]):
            if self.tokenizer.bos_token:  # type: ignore
                # lstrip bos_token because vllm will add it.
                prompts = [p.lstrip(self.tokenizer.bos_token) for p in prompts]  # type: ignore
            return prompts

        def _check_bos(outputs):
            if self.tokenizer.bos_token:  # type: ignore
                # make sure vllm added bos_token.
                assert self.tokenizer.bos_token_id in outputs[0].prompt_token_ids  # type: ignore

        self.generate_mode = True
        prompts = _strip_bos(prompts)
        outputs = self.llm.generate(
            prompts, sampling_params=sampling_params, use_tqdm=False
        )
        _check_bos(outputs)
        self.generate_mode = False
        return outputs
