from dataclasses import dataclass

from trl import GRPOConfig

__all__ = ["GuidedGRPOConfig", "get_default_grpo_config"]

@dataclass
class GuidedGRPOConfig(GRPOConfig):
    max_steps: int = 10
    sleep_time: float = 1.0
    mask_env_response: bool = True
    max_workers: int = 10

    env_mask: int = None  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        self.env_mask = 0 if self.mask_env_response else 1


def get_default_grpo_config(
    run_name: str,
    num_gpus: int = 1,
    **kwargs,
) -> GuidedGRPOConfig:
    default_args = dict(
        output_dir=f"outputs/{run_name}",
        run_name=run_name,
        learning_rate=1e-6,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=20,
        num_train_epochs=1,
        bf16=True,
        adam_beta1=0.9,
        adam_beta2=0.99,
        max_grad_norm=0.1,
        num_iterations=1,
        beta=0.04,
        max_prompt_length=1024,
        max_completion_length=1024,
        per_device_train_batch_size=2,
        num_generations=(2 * num_gpus - 2 if num_gpus > 1 else 2),
        gradient_accumulation_steps=int(16 / num_gpus),
        gradient_checkpointing=True,
        save_strategy="steps",
        save_steps=100,
        save_only_model=True,
        use_vllm=True,
        vllm_device=f"cuda:{num_gpus-1}",
        vllm_gpu_memory_utilization=0.7 if num_gpus > 1 else 0.3,
        logging_steps=1,
        log_on_each_node=False,
        log_completions=True,
        report_to="wandb",
        reward_weights=None,
    )
    # overwrites default args with kwarg
    return GuidedGRPOConfig(**default_args, **kwargs)
