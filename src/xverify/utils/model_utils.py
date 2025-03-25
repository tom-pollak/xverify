from importlib.util import find_spec
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


def is_liger_available() -> bool:
    return find_spec("liger_kernel") is not None


def get_model(model_name: str, **model_kwargs):
    def _get_liger(nm, **kws):
        from liger_kernel.transformers import AutoLigerKernelForCausalLM  # type: ignore

        return AutoLigerKernelForCausalLM.from_pretrained(nm, **kws)

    def _get_hf(nm, **kws):
        return AutoModelForCausalLM.from_pretrained(nm, **kws)

    model_kwargs = {
        "torch_dtype": torch.bfloat16,
        "attn_implementation": "flash_attention_2",
        "use_cache": False,
        "quantization_config": BitsAndBytesConfig(load_in_4bit=True),
        **model_kwargs,
    }
    if is_liger_available():
        print("Using Liger kernel")
        try:
            return _get_liger(model_name, **model_kwargs)
        except Exception:
            print("Failed to load Liger kernel, falling back to Hugging Face")
            return _get_hf(model_name, **model_kwargs)
    else:
        return _get_hf(model_name, **model_kwargs)


def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if not hasattr(tokenizer, "pad_token"):
        print("Tokenizer does not have pad_token, setting it to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if not hasattr(tokenizer, "chat_template"):
        raise ValueError(f"Tokenizer for model {model_name} does not have chat_template attribute, \
                            and could not find a tokenizer with the same name as the model with suffix \
                            '-Instruct'. Please provide a tokenizer with the chat_template attribute.")
    return tokenizer


def get_model_and_tokenizer(model_name: str, **model_kwargs) -> tuple[Any, Any]:
    model = get_model(model_name, **model_kwargs)
    tokenizer = get_tokenizer(model_name)
    return model, tokenizer
