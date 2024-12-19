import pytest
import torch
from habana_frameworks.torch import core as htcore
from pprint import pprint


# device = torch.device("cpu")
device = torch.device("hpu")

from torchao.utils import (
    TORCH_VERSION_AT_LEAST_2_5,
)

# if not TORCH_VERSION_AT_LEAST_2_5:
#     pytest.skip("Unsupported PyTorch version", allow_module_level=True)

import copy
import io
import random
import unittest
from contextlib import nullcontext
from functools import partial
from typing import Tuple

import pytest
import torch
from torch._inductor.test_case import TestCase as InductorTestCase

# from torch.testing._internal import common_utils

from torchao.float8.float8_utils import compute_error
from torchao.quantization import (
    float8_dynamic_activation_float8_weight,
    float8_weight_only,
    quantize_,
)
from torchao.quantization.granularity import (
    PerRow,
    PerTensor,
)
from torchao.quantization.quant_api import (
    float8_static_activation_float8_weight,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_affine,
)

random.seed(0)
torch.manual_seed(0)
# torch._dynamo.config.inli


class ToyLinearModel(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_features, out_features, bias=False)
        self.linear2 = torch.nn.Linear(out_features, in_features, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


class RepeatLinearModel(torch.nn.Module):
    def __init__(self, dim=1024, num_layer=32):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [torch.nn.Linear(dim, dim, bias=False) for _ in range(num_layer)]
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def timeit(func, repeat=100):
    import time

    start = time.time()
    out = func()
    out.cpu()
    return time.time() - start


import torch


@torch.no_grad()
def gen_text(model, tokenizer, msg="", device=None, prompt="What's AI?", max_length=20):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(prompt, return_tensors="pt")
    model = model.to(device)
    new_tokens = model.generate(**inputs.to(device), max_length=max_length)
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(f"Generated text ({msg}): {text}")


def main(args):
    SCALING_FACTOR = 100
    sizes = ((128,), 256, 128)
    # multiple of SCALING_FACTOR
    dtype = torch.bfloat16
    granularity = PerTensor()
    compile = True

    M, N, K = sizes
    # Create a linear layer with bfloat16 dtype
    # if args.model == "llama":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = "facebook/opt-125m"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    # model_name = "/git_lfs/data/pytorch/llama3.1/Meta-Llama-3.1-8B"
    from optimum.habana.transformers.modeling_utils import (
        adapt_transformers_to_gaudi,
    )

    # adapt_transformers_to_gaudi()
    original_model = AutoModelForCausalLM.from_pretrained(
        model_name, attn_implementation="eager", torch_dtype=dtype
    )
    model = original_model.to(dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    input_tensor = tokenizer("Hello, my dog is cute", return_tensors="pt")[
        "input_ids"
    ].to(device)

    # else:
    #     raise ValueError(f"Unknown model {args.model}")
    print(f"Model: {model}")
    # Get a "reasonable" scale for the input tensor even though
    # we use the same scale for multiple activations
    mode_map = {}
    if args.mode == "dynamic":
        mode_map[args.mode] = partial(
            float8_dynamic_activation_float8_weight, granularity=granularity
        )
    elif args.mode == "weight-only":
        mode_map[args.mode] = float8_weight_only
    else:
        raise ValueError(f"Unknown mode {args.mode}")
    mode = "dynamic"
    prompt = "Hi, please introduce the history of AI."
    max_length = 50
    gen_text(
        model,
        tokenizer,
        "Original",
        prompt=prompt,
        max_length=max_length,
        device=device,
    )
    quantized_model = copy.deepcopy(model)
    factory = mode_map[mode]()
    quantize_(quantized_model, factory)
    gen_text(
        quantized_model,
        tokenizer,
        "Quantized",
        prompt=prompt,
        max_length=max_length,
        device=device,
    )
    if compile:
        backend = "hpu_backend" if device.type == "hpu" else "inductor"
        quantized_model = torch.compile(
            quantized_model, backend=backend, options={"keep_input_mutations": True}
        )
    gen_text(
        quantized_model,
        tokenizer,
        "Quantized Compiled",
        prompt=prompt,
        max_length=max_length,
        device=device,
    )

    # output_original = model(input_tensor)
    # output_quantized = quantized_model(input_tensor)
    if args.model == "llama":
        # decode the output
        breakpoint()
        output_quantized = model._tokenizer.decode(
            output_quantized[0], skip_special_tokens=True
        )
        output_original = model._tokenizer.decode(
            output_original[0], skip_special_tokens=True
        )

    print(f"output_original: {output_original}")
    print(f"output_quantized: {output_quantized}")
    if not args.model == "llama":
        diff = (
            output_original.to(torch.float32)
            .subtract(output_quantized.to(torch.float32))
            .abs()
            .max()
        )
        print(f"Max diff: {diff}")

    time_original = timeit(lambda: model(input_tensor))
    time_quantized = timeit(lambda: quantized_model(input_tensor))
    print(f"Time original: {time_original}")
    print(f"Time quantized: {time_quantized}")
    print(f"Speedup: {time_original/time_quantized}")


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama")
    # mode
    parser.add_argument("--mode", type=str, default="dynamic")
    # compile
    parser.add_argument("--compile", type=bool, default=True, help="Compile the model")
    # num layers
    parser.add_argument(
        "--num-layers", type=int, default=32, help="Number of layers in the model"
    )
    # dim
    parser.add_argument("--dim", type=int, default=1024, help="Dimension of the model")
    args = parser.parse_args()
    pprint(vars(args))
    main(args)