# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Dict, Optional

import torch
from torch.utils._python_dispatch import TorchDispatchMode

__all__ = [
    "find_multiple",
    "log_with_rank",
    "clear_logs",
    "compute_error",
    "apply_logging_hook",
    "get_model_size_in_bytes",
]


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


def log_with_rank(*args):
    # append
    #
    #   {thing_to_log}
    #
    # to {file}_{rank}.txt, for printing stuff from multiple GPUs
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_fname, "a") as f:
        f.write(" ".join([str(s) for s in args]) + "\n")
    if local_rank == 0:
        print(*args)


def clear_logs():
    if os.path.isfile(log_fname):
        os.remove(log_fname)


# basic SQNR
def compute_error(x, y):
    Ps = torch.linalg.norm(x)
    Pn = torch.linalg.norm(x - y)
    return 20 * torch.log10(Ps / Pn)


# logger for fqn + op + shape
# note: not safe for any kind of multithreading
_cur_fqn: Optional[str] = None


def _get_logging_hook(fqn):
    def forward_hook(module, input):
        global _cur_fqn
        _cur_fqn = fqn

    return forward_hook


def apply_logging_hook(model):
    for name, mod in model.named_modules():
        mod.register_forward_pre_hook(_get_logging_hook(name))


# collections.defaultdict printing is weird with lambdas, so hand writing for now
fqn_to_op_to_shape_to_count: Dict[
    Optional[str], Dict[Optional[str], Dict[Optional[str], int]]
] = {}


class LoggingTensorMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        global _cur_fqn
        op_name: str = f"{func.__module__}.{func.__name__}"
        shape_str = ""
        for arg in args:
            if isinstance(arg, torch.Tensor):
                shape_str += str(list(arg.shape)) + ", "
        if shape_str != "":
            shape_str = shape_str[:-2]

        if _cur_fqn not in fqn_to_op_to_shape_to_count:
            fqn_to_op_to_shape_to_count[_cur_fqn] = {}
        if op_name not in fqn_to_op_to_shape_to_count[_cur_fqn]:
            fqn_to_op_to_shape_to_count[_cur_fqn][op_name] = {}
        if shape_str not in fqn_to_op_to_shape_to_count[_cur_fqn][op_name]:
            fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] = 0
        fqn_to_op_to_shape_to_count[_cur_fqn][op_name][shape_str] += 1

        return rs


# https://discuss.pytorch.org/t/finding-model-size/130275
def get_model_size_in_bytes(model):
    s = 0
    for p in model.parameters():
        s += p.nelement() * p.element_size()
    for b in model.buffers():
        s += b.nelement() * b.element_size()
    return s


import logging
# set logger format with filename and line number
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S"
)
streamHandler = logging.StreamHandler()
streamHandler.setFormatter(formatter)

logger = logging.getLogger("torchao")
logger.handlers.clear()
logger.setLevel(logging.INFO)
logger.addHandler(streamHandler)



def inspect_arguments(func):
    def wrapper(*args, **kwargs):
        # Print function name
        print(f"Function: {func.__name__}")

        # Print positional arguments
        if args:
            print("Positional Arguments:")
            for i, arg in enumerate(args):
                if isinstance(arg, torch.Tensor):
                    print(f"Arg {i + 1}: {arg.shape}")
                else:
                    print(f"Arg {i + 1}: {arg}")

        # Print keyword arguments
        if kwargs:
            print("Keyword Arguments:")
            for key, value in kwargs.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key}: {value.shape}")
                print(f"{key}: {value}")

        # Print argument names and values
        arg_names = func.__code__.co_varnames[:func.__code__.co_argcount]
        print("Argument Names and Values:")
        for name, value in zip(arg_names, args):
            if isinstance(value, torch.Tensor):
                print(f"{name}: {value.shape}")
            else:
                print(f"{name}: {value}")

        # Call the original function
        result = func(*args, **kwargs)

        # # Print the result if needed
        # print(f"Result: {result}")

        return result

    return wrapper
