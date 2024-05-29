import torch
from torchao import quantization
import torchao.quantization.quant_primitives as ao_prim
from typing import List, Any, Dict, Optional, Union, Tuple
from torchao.quantization import GPTQ
import torchao as ao
import logging

import auto_round.utils

logger = logging.getLogger(__name__)
from torchao.quantization.utils import (
    _apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    _fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)

class NBitsWeightOnlyQuantizer:
    pass


# Internal implementation
from dataclasses import dataclass


import torch
def groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):
    scales, zeros = ao_prim.get_groupwise_affine_qparams(w, n_bit, groupsize, dtype)
    w_int4x8 = ao_prim.groupwise_affine_quantize_tensor_from_qparams(
        w, scales, zeros, n_bit, groupsize
    )
    scales_and_zeros = ao_prim.pack_tinygemm_scales_and_zeros(scales, zeros)
    return w_int4x8, scales_and_zeros


device = "cuda"
in_features = 1024
out_features = 1024
inner_k_tiles = 4
groupsize = 128
dtype = torch.bfloat16
linear = torch.nn.Linear(in_features, out_features, device=device)
input = torch.randn(1, in_features, device=device)
output = linear(input)
w = linear.weight
input_int4x8, scales_and_zeros = groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=groupsize, dtype=dtype)
int_data = torch.ops.aten._convert_weight_to_int4pack(input_int4x8, inner_k_tiles)



transpose = False
woq_weight = quantization.subclass.Int4WeightOnlyQuantizedLinearWeight(int_data, scales_and_zeros, False, w.shape, groupsize=groupsize, inner_k_tiles=inner_k_tiles, dtype=torch.bfloat16)
linear.weight = torch.nn.Parameter(woq_weight)
new_out = linear(input)
SQNR_diff = SQNR(new_out, output)
print(f"SQNR_diff = {SQNR_diff}")
print(f"diff = {torch.amax(new_out - output)}")
print(woq_weight)
    # @classmethod
    # def to_qtensor_components(cls, input_float, groupsize=128, inner_k_tiles=8):
    #     assert groupsize in [256, 128, 64, 32]
    #     assert inner_k_tiles in [8, 4, 2]
    #     orig_out_features, orig_in_features = input_float.shape

    #     # padding
    #     in_features = find_multiple(orig_in_features, 1024)
    #     out_features = find_multiple(orig_out_features, 8)
    #     input_float = torch.nn.functional.pad(
    #         input_float,
    #         (0, in_features - orig_in_features, 0, out_features - orig_out_features),
    #     )

    #     # quantization and packing
    #     input_int4x8, scales_and_zeros = groupwise_affine_quantize_tensor(
    #         input_float, 4, groupsize, dtype=input_float.dtype
    #     )
    #     int_data = aten._convert_weight_to_int4pack(input_int4x8, inner_k_tiles)
    #     return int_data, scales_and_zeros, False, groupsize, inner_k_tiles
    
    
        #     int_data, scales_and_zeros, transposed, groupsize, inner_k_tils = cls.to_qtensor_components(input_float, groupsize, inner_k_tiles)
        # return cls(
        #     int_data,
        #     scales_and_zeros,
        #     transposed,
        #     input_float.shape,
        #     groupsize,
        #     inner_k_tiles,
        #     dtype=dtype,
        # )