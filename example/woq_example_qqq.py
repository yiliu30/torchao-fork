import torch
import torchao


class TwoLiear(torch.nn.Module):
    def __init__(self, dim):
        super(TwoLiear, self).__init__()
        self.linear1 = torch.nn.Linear(dim, dim * 2)
        self.linear2 = torch.nn.Linear(dim * 2, dim)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# torch.nn.functional.linear

# torch.ops.aten._weight_int4pack_mm
import copy

import pytest
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests

from torchao.dtypes import MarlinQQQLayout
from torchao.quantization.marlin_qqq import (
    pack_to_marlin_qqq,
    unpack_from_marlin_qqq,
)
from torchao.quantization.quant_api import (
    int8_dynamic_activation_int4_weight,
    quantize_,
)
from torchao.quantization.quant_primitives import (
    MappingType,
    choose_qparams_and_quantize_affine_qqq,
)
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

from torchao.quantization import quantize_, int4_weight_only, marlin_qqq

with torch.no_grad(), torch.device("cuda"):
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    dim = 1024
    example_input = torch.randn(2, dim)
    model = TwoLiear(dim).eval()
    float_out = model(example_input)
    group_size = 128
    quantize_(
        model,
        int8_dynamic_activation_int4_weight(
            group_size=group_size,
            mapping_type=MappingType.SYMMETRIC,
            act_mapping_type=MappingType.SYMMETRIC,
            layout=MarlinQQQLayout(),
        ),
    )

    # breakpoint()
    print(f"qmodel : {model}")
    qmodel_out = model(example_input)
    print(f"float_out: {float_out}")
    print(f"qmodel_out: {qmodel_out}")
    maxdiff = (float_out - qmodel_out).abs().max()
    print(f"maxdiff: {maxdiff}")


"""

/home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/__init__.py:1144: UserWarning: torch.set_default_tensor_type() is deprecated as of PyTorch 2.1, please use torch.set_default_dtype() and torch.set_default_device() as alternatives. (Triggered internally at ../torch/csrc/tensor/python_tensor.cpp:432.)
  _C._set_default_tensor_type(t)
qmodel : TwoLiear(
  (linear1): Linear(in_features=1024, out_features=2048, weight=LinearActivationQuantizedTensor(activation=<function _int8_symm_per_token_quant at 0x7f5ae02d9580>, weight=MarlinQQQTensor(shape=torch.Size([2048, 1024]), block_size=(1, 128), device=cuda:0, _layout=MarlinQQQLayout(), tensor_impl_dtype=torch.int32, quant_min=-8, quant_max=7)))
  (linear2): Linear(in_features=2048, out_features=1024, weight=LinearActivationQuantizedTensor(activation=<function _int8_symm_per_token_quant at 0x7f5ae02d9580>, weight=MarlinQQQTensor(shape=torch.Size([1024, 2048]), block_size=(1, 128), device=cuda:0, _layout=MarlinQQQLayout(), tensor_impl_dtype=torch.int32, quant_min=-8, quant_max=7)))
)
float_out: tensor([[ 0.4062, -0.3086, -0.0261,  ...,  0.0085, -0.2520, -0.4492],
        [-0.0498, -0.0204, -0.5117,  ..., -0.0104,  0.1377, -0.2021]],
       device='cuda:0')
qmodel_out: tensor([[ 0.3828, -0.3164, -0.0520,  ...,  0.0625, -0.2383, -0.4395],
        [-0.0549,  0.0210, -0.5430,  ..., -0.0150,  0.0786, -0.1826]],
       device='cuda:0')
maxdiff: 0.12890625

"""


"""
-> isinstance(input_tensor, AffineQuantizedTensor)
(Pdb) bt
  /home/user/torchao/example/woq_example_qqq.py(63)<module>()
-> qmodel_out = model(example_input)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1736)_wrapped_call_impl()
-> return self._call_impl(*args, **kwargs)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1747)_call_impl()
-> return forward_call(*args, **kwargs)
  /home/user/torchao/example/woq_example_qqq.py(12)forward()
-> x = self.linear1(x)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1736)_wrapped_call_impl()
-> return self._call_impl(*args, **kwargs)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1747)_call_impl()
-> return forward_call(*args, **kwargs)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/linear.py(125)forward()
-> return F.linear(input, self.weight, self.bias)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/utils/_device.py(106)__torch_function__()
-> return func(*args, **kwargs)
  /home/user/torchao/torchao/utils.py(431)_dispatch__torch_function__()
-> return cls._ATEN_OP_OR_TORCH_FN_TABLE[func](func, types, args, kwargs)
  /home/user/torchao/torchao/utils.py(410)wrapper()
-> return func(f, types, args, kwargs)
  /home/user/torchao/torchao/quantization/linear_activation_quantized_tensor.py(124)_()
-> return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
  /home/user/torchao/torchao/quantization/linear_activation_quantized_tensor.py(84)_quantized_linear_op()
-> return torch.nn.functional.linear(aqt, original_weight_tensor, bias)
  /home/user/torchao/torchao/utils.py(431)_dispatch__torch_function__()
-> return cls._ATEN_OP_OR_TORCH_FN_TABLE[func](func, types, args, kwargs)
  /home/user/torchao/torchao/utils.py(410)wrapper()
-> return func(f, types, args, kwargs)
  /home/user/torchao/torchao/dtypes/affine_quantized_tensor_ops.py(163)_()
-> return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
  /home/user/torchao/torchao/dtypes/affine_quantized_tensor_ops.py(95)_quantized_linear_op()
-> if dispatch_condition(input_tensor, weight_tensor, bias):
> /home/user/torchao/torchao/dtypes/uintx/marlin_qqq_layout.py(229)_linear_int8_act_int4_weight_marlin_qqq_check()
-> isinstance(input_tensor, AffineQuantizedTensor)

"""
