"""
torch.Tensor: QTensor
        - layout: layout <-> impl, e.g, MarlinQQQLayout <-> MarlinQQQAQTTensorImpl
 -> torch.nn.functional.linear
    -> torch.ops.aten._weight_int4pack_mm
    -> _linear_int8_act_int4_weight_marlin_qqq_impl -> marlin_qqq_gemm



[WOQ]
model.linear1: torch.nn.Linear                                             [Torch]
    - weight: AffineQuantizedTensor                                        [AO]
        -> torch.nn.functional.linear                                      [Torch]
            -> _quantized_linear_op: function                              [AO]
                -> (option1) torch.ops.aten._weight_int4pack_mm            [Torch]
                -> (option2) _linear_int8_act_int4_weight_marlin_qqq_impl  [AO]
                    -> marlin_qqq_gemm
                ...
        -> torch.mm                                                        [Torch]
            ...


[MXFP8]
torch.matmul                                    [Torch]
    - mat1: MXTensor                            [AO]
        -> torch.ops.aten.mm                    [Torch]
            -> mx_mm: function                  [AO]
                -> torch.ops.aten.mm            [Torch]
                -> ....

[Standard FP8]
1) W8A8, need H100
2) W16A8, dequantize weight to float

"""

import logging
import logging.config
import logging

logging.basicConfig(level=logging.INFO)

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

from torchao.quantization import quantize_, int4_weight_only

torch.set_default_tensor_type(torch.BFloat16Tensor)
dim = 10
example_input = torch.randn(2, dim)
model = TwoLiear(dim).eval()
float_out = model(example_input)
quantize_(model, int4_weight_only(group_size=32))
breakpoint()
print(f"qmodel : {model}")
qmodel_out = model(example_input)
print(f"float_out: {float_out}")
print(f"qmodel_out: {qmodel_out}")
maxdiff = (float_out - qmodel_out).abs().max()
print(f"maxdiff: {maxdiff}")


# with torch.no_grad(), torch.device("cpu"):
#     torch.set_default_tensor_type(torch.BFloat16Tensor)
#     dim = 1024
#     example_input = torch.randn(2, dim)
#     model = TwoLiear(dim).eval()
#     float_out = model(example_input)
#     quantize_(model, int4_weight_only(group_size=32))
#     breakpoint()
#     print(f"qmodel : {model}")
#     qmodel_out = model(example_input)
#     print(f"float_out: {float_out}")
#     print(f"qmodel_out: {qmodel_out}")
#     maxdiff = (float_out - qmodel_out).abs().max()
#     print(f"maxdiff: {maxdiff}")

"""
> /home/user/torchao/torchao/dtypes/uintx/tensor_core_tiled_layout.py(80)_linear_bf16_act_uint4_weight_impl()
-> if is_device(input_tensor.device.type, "cpu") and TORCH_VERSION_AT_LEAST_2_6:
(Pdb) bt
  /home/user/torchao/example/woq_example.py(31)<module>()
-> qmodel_out = model(example_input)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1736)_wrapped_call_impl()
-> return self._call_impl(*args, **kwargs)
  /home/user/miniforge3/envs/ao/lib/python3.11/site-packages/torch/nn/modules/module.py(1747)_call_impl()
-> return forward_call(*args, **kwargs)
  /home/user/torchao/example/woq_example.py(12)forward()
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
  /home/user/torchao/torchao/dtypes/affine_quantized_tensor_ops.py(163)_()
-> return weight_tensor._quantized_linear_op(input_tensor, weight_tensor, bias)
  /home/user/torchao/torchao/dtypes/affine_quantized_tensor_ops.py(96)_quantized_linear_op()
-> return impl(input_tensor, weight_tensor, bias)
> /home/user/torchao/torchao/dtypes/uintx/tensor_core_tiled_layout.py(80)_linear_bf16_act_uint4_weight_impl()
-> if is_device(input_tensor.device.type, "cpu") and TORCH_VERSION_AT_LEAST_2_6:

"""
