import torch

from torchao.prototype.mx_formats.mx_tensor import MXTensor

# Note: MX int8 is not implemented yet
from torchao.prototype.mx_formats.constants import (
    DTYPE_FP6_E2M3,
    DTYPE_FP6_E3M2,
    DTYPE_FP4,
)

x = torch.randn(32, 32, device="cuda")

# elem_dtype can be torch.float8_e4m3fn, torch.float8_e5m2, DTYPE_FP6_E2M3, DTYPE_FP6_E3M2, DTYPE_FP4
elem_dtype = torch.float8_e4m3fn

# high precision to MX, block size defaults to 32
x_mx = MXTensor.to_mx(x, elem_dtype)

# mx back to high precision
x_hp = x_mx.to_dtype(torch.float)
print(f"x_hp: {x_hp}")

from torchao.prototype.mx_formats.mx_linear import swap_linear_with_mx_inference_linear

m = torch.nn.Sequential(torch.nn.Linear(32, 32)).cuda()
elem_dtype = torch.float8_e4m3fn
block_size = 32
example_input = x
float_model_out = m(example_input)
swap_linear_with_mx_inference_linear(m, elem_dtype, block_size)
print(f"mx model: {m}")
mx_model_out = m(example_input)

print(f"float_model_out: {float_model_out}")
print(f"mx_model_out: {mx_model_out}")
max_diff = (float_model_out - mx_model_out).abs().max()
print(f"max_diff: {max_diff}")

out = torch.matmul(x_mx, x_mx)
print(out)


class TwoMatMulModel(torch.nn.Module):
    def __init__(self):
        super(TwoMatMulModel, self).__init__()

    def forward(self, x_mx, mat1, mat2):
        x_high: torch.Tensor = torch.matmul(x_mx, mat1)
        x_mx = MXTensor.to_mx(x_high, elem_dtype)
        out = torch.matmul(x_mx, mat2)
        return out


m = TwoMatMulModel().cuda()
x = torch.randn(32, 32, device="cuda")
x_mx = MXTensor.to_mx(x.clone(), elem_dtype)
mat1 = MXTensor.to_mx(x.clone(), elem_dtype)
mat2 = MXTensor.to_mx(x.clone(), elem_dtype)
float_model_out = m(x_mx, mat1, mat2)


"""
torch.matmul                                    [Torch]
    - mat1: MXTensor                            [AO]
        -> torch.ops.aten.mm                    [Torch]
            -> mx_mm: function                  [AO]
                -> torch.ops.aten.mm            [Torch]
                -> ....

"""

"""
@implements([aten.mm.default, aten.matmul.default])
def mx_mm(aten_op, args, kwargs=None):
    a = args[0]
    b = args[1]
    assert isinstance(a, MXTensor) and isinstance(b, MXTensor)
    a_hp = a.to_dtype(a._orig_dtype)
    b_hp = b.to_dtype(b._orig_dtype)
    res = aten_op(a_hp, b_hp)
    return res
"""
