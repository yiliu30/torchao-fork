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

from torchao.quantization import (
    quantize_,
    float8_static_activation_float8_weight,
    float8_weight_only,
)

with torch.no_grad(), torch.device("cuda"):
    torch.set_default_tensor_type(torch.BFloat16Tensor)
    dim = 1024
    example_input = torch.randn(2, dim)
    model = TwoLiear(dim).eval()
    float_out = model(example_input)
    scale = torch.tensor(1.0)
    quantize_(model, float8_static_activation_float8_weight(scale=scale))
    print(f"qmodel : {model}")
    qmodel_out = model(example_input)
    print(f"float_out: {float_out}")
    print(f"qmodel_out: {qmodel_out}")
    maxdiff = (float_out - qmodel_out).abs().max()
    print(f"maxdiff: {maxdiff}")
