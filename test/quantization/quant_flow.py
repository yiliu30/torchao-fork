"""

## Affine Quantization
Affine quantization refers to the type of quantization that maps from floating point numbers to quantized numbers (typically integer) with an affine transformation,
i.e.: `quantized_val = float_val / scale + zero_point` where `scale` and `zero_point` are quantization parameters for some granularity and based on some data.

### Quantization Primitives
We used to have different quantize and dequantize operators for quantization with different granularities.
But in the end these can all be expressed with a `block_size` argument with different settings,
so we unified existing quant primitives to `choose_qparams_affine`, `quantize_affine` and `dequantize_affine`
that can represent symmetric/asymmetric per tensor/channel/token/channel_group quantization,
this can be used to implement the unified quantized tensor subclass.

### Quantized Tensor Subclass
We also have a unified quantized tensor subclass that implements how to get a quantized tensor from floating point tensor and what does it mean to call linear ops on an instance of the tensor, e.g. `F.linear` and `aten.addmm`, with this we could dispatch to different operators (e.g. `int4mm` op) based on device (cpu, cuda) and quantization settings (`int4`, `int8`) and also packing formats (e.g. format optimized for cpu int4 mm kernel)

### Quantization Flow
What we need to do afterwards is roughly the following

```
for n, m in model.named_modules():
    # or use some filter_fn
    if isinstance(m, torch.nn.Linear):
        # optional filtering for module name, shape etc.
        # quantization activation (needed by dynamic quantization)
        # m.weight = nn.Parameter(to_laq(m.weight, device=..., layout=..., ...))
        m.weight = nn.Parameter(to_aq(m.weight, device=..., layout=..., ...))
```
The model/tensor subclass should also be compatible with AOTI and torch.export, currently we can support
`torch.export.export` and `torch.aot_compile` with the following workaround:
```
from torchao.quantization.utils import unwrap_tensor_subclass
m_unwrapped = unwrap_tensor_subclass(m)


"""

import torch
from torchao.dtypes.aqt import AffineQuantizedTensor
from torchao.quantization.quant_api import quantize
from torchao.quantization.quant_primitives import MappingType, ZeroPointDomain
from torchao.dtypes import to_aq
from torch._inductor.runtime.runtime_utils import do_bench_gpu
import copy


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=64, n=32, k=64):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)
        self.linear2 = torch.nn.Linear(n, k, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# weight settings

groupsize = 32
mapping_type = MappingType.ASYMMETRIC
block_size = (1, groupsize)
target_dtype = torch.int32
quant_min = 0
quant_max = 15
eps = 1e-6
preserve_zero = False
zero_point_dtype = torch.bfloat16
zero_point_domain = ZeroPointDomain.FLOAT

dtype = torch.bfloat16
m = ToyLinearModel(1024, 1024, 1024).eval().to(dtype).to("cuda")
m_bf16 = copy.deepcopy(m)
example_inputs = m.example_inputs(dtype=dtype, device="cuda")

m_bf16 = torch.compile(m_bf16, mode="max-autotune")


def apply_weight_quant(weight) -> AffineQuantizedTensor:
    return to_aq(
        weight,
        mapping_type,
        block_size,
        target_dtype,
        quant_min,
        quant_max,
        eps,
        zero_point_dtype=zero_point_dtype,
        preserve_zero=preserve_zero,
        zero_point_domain=zero_point_domain,
    )


m = quantize(m, apply_weight_quant)

torch._inductor.config.force_fuse_int_mm_with_mul = True
torch._inductor.config.use_mixed_mm = True

# compile the model to improve performance
m = torch.compile(m, mode="max-autotune")


# benchmark to see the speedup
from torch.utils.benchmark import Timer


def benchmark(f, *args, **kwargs):
    t0 = Timer(stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f})
    # blocked_autorange doesn't check for variance in times and would often only run the model a single
    # time, as a result many unstable times were showing up. adaptive_autorange solves the issue by checking
    # whether the IQR/median < .03 and repeating if not.
    res = t0.adaptive_autorange(0.03, max_run_time=20)
    return res.median * 1e3


bf16_time = benchmark(m_bf16, *example_inputs)
print(f"bf16 median time: {bf16_time}")
int4_time = benchmark(m, *example_inputs)
print(f"int4 weight only quantized median time: {int4_time}")
print(f"speedup: {bf16_time / int4_time}")


# output
# bf16 median time: 0.5524866282939911
# int4 weight only quantized median time: 0.47659454867243767
# speedup: 1.1592382452400098
