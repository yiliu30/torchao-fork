import torch

torch.manual_seed(0)
from typing import Union, Tuple
from torch.utils._python_dispatch import return_and_correct_aliasing


from torchao.quantization.quant_primitives import dynamically_quantize_per_channel

from torchao.quantization.utils import logger


def quant_tensor_fn(
    tensor: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # TODO: align with HQQ
    return dynamically_quantize_per_channel(
        tensor, quant_min=-128, quant_max=127, target_dtype=torch.int8
    )


class DoubleQuantLinearWeight(torch.Tensor):

    @staticmethod
    def __new__(cls, int_data, scale, zero, transposed, shape, *args, **kwargs):
        kwargs["requires_grad"] = False
        # TODO: study more `_make_wrapper_subclass`
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(
        self,
        int_data: torch.Tensor,
        scale: Union[torch.Tensor, "DoubleQuantLinearWeight"],
        zero: Union[torch.Tensor, "DoubleQuantLinearWeight"],
        transposed,
        shape,
    ) -> None:
        logger.info(
            "Create new `DoubleQuantLinearWeight` with int_data shape: %s, scale shape: %s, zero shape: %s",
            int_data.shape,
            scale.shape,
            zero.shape,
        )
        self.int_data = int_data
        self.scale = scale
        self.zero = zero
        self.transposed = transposed

    @classmethod
    def from_float(cls, input_float, quant_scale, quant_zero):
        logger.info("input_float: %s", input_float)
        w_int_repr, scale, zero = quant_tensor_fn(input_float)
        logger.info(
            "w_int_repr shape: %s, scale shape: %s, zero shape: %s",
            w_int_repr.shape,
            scale.shape,
            zero.shape,
        )
        logger.info("scale: %s, zero: %s", scale, zero)
        int_data = w_int_repr.contiguous().t()
        if quant_scale:
            scale = DoubleQuantLinearWeight.from_float(
                scale, quant_scale=False, quant_zero=False
            )
        if quant_zero:
            zero = DoubleQuantLinearWeight.from_float(
                zero, quant_scale=False, quant_zero=False
            )
        return cls(int_data, scale, zero, transposed=False, shape=input_float.shape)

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):

        # Handle the `mm` and `addmm`
        if func in [torch.ops.aten.mm.default, torch.ops.aten.addmm.default]:
            if func == torch.ops.aten.addmm.default:
                mat1, w_qtensor, bias = args[1], args[2], args[0]
            else:
                mat1, w_qtensor, bias = args[0], args[1], None
            return cls._handle_mm_or_addmm(mat1, w_qtensor, bias)

        # * the `detach` will be called when create the `torch.nn.Parameter`
        if func is torch.ops.aten.detach.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._detach())

        # * Handle the `transpose` which performed under forward
        if func is torch.ops.aten.t.default:
            return return_and_correct_aliasing(func, args, kwargs, args[0]._transpose())

    def _transpose(self):
        logger.info("Calls `_transpose`")
        return self.__class__(
            self.int_data,  # * Not real transpose data
            self.scale,
            self.zero,
            transposed=not self.transposed,  # * Update the `transposed` and `shape` used by external representation
            shape=self.shape[::-1],
        )

    def _detach(self):
        logger.info("Calls `_detach`")
        return self.__class__(
            self.int_data, self.scale, self.zero, self.transposed, shape=self.shape
        )

    @classmethod
    def dequantize(cls, qtensor: "DoubleQuantLinearWeight") -> torch.Tensor:
        # Dequantize a `DoubleQuantLinearWeight` to `torch.Tensor
        logger.info("qtensor.int_data shape is : %s", qtensor.int_data.shape)
        int_data = qtensor.int_data
        scale = qtensor.scale
        zero = qtensor.zero
        if isinstance(scale, DoubleQuantLinearWeight):
            scale = cls.dequantize(scale)
        if isinstance(zero, DoubleQuantLinearWeight):
            zero = cls.dequantize(zero)
        logger.info(
            f"shapes of int_data, scale, zero: {int_data.shape}, {scale.shape}, {zero.shape}"
        )
        # TODO: handle `zero` is not zero, double check the correctness
        return (int_data - zero) * scale

    @classmethod
    def _handle_mm_or_addmm(
        cls, input: torch.Tensor, other: "DoubleQuantLinearWeight", bias
    ) -> torch.Tensor:
        # https://pytorch.org/docs/stable/generated/torch.mm.html
        # https://pytorch.org/docs/stable/generated/torch.bmm.html
        qdq_weight = cls.dequantize(other)
        output = torch.matmul(input, qdq_weight)
        if bias:
            output = output + bias
        return output

    def __repr__(self):
        return self.___repr__(1)

    def _get_scale__repr(self, indent):
        if isinstance(self.scale, DoubleQuantLinearWeight):
            return self.scale.___repr__(indent + 1)
        else:
            return self.scale.__repr__()

    def _get_zero__repr(self, indent):
        if isinstance(self.zero, DoubleQuantLinearWeight):
            return self.zero.___repr__(indent + 1)
        else:
            return self.zero.__repr__()

    def ___repr__(self, indent):
        return (
            f"{self.__class__.__name__}(\n"
            f"{'    '*indent}" + f"data={self.int_data},\n"
            f"{'    '*indent}" + f"shape={self.shape}, \n"
            # f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad},\n"
            f"\n{'    '*indent}" + f"scale={self._get_scale__repr(indent)} \n"
            f"{'    '*indent}" + f"zero={self._get_zero__repr(indent)} \n"
        )


device = "cpu"
in_feats = 32
out_feats = 64
lin = torch.nn.Linear(
    in_features=in_feats, out_features=out_feats, bias=False, device=device
)
bs = 4
input = torch.randn(bs, in_feats, device=device)
float_ref = lin(input)


logger.info(f"lin.weight shape: {lin.weight.shape}")
new_dd_weight = DoubleQuantLinearWeight.from_float(
    lin.weight, quant_scale=True, quant_zero=False
)

lin.weight = torch.nn.Parameter(data=new_dd_weight, requires_grad=False)

output = lin(input)
logger.info(output.shape)

amax = torch.max(torch.abs(float_ref - output))
logger.info(amax)

assert torch.allclose(
    float_ref, output, atol=0.01
), f"Not allclose, the max diff is: {amax}"

torch.save(lin.state_dict(), "lin.double_quant.pt")

re_loaded_lin_state_dict = torch.load("lin.double_quant.pt")
re_loaded_lin = torch.nn.Linear(
    in_features=in_feats, out_features=out_feats, bias=False, device=device
)
# TODO: reload the state dict back to linear
# re_loaded_lin.load_state_dict(re_loaded_lin_state_dict)
# re_output = re_loaded_lin(input)
# print(re_output)
# assert torch.allclose(output, re_output)
