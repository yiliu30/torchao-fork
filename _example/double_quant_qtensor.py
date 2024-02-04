import torch
from typing import Union, Tuple


def quant_tensor_fn(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Quantize a float tensor into int format, with scale and zero point
    # TODO: align with HQQ
    scale = torch.max(torch.abs(tensor))
    zero = torch.zeros_like(tensor)
    int_data = torch.round(tensor / scale)
    return int_data, zero, scale

indent_str = "    "

class DoubleQuantQTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, int_data, scale, zero, shape, *args, **kwargs):
        kwargs["requires_grad"] = False
        # TODO: study more `_make_wrapper_subclass`
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)
    
    def __init__(self,
                 int_data: torch.Tensor,
                 scale: Union[torch.Tensor, 'DoubleQuantQTensor'],
                 zero: Union[torch.Tensor, 'DoubleQuantQTensor'],
                 shape,
                 ) -> None:
        
        self.int_data = int_data
        self.scale = scale
        self.zero = zero
    
    import math
    a = math.sqrt(1)
    
    @classmethod
    def from_float(cls, weight, quant_scale, quant_zero):
        init_data, zero, scale = quant_tensor_fn(weight)
        if quant_scale:
            scale = DoubleQuantQTensor.from_float(scale, quant_scale=False, quant_zero=False)
        if quant_zero:
            zero = DoubleQuantQTensor.from_float(zero, quant_scale=False, quant_zero=False)
        return cls(init_data, scale, zero, shape=weight.shape)

    def _get_scale__repr(self, indent):
        if isinstance(self.scale, DoubleQuantQTensor):
            return self.scale.___repr__(indent+1)
        else:
            return self.scale.__repr__()
    
    def _get_zero__repr(self, indent):
        if isinstance(self.zero, DoubleQuantQTensor):
            return self.zero.___repr__(indent+1)
        else:
            return self.zero.__repr__()

    def ___repr__(self, indent):
        return (
            f"{self.__class__.__name__}(\n"
            f"{indent_str*indent}" + f"data={self.int_data},\n"
            f"{indent_str*indent}" + f"shape={self.shape}, \n"
            # f"device={self.device}, dtype={self.dtype}, requires_grad={self.requires_grad},\n"
            f"{indent_str*indent}" + f"scale={self._get_scale__repr(indent)} \n"
            f"{indent_str*indent}" + f"zero={self._get_zero__repr(indent)} \n"
        )
    
    def __repr__(self):
        return self.___repr__(1)

    __torch_function__ = torch._C._disabled_torch_function_impl
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs):
        return None
        


in_feats = 32
out_feats = 64
float_lin = torch.nn.Linear(in_features=in_feats, out_features=out_feats, bias=False)
new_dd_weight = DoubleQuantQTensor.from_float(float_lin.weight, quant_scale=True, quant_zero=True)
print(new_dd_weight)


