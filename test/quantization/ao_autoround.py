import torch
import torchao
from torchao import quantization
import torchao.quantization
import torchao.quantization.quant_primitives as ao_prim
from typing import List, Any, Dict, Optional, Union, Tuple
from torchao.quantization import GPTQ

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


@dataclass
class AutoRoundConfig:
    n_bit: int = 4
    groupsize: int = 128
    scale_and_zero_dtyoe: torch.dtype = torch.float32


@dataclass
class AutoRoundLayer:
    # TODO: should we use int_w directly?
    qdq_weight: torch.Tensor
    scales: torch.Tensor
    zeros: torch.Tensor
    config: AutoRoundConfig


# self.weight_config[n]["scale"] = m.scale
# self.weight_config[n]["zp"] = m.zp
# self.weight_config[n]["data_type"] = "float"
# if self.amp_dtype == torch.bfloat16:
#     self.weight_config[n]["data_type"] = "bfloat"
# self.weight_config[n]["bits"] = 16
# self.weight_config[n]["group_size"] = None
# self.weight_config[n]["sym"] = None
K_SCALE = "scale"
K_ZP = "zp"
K_DATA_TYPE = "data_type"
K_BITS = "bits"
K_GROUP_SIZE = "group_size"
K_SYM = "sym"
auto_round_weight_info: Dict[str, Dict[str, Optional[Union[torch.Tensor, int, bool]]]] = {}


@torch.no_grad()
def create_aq(float_layer, q_layer: AutoRoundLayer):
    # weight settings
    groupsize = q_layer.config.groupsize
    mapping_type = torchao.quantization.quant_primitives.MappingType.ASYMMETRIC
    block_size = (1, groupsize)
    target_dtype = torch.int32
    quant_min = 0
    quant_max = 15
    eps = 1e-6
    preserve_zero = False
    zero_point_dtype = torch.float32
    zero_point_domain = torchao.quantization.quant_primitives.ZeroPointDomain.INT

    # use 1024 so that we don't need padding
    m = float_layer.eval()  # .to(torch.bfloat16) #.to("cuda")
    m_copy = copy.deepcopy(m)
    # example_inputs = tuple(map(lambda x: x.to(torch.bfloat16).to("cuda"), m.example_inputs()))

    config = q_layer.config
    n_bit = config.n_bit
    groupsize = config.groupsize
    qdq_weight, scales, zeros = q_layer.qdq_weight, q_layer.scales, q_layer.zeros
    # import pdb; pdb.set_trace()
    w_int4x8 = ao_prim.groupwise_affine_quantize_tensor_from_qparams(qdq_weight, scales, zeros, n_bit, groupsize)
    w_int4x8_2 = _rounder_groupwise_affine_quantize_tensor_from_qparams(qdq_weight, scales, zeros, n_bit, groupsize)
    # input: torch.Tensor,
    # block_size: Tuple[int, ...],
    # scale: torch.Tensor,
    # zero_point: Optional[torch.Tensor],
    # output_dtype: torch.dtype,
    # quant_min: Optional[int] = None,
    # quant_max: Optional[int] = None,
    # zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    # !!! Error
    # w_int4x8_3 = ao_prim.quantize_affine(
    #     input=qdq_weight,
    #     block_size=(1, groupsize),
    #     scale=q_layer.scales,
    #     zero_point=q_layer.zeros,
    #     output_dtype=torch.int32,
    #     quant_min=quant_min,
    #     quant_max=quant_max,
    #     zero_point_domain=ao_prim.ZeroPointDomain.FLOAT)

    # return cls(
    #     int_data,
    #     scale,
    #     zero_point,
    #     block_size,
    #     input_float.shape,
    #     quant_min,
    #     quant_max,
    #     zero_point_domain,
    #     dtype=input_float.dtype
    # )

    #     self,
    # int_data: torch.Tensor,
    # scale: torch.Tensor,
    # zero_point: torch.Tensor,
    # block_size: Tuple[int, ...],
    # shape: torch.Size,
    # quant_min: Optional[int] = None,
    # quant_max: Optional[int] = None,
    # zero_point_domain: ZeroPointDomain = ZeroPointDomain.INT,
    # dtype=None,
    # strides=None,
    tensor = torchao.dtypes.AffineQuantizedTensor(
        int_data=w_int4x8_2,
        scale=q_layer.scales,
        zero_point=q_layer.zeros,
        block_size=block_size,
        shape=w_int4x8_2.shape,
        quant_min=quant_min,
        quant_max=quant_max,
        zero_point_domain=zero_point_domain,
        dtype=zero_point_dtype,
    )
    return tensor


def _rounder_groupwise_affine_quantize_tensor_from_qparams(
    w,
    scales,
    zeros,
    n_bit=4,
    groupsize=128,
):
    """This is tinygemm specific, we'll keep this for now"""
    assert groupsize > 1
    # needed for GPTQ single column quantize
    if groupsize > w.shape[-1] and scales.shape[-1] == 1:
        groupsize = w.shape[-1]

    assert w.shape[-1] % groupsize == 0
    assert w.dim() == 2

    to_quant = w.reshape(-1, groupsize)
    # assert torch.isnan(to_quant).sum() == 0

    scales = scales.reshape(-1, 1)
    zeros = zeros.reshape(-1, 1)

    #     min_val = zeros - scales * (2 ** (n_bit - 1))
    max_int = 2**n_bit - 1
    min_int = 0
    # w_int4x8 = (
    #     to_quant.sub(min_val)
    #     .div(scales)
    #     .round()
    #     .clamp_(min_int, max_int)
    #     .to(torch.int32)
    #     .reshape_as(w)
    # )
    # scales = scales.reshape(-1, 1)
    # zeros = zeros.reshape(-1, 1)
    min_val = zeros - scales * (2 ** (n_bit - 1))
    # max_int = 2**n_bit - 1
    # min_int = 0

    # min_val = zeros - scales * (2 ** (n_bit - 1))
    # max_int = 2**n_bit - 1
    # min_int = 0
    # w_int4x8 = (
    #     torch.round(to_quant.mul(1.0 / scales)).add(zeros).to(torch.int32).reshape_as(w)
    # )

    w_int4x8 = to_quant.div(scales).round().add(zeros).clamp_(min_int, max_int).to(torch.int32).reshape_as(w)
    # w_int4x8 = (
    #     to_quant.sub(min_val)
    #     .div(scales)
    #     .round()
    #     .clamp_(min_int, max_int)
    #     .to(torch.int32)
    #     .reshape_as(w)
    # )

    return w_int4x8


class Int4WeightOnlyAutoRoundQuantizer(quantization.Int4WeightOnlyQuantizer):
    def quantize(self, model: torch.nn.Module, *args: Any, **kwargs: Any) -> torch.nn.Module:
        state_dict, quantized_model_name_set = self._create_quantized_state_dict(model, *args, **kwargs)
        torch.save(state_dict, "auto_ao_state_dict.pt")

        def get_skip_layer_func(name_list):
            def skip_layer_func(name):
                if isinstance(name, str) and name not in name_list:
                    logger.warning(f"Skipping layer {name}")
                    return True

            return skip_layer_func

        # model = self._convert_for_runtime(model, None)
        # import pdb; pdb.set_trace()
        # model.load_state_dict(state_dict, strict=False)
        return model

    @staticmethod
    def _get_layer(
        name: str,
        weight: torch.Tensor,
        weight_info: Dict[str, Dict[str, Optional[Union[torch.Tensor, int, bool]]]],
    ):
        # TODO refactor it
        # Fetch the quantized layer by name
        import auto_round

        if not auto_round.utils.check_to_quantized(weight_info.get(name)):
            logger.warning(f"Layer {name} is not quantized")
            return None
        layer_info = weight_info.get(name)
        scales = layer_info[K_SCALE].to(torch.bfloat16)
        zeros = layer_info[K_ZP].to(torch.bfloat16)

        q_config: AutoRoundConfig = AutoRoundConfig(
            n_bit=layer_info[K_BITS], groupsize=layer_info[K_GROUP_SIZE], scale_and_zero_dtyoe=torch.float32
        )
        q_layer: AutoRoundLayer = AutoRoundLayer(weight, scales, zeros, q_config)

        return q_layer

    # def _groupwise_affine_quantize_tensor(w, n_bit=4, groupsize=128, dtype=torch.bfloat16):

    #     scales, zeros = get_groupwise_affine_qparams(w, n_bit, groupsize, dtype)
    #     w_int4x8 = ao_prim.groupwise_affine_quantize_tensor_from_qparams(
    #         w, scales, zeros, n_bit, groupsize
    #     )
    #     scales_and_zeros = ao_prim.pack_tinygemm_scales_and_zeros(scales, zeros)
    #     return w_int4x8, scales_and_zeros

    @staticmethod
    def _pack_layer(q_layer: AutoRoundLayer):
        config = q_layer.config
        n_bit = config.n_bit
        groupsize = config.groupsize
        qdq_weight, scales, zeros = q_layer.qdq_weight, q_layer.scales, q_layer.zeros
        # import pdb; pdb.set_trace()
        w_int4x8 = ao_prim.groupwise_affine_quantize_tensor_from_qparams(qdq_weight, scales, zeros, n_bit, groupsize)
        w_int4x8_2 = _rounder_groupwise_affine_quantize_tensor_from_qparams(qdq_weight, scales, zeros, n_bit, groupsize)
        # print(f"w_int4x8_2, {w_int4x8_2}")
        scales_and_zeros = ao_prim.pack_tinygemm_scales_and_zeros(scales, zeros)
        return w_int4x8_2, scales_and_zeros

    def _quantize(self, model):
        # TODO refactor this part
        from auto_round import AutoRound

        tokenizer = model._tokenizer
        bits, group_size, sym = 4, 128, False
        ##device:Optional["auto", None, "hpu", "cpu", "cuda"]
        n_samples = 20
        iters = 20
        n_blocks = 1
        rounder = AutoRound(
            model,
            tokenizer,
            bits=bits,
            group_size=group_size,
            sym=sym,
            device=None,
            iters=iters,
            n_samples=n_samples,
            n_blocks=n_blocks,
            scale_dtype="bf16",
        )
        model, weight_info = rounder.quantize()
        return model, weight_info

    def _create_quantized_state_dict(self, model: torch.nn.Module, weight_info) -> Dict[str, torch.Tensor]:
        # model, weight_info = self._quantize(model)
        # Update the state_dict with the quantized weight and scales_and_zeros
        cur_state_dict = model.state_dict()
        quantized_model_name_set = set()
        for fqn, mod in model.named_modules():
            if isinstance(mod, torch.nn.Linear):
                # assert not mod.bias
                out_features = mod.out_features
                in_features = mod.in_features
                # assert out_features % 8 == 0, "require out_features % 8 == 0"
                print(f"linear: {fqn}, in={in_features}, out={out_features}")
                assert (
                    in_features % self.groupsize == 0
                ), f"require in_features:{in_features} % self.groupsize:{self.groupsize} == 0"

                weight = mod.weight.data
                if not GPTQ._check_linear_int4_k(in_features, self.groupsize, self.inner_k_tiles):
                    if self.padding_allowed:
                        # from .utils import find_multiple
                        import torch.nn.functional as F

                        print(f"warning: {fqn} is padded to satisfy in_features % 1024 == 0")
                        padded_in_features = quantization.utils.find_multiple(in_features, 1024)
                        weight = F.pad(weight, pad=(0, padded_in_features - in_features))
                    else:
                        print(
                            f"warning: {fqn} is skipped, int4 requires that in_features is 32, 64, or is divisible by 1024, "
                            + "and that groupsize and inner_k_tiles*16 evenly divide into it"
                        )
                        continue

                q_layer = self._get_layer(fqn, weight, weight_info)
                if q_layer is None:
                    continue
                # import pdb; pdb.set_trace()
                # w_int4x8, scales_and_zeros = self._pack_layer(q_layer)
                # weight_int4pack = torch.ops.aten._convert_weight_to_int4pack(
                #     w_int4x8.to(self.device), self.inner_k_tiles
                # )
                # quantized_model_name_set.add(fqn)
                # cur_state_dict[f"{fqn}.weight"] = weight_int4pack.to(self.device)
                # cur_state_dict[f"{fqn}.scales_and_zeros"] = scales_and_zeros.to(self.device)
                tensor = create_aq(mod, q_layer)
                mod.weight = torch.nn.Parameter(tensor)
                logger.info(f"update the weight of {fqn}")

        return cur_state_dict, quantized_model_name_set


from transformers import AutoModelForCausalLM, AutoTokenizer


def load_state_dict_from_local(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    weight_config = torch.load(f"{path}/weight_config.pt")
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, weight_config, tokenizer


path = (
    "/home/yliu7/workspace/inc/3rd-party/auto-round/examples/language-modeling/opt-result/opt-125m-autoround-w4g128-qdq"
)
model, weight_config, tokenizer = load_state_dict_from_local(path)
# fqn = "model.decoder.layers.0.self_attn.k_proj"
# float_layer = model.model.decoder.layers[0].self_attn.k_proj
# float_layer.bias = None
# w = float_layer.weight
# q_layer = Int4WeightOnlyAutoRoundQuantizer._get_layer(fqn, float_layer.weight, weight_config)
# groupsize = q_layer.config.groupsize

# examples_inputs = torch.randn(1, float_layer.weight.shape[0])
# ref_out = float_layer(examples_inputs)

# device = "cpu"
# inner_k_tiles = 2
# w_int4x8, scales_and_zeros = Int4WeightOnlyAutoRoundQuantizer._pack_layer(q_layer)
# int_data = torch.ops.aten._convert_weight_to_int4pack(
#     w_int4x8.to(device), inner_k_tiles
# )
import copy

# transpose = False
# woq_weight = quantization.subclass.Int4WeightOnlyQuantizedLinearWeight(int_data, scales_and_zeros, transpose, w.shape, groupsize=groupsize, inner_k_tiles=inner_k_tiles, dtype=torch.bfloat16)
# woq_weight.dequantize()
# float_layer.weight = torch.nn.Parameter(tensor)
# new_out = float_layer(examples_inputs.to(torch.float32))
# print(f"new_out: {new_out}")
# amax_diff = torch.max(torch.abs(new_out - ref_out))
# print(f"amax_diff = {amax_diff}")
# SQNR_diff = SQNR(new_out, ref_out)
# print(f"SQNR_diff = {SQNR_diff}")
# import pdb; pdb.set_trace()
# # Usage
groupsize: int = 128
padding_allowed: bool = True
inner_k_tiles: Optional[int] = 2
# device: torch.device = torch.device("cuda")
autoround_quantizer = Int4WeightOnlyAutoRoundQuantizer(
    groupsize=groupsize, padding_allowed=padding_allowed, inner_k_tiles=inner_k_tiles
)

# from transformers import AutoModelForCausalLM, AutoTokenizer

# # model_name = "/models/opt-125m/"
# model_name = "/models/Llama-2-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# model._tokenizer = tokenizer

quantized_model = autoround_quantizer.quantize(model, weight_config)

# quantized_model = model
with torch.no_grad():
    for text in ["Hello, I am", "World, you", "Good", "Morning"][::-1]:
        quantized_model.eval()
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        output = quantized_model(input_ids)
        # decode
        logits = output[0]
        print(logits)
        print(text, "  >> ", tokenizer.decode(output[0].argmax(-1).tolist()[0]))


"""
tensor([[[ -3.5963,  -3.5982,  11.7140,  ...,  -3.6444,  -3.6610,  -3.8519],
         [-10.0658, -10.0564,  -0.7294,  ..., -10.1068,  -9.9747,  -9.8435],
         [ -9.8175,  -9.8068,  -0.0926,  ...,  -9.8808,  -9.8096,  -9.6742],
         [ -7.6226,  -7.6113,  -0.3625,  ...,  -7.7011,  -7.7173,  -7.3150],
         [ -9.5259,  -9.5162,  -3.2547,  ...,  -9.5997,  -9.5954,  -9.3634]]])
Hello, I am   >>  I, I'm a
tensor([[[ -3.5963,  -3.5982,  11.7140,  ...,  -3.6444,  -3.6610,  -3.8519],
         [ -6.7360,  -6.7298,   0.8636,  ...,  -6.6322,  -6.6539,  -6.6679],
         [-10.3545, -10.3435,  -1.6711,  ..., -10.2371, -10.1187, -10.2492],
         [ -7.5439,  -7.5319,   0.2816,  ...,  -7.5135,  -7.3525,  -7.4026]]])
World, you   >>  I's the are
tensor([[[-3.5963, -3.5982, 11.7140,  ..., -3.6444, -3.6610, -3.8519],
         [-6.2800, -6.3041, -0.5378,  ..., -6.3249, -6.2480, -6.1576]]])
Good   >>  I luck
tensor([[[-3.5963, -3.5982, 11.7140,  ..., -3.6444, -3.6610, -3.8519],
         [-6.7401, -6.7572,  0.0811,  ..., -6.6444, -6.5555, -6.5655]]])
Morning   >>  I,
======================
tensor([[[ -3.6062,  -3.6081,  11.5946,  ...,  -3.6525,  -3.6708,  -3.8574],
         [-10.0625, -10.0531,  -0.7243,  ..., -10.1034,  -9.9711,  -9.8405],
         [ -9.8150,  -9.8044,  -0.0982,  ...,  -9.8785,  -9.8070,  -9.6721],
         [ -7.6185,  -7.6072,  -0.3584,  ...,  -7.6971,  -7.7131,  -7.3116],
         [ -9.5212,  -9.5115,  -3.2538,  ...,  -9.5953,  -9.5901,  -9.3589]]])
Hello, I am   >>  I, I'm a
tensor([[[-7.2807, -7.2758,  4.1270,  ..., -7.2971, -7.1769, -7.1751],
         [-7.2786, -7.2736,  4.1283,  ..., -7.2949, -7.1748, -7.1735],
         [-7.2802, -7.2753,  4.1281,  ..., -7.2966, -7.1765, -7.1751],
         [-7.2792, -7.2743,  4.1276,  ..., -7.2955, -7.1755, -7.1741]]])
World, you   >>  ////
tensor([[[-7.2425, -7.2377,  4.1361,  ..., -7.2606, -7.1387, -7.1428],
         [-7.2413, -7.2364,  4.1413,  ..., -7.2593, -7.1375, -7.1419]]])
Good   >>  //
tensor([[[-7.2279, -7.2230,  4.1402,  ..., -7.2465, -7.1240, -7.1302],
         [-7.2268, -7.2219,  4.1458,  ..., -7.2454, -7.1230, -7.1295]]])
Morning   >>  //

"""
