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
    w_int4x8_2 = _rounder_groupwise_affine_quantize_tensor_from_qparams(qdq_weight, scales, zeros, n_bit, groupsize)
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

    w_int4x8 = to_quant.div(scales).round().add(zeros).clamp_(min_int, max_int).to(torch.int32).reshape_as(w)

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
import copy

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

