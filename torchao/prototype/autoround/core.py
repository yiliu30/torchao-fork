import dataclasses
import logging
from typing import Callable, Dict, Optional, Tuple

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization as ao_quant
from torchao.dtypes import TensorCoreTiledLayoutType, to_affine_quantized_static
from torchao.prototype.autoround.multi_tensor import _multi_tensor_config, MultiTensor
from torchao.quantization.quant_primitives import ZeroPointDomain
from torchao.utils import find_multiple


@ar_utils.singleton
@dataclasses.dataclass
class _AutoRoundConfig:
    bits: int = 4
    group_size: int = 128
    iters: int = 200


_auto_round_config = _AutoRoundConfig()


@ar_utils.singleton
@dataclasses.dataclass
class _OptimizationTracker:
    num_layers: int = 0
    optimized_layers: int = 0


_optimization_tracker = _OptimizationTracker()


@torch.no_grad()
def prepare_model_for_applying_auto_round_(
    model: torch.nn.Module,
    is_target_module: Callable[[torch.nn.Module, str], bool],
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    device: Optional[torch.types.Device] = None,
):

    _multi_tensor_config.accelerator_name = device

    _auto_round_config.bits = bits
    _auto_round_config.group_size = group_size
    _auto_round_config.iters = iters

    def forward_hook(
        module,
        args: Tuple[MultiTensor],
        kwargs: Dict[str, MultiTensor],
        output: Tuple[MultiTensor],
    ):
        apply_auto_round_optimization(
            module, args, kwargs, output, config=_auto_round_config
        )
        return output

    def _register_forward_hook(module: torch.nn.Module):
        forward_hook_handle = module.register_forward_hook(
            forward_hook, with_kwargs=True
        )
        module._forward_hook_handle_for_auto_round = forward_hook_handle
        _optimization_tracker.num_layers += 1
        return module

    model.eval()
    ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
        model, _register_forward_hook, is_target_module
    )


def apply_auto_round():

    def _apply_auto_round(optimized_model: torch.nn.Module):
        """Create a quantized model from the model optimized by auto-round.

        The `optimized_model` includes `Linear` layers optimized by auto-round, which includes `qdq_weight`, `scale`, `zp`.
        """

        @torch.no_grad()
        def convert_weight_to_affine_quantized_tensor(observed_linear: torch.nn.Module):
            device = observed_linear.weight.device
            scale = observed_linear.scale.to(device)
            zero_point = observed_linear.zp.to(device)

            def to_uintx_weight(input_float):
                quant_min = 0
                quant_max = _auto_round_config.bits**2 - 1
                block_size = (1, observed_linear.group_size)
                from torchao.dtypes.uintx.Uintx import UintxLayoutType
                from torchao.quantization.quant_primitives import ZeroPointDomain

                pack_dim = -1
                bit_width = _auto_round_config.bits
                layout_type = UintxLayoutType(bit_width=bit_width, pack_dim=pack_dim)
                return to_affine_quantized_static(
                    input_float=input_float,
                    scale=scale.to(input_float.dtype),
                    zero_point=zero_point,
                    block_size=block_size,
                    target_dtype=torch.uint8,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    zero_point_domain=ZeroPointDomain.INT,
                    layout_type=layout_type,
                )

            def to_int4_tinygemm_weight(input_float):
                # TODO(Yi): check the weight shape, `group_size`, and `inner_k_tiles` to make sure the tinygemm can handle it
                inner_k_tiles = 8
                quant_min = 0
                quant_max = _auto_round_config.bits**2 - 1
                # Shift the `zero_point` to align with tiny gemm.
                # The dequantization process in tiny gemm:
                #   tiny_dequant = (tiny_quant - 8) * scale + tiny_zp
                # The dequantization porcess in auto-round
                #   dequant = (quant - zp) * scale
                # To align with tiny gemm:
                #   dequant = (quant - 8 + 8 - zp) * scale
                #           = (quant - 8) * scale + (8 - zp) * scale
                #              \__/                 \______________/
                #            tiny_quant                 tiny_zp
                mid_point = (quant_max + quant_min + 1) / 2
                shifted_zero_point = (mid_point - zero_point) * scale
                block_size = (1, observed_linear.group_size)
                orig_out_features, orig_in_features = input_float.shape
                in_features = find_multiple(orig_in_features, 1024)
                out_features = find_multiple(orig_out_features, 8)
                orig_num_groups = orig_in_features // observed_linear.group_size
                new_num_groups = in_features // observed_linear.group_size
                # pad scale/zero_point from [orig_out_features, orig_num_groups] to [out_features, new_num_groups]
                pad_scale = torch.nn.functional.pad(
                    scale,
                    (
                        0,
                        new_num_groups - orig_num_groups,
                        0,
                        out_features - orig_out_features,
                    ),
                )
                pad_shifted_zero_point = torch.nn.functional.pad(
                    shifted_zero_point,
                    (
                        0,
                        new_num_groups - orig_num_groups,
                        0,
                        out_features - orig_out_features,
                    ),
                )
                return to_affine_quantized_static(
                    input_float=input_float,
                    scale=pad_scale.to(torch.bfloat16),
                    zero_point=pad_shifted_zero_point.to(torch.bfloat16),
                    block_size=block_size,
                    target_dtype=torch.int32,
                    quant_min=quant_min,
                    quant_max=quant_max,
                    zero_point_domain=ZeroPointDomain.FLOAT,
                    layout_type=TensorCoreTiledLayoutType(inner_k_tiles=inner_k_tiles),
                )

            # TODO(Yi): better way to select the weight quantization function
            if (
                _auto_round_config.bits == 4
                and observed_linear.weight.device.type == "cuda"
            ):
                weight_func = to_int4_tinygemm_weight
            else:
                weight_func = to_uintx_weight

            observed_linear.weight = torch.nn.Parameter(
                weight_func(observed_linear.weight), requires_grad=False
            )
            del observed_linear.scale
            del observed_linear.zp
            return observed_linear

        def _is_observed_linear(mod: torch.nn.Module, fqn: str):
            return hasattr(mod, "scale")

        qmodel = ao_quant.quant_api._replace_with_custom_fn_if_matches_filter(
            optimized_model,
            convert_weight_to_affine_quantized_tensor,
            _is_observed_linear,
        )
        return qmodel

    return _apply_auto_round


@torch.no_grad()
def _apply_auto_round_optimization(
    block, grouped_args, spec, block_outputs, config: _AutoRoundConfig
):
    # Call the auto-round to execute the optimization process.
    # https://github.com/intel/auto-round/tree/patch-for-ao-2
    # TODO(Yi), make the branch more stable
    if ar_utils.is_auto_round_available():
        import auto_round
    else:
        raise ImportError(
            (
                "This example requires the `auto-round` library."
                "Please install it with `pip install https://github.com/intel/auto-round.git@patch-for-ao-2`"
            )
        )
    block = block.to(_multi_tensor_config.accelerator_name)
    _optimization_tracker.optimized_layers += 1
    logging.warning(
        "Apply auto-round optimization on layer %d / %d.",
        _optimization_tracker.optimized_layers,
        _optimization_tracker.num_layers,
    )

    # Start the training process to update the v, alpha and betta.
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,  # Both `True` and `False` are OK, but use `asym` by default for using the `tinygemm` by default
        bits=config.bits,
        iters=config.iters,
        group_size=config.group_size,
        use_quant_input=False,  # disable it for now
        amp=True,
        model_dtype=next(block.parameters()).dtype,
    )

    @torch.no_grad()
    def _unflatten_grouped_args(grouped_args, spec):
        inputs = []
        for inp in grouped_args:
            cur_args, cur_kwargs = tree_unflatten(inp, spec)
            inputs.append((cur_args, cur_kwargs))
        return inputs

    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    with torch.enable_grad():
        rounder.quant_block_v2_(
            block,
            inputs=block_inputs,
            outputs=block_outputs,
            device=_multi_tensor_config.accelerator_name,
        )


@ar_utils.dump_elapsed_time()
@torch.no_grad()
def apply_auto_round_optimization(
    module: torch.nn.Module,
    args: Tuple[MultiTensor],
    kwargs: Dict[str, MultiTensor],
    output: Tuple[MultiTensor],
    config: _AutoRoundConfig,
):
    # Remove the hook to avoid recursive calls
    module._forward_hook_handle_for_auto_round.remove()
    flat_args, spec = tree_flatten((args, kwargs))
    grouped_args = MultiTensor.flat_to_grouped(flat_args)
    output_flat_args, output_spec = tree_flatten((output, {}))
    output_grouped_args = MultiTensor.flat_to_grouped(output_flat_args)
    _apply_auto_round_optimization(
        module, grouped_args, spec, output_grouped_args, config
    )