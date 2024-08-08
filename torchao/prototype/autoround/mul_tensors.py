"""
# Modified from HDCharles's  https://github.com/pytorch/ao/issues/577

Goal: Pause module(decoder block) forward and do the optimization 

Background:
Auto-round requires us to, for each decoder,

1. Track all activations going to that decoder
2. Apply the Auto-round algorithm to update that linear’s weight
3. Use the updated weight to generate outputs for that linear
4. Repeat for the next decoder

Some details:
- Issue, as we accumulate all the activations, just like we use a very big batch size, it's easy to OoM.
    - Solution1, load the weight into GPU before run, and offload it after run, a lot of load/offload overheads?
    - 


"""

import logging

logging.basicConfig(level=logging.INFO)
import sys

auto_round_path = (
    "/home/yliu7/workspace/inc/3rd-party/torchao/third_party/auto-round-for-ao"
)
sys.path.insert(0, auto_round_path)
import gc
from typing import Dict, List, Tuple

import torch

torch.cuda.memory._record_memory_history()

from dataclasses import dataclass

import torch.nn as nn
from simple_parsing import ArgumentParser
from torch.utils._pytree import tree_flatten, tree_unflatten

import torchao.quantization as ao_quant
from torchao.prototype.autoround import utils as ar_utils

logger = logging.getLogger(__name__)
import auto_round.calib_dataset


@dataclass
class AutoRoundConfig:
    bits: int = 4
    sym: bool = False
    iters: int = 200
    group_size: int = 128
    train_bs: int = 8
    eval_bs: int = 4
    seed: int = 42
    amp: bool = False
    nsamples: int = 128
    seqlen: int = 2048


parser = ArgumentParser()
parser.add_argument(
    "-m", "--model_name_or_path", type=str, default="facebook/opt-125m", help="model"
)
parser.add_arguments(AutoRoundConfig, dest="ar_config")
args = parser.parse_args()
auto_round_config = args.ar_config
logger.info(auto_round_config)
ar_utils.freeze_random(auto_round_config.seed)


# ==------------------------------------------------------------------------------------------==
# MultiTensor, copied from https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227
# ==------------------------------------------------------------------------------------------==


def inspect_model(model):
    for name, param in model.named_parameters():
        print(f"{name}: shape: {param.shape}, type:{type(param)}")


def replace_buffers_and_params(model):
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, nn.Parameter(MultiTensor([param]), False))
    return model


def revert_replace_buffers_and_params(model):
    for name, buf in model.named_buffers(recurse=False):
        if isinstance(buf, MultiTensor):
            setattr(model, name, buf.values[0])
    for name, param in model.named_parameters(recurse=False):
        if isinstance(param, MultiTensor):
            setattr(model, name, nn.Parameter(param.values[0], False))
    return model


def _replace_with_custom_fn_if_matches_filter(
    model: torch.nn.Module,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
    for name, child in model.named_children():
        new_child = _replace_with_custom_fn_if_matches_filter(
            child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
        )
        if new_child is not child:
            setattr(model, name, new_child)
    return model


class MultiTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, input, **kwargs):
        if isinstance(input, (list, tuple)):
            input = input[0]
        kwargs["dtype"] = kwargs.get("dtype", input.dtype)
        shape = kwargs.pop("shape", input.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, input, **kwargs):
        self.values = []
        self.count = 0
        self.add_tensors(input)
        self.debug = True

    def __repr__(self):
        return f"{self.__class__.__name__}(data={self.values})"

    def add_tensors(self, input):
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(
                input, torch.Tensor
            ), f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            self.count += 1
            self.values.append(input)
        return self

    def pad_to_length(self, length):
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]] * (length - self.count))
        return self

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None, skip_gptq=False):
        def flat_to_grouped(flat):
            # size of biggest MultiTensor
            multi_tensor_size = max(
                [x.count if isinstance(x, MultiTensor) else 1 for x in flat]
            )
            # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
            grouped = list(
                zip(
                    *[
                        (
                            x.pad_to_length(multi_tensor_size).values
                            if isinstance(x, MultiTensor)
                            else [x] * multi_tensor_size
                        )
                        for x in flat
                    ]
                )
            )
            return grouped

        # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
        # where A is nontensor, b's,c's are tensors
        def grouped_to_flat(grouped):
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [(A,A,A), (b1,b2,b3), (c1,c2,c3)]
            flat_tups = list(zip(*grouped))
            # convert [(A,A,A), (b1,b2,b3), (c1,c2,c3)] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flattened = [
                cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0]
                for tup in flat_tups
            ]
            # need to check that getting rid of all but one from each nonTensor tuple is OK
            non_tensors_equal = min(
                [True]
                + [
                    min(
                        [True]
                        + [  # handle situation where tuples have size 0
                            tup[0] == x for x in tup  # check all elements match
                        ]
                    )
                    for tup in flat_tups
                    if not isinstance(
                        tup[0], torch.Tensor
                    )  # look at tuples of nonTensors
                ]
            )
            if not non_tensors_equal:
                breakpoint()
            return flattened, non_tensors_equal

        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = flat_to_grouped(flat_args)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            # Note: for the decoder, we need to optimize it
            if func is torch.ops.transformers_ops.general_decoder:
                outputs = optimize_decoder(func, grouped_args, spec)
            else:
                for i, inp in enumerate(grouped_args):
                    cur_args, cur_kwargs = tree_unflatten(inp, spec)
                    cur_args = ar_utils.move_input_to_device(cur_args, "cuda")
                    cur_kwargs = ar_utils.move_input_to_device(cur_kwargs, "cuda")
                    out = func(*cur_args, **cur_kwargs)
                    outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)
            grouped_outputs = [tree_flatten(x)[0] for x in outputs]
            out_spec = tree_flatten(outputs[0])[1]
            # convert [[A,b1,c1], [A,b2,c2] [A,b3,c3]] => [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)]
            flat_outputs, non_tensors_equal = grouped_to_flat(grouped_outputs)
            assert non_tensors_equal, (
                f"ERR: found a function in model: {func} which "
                + "caused an error in GPTQMultiInput, the function dispatch only works for functions"
                + " with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
            )
            return tree_unflatten(flat_outputs, out_spec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}):
        print(
            f"under __torch_function__ func: {func.__name__}, start to handle {i}-th input: "
        )
        pass

    def __tensor_flatten__(self):
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        cls(tensor_data_dict["values"])


# ==------------------------------------------------------------------------------------------==
# Pause module forward and do the optimization
# ==------------------------------------------------------------------------------------------==


# Some helper functions, not important, go to Main logic section directly
def replace_decoder_block(mod):
    global idx
    idx += 1
    return DecoderLayerWrapper(mod, idx)


def is_decoder(mod, fqn):
    return isinstance(mod, decoder_cls)


def _get_decoder_layers_by_index(idx):
    global mods_mapping
    return mods_mapping[idx].to("cuda")


def is_wrapped_decoder(mod, fqn):
    return isinstance(mod, DecoderLayerWrapper)


def revert_decoder_block_replacement(mod):
    return mods_mapping[mod.idx]


@torch.no_grad()
def _infer_mod(mod, cur_args, cur_kwargs):
    cur_args = ar_utils.move_input_to_device(cur_args, "cuda")
    cur_kwargs = ar_utils.move_input_to_device(cur_kwargs, "cuda")
    out = mod(*cur_args, **cur_kwargs)
    return out.cpu() if isinstance(out, torch.Tensor) else out


@torch.no_grad()
def infer_mod(mod, grouped_args, spec):
    outputs = []
    for i, inp in enumerate(grouped_args):
        cur_args, cur_kwargs = tree_unflatten(inp, spec)
        cur_kwargs.pop("idx")
        out = _infer_mod(mod, cur_args, cur_kwargs)
        outputs.append(out)
    return outputs


def _unflatten_grouped_args(grouped_args, spec):
    inputs = []
    for i, inp in enumerate(grouped_args):
        cur_args, cur_kwargs = tree_unflatten(inp, spec)
        cur_kwargs.pop("idx")
        inputs.append((cur_args, cur_kwargs))
    return inputs


def apply_auto_round(observed_block, block_inputs, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round

    breakpoint()
    device = next(observed_block.parameters()).device
    device = "cuda"
    block = observed_block

    # Start the training process to update the v, alpha and betta.
    # TODO: refactor the `quant_block_` to a static method

    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )
    rounder.quant_block_(block, block_inputs, block_outputs, device=device)


def apply_auto_round_orig(observed_block, grouped_args, spec, block_outputs):
    # Here we got the decoder block, block inputs, and block outputs, we can start the optimization process
    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    block_outputs = [hidden_states for (hidden_states, kv_cache) in block_outputs]
    # Call the auto-round to execute the optimization process
    import auto_round

    device = next(observed_block.parameters()).device
    device = "cuda"
    block = observed_block

    # Start the training process to update the v, alpha and betta.
    # TODO: refactor the `quant_block_` to a static method

    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )

    # input_ids, input_others, q_input=None,
    def pre_process(
        block_inputs: List[Tuple[Tuple[torch.Tensor], Dict[str, torch.Tensor]]]
    ):
        input_ids = []
        input_others = []
        for inputs, others in block_inputs:
            input_ids.append(inputs[0])
            input_others.append(others["position_ids"])
        input_ids_flat = torch.cat(input_ids, dim=0)
        # input_others_flat = torch.cat(input_others, dim=0)
        input_others_flat = input_others[0]
        return input_ids_flat, {"position_ids": input_others_flat}

    breakpoint()
    input_ids, input_others = pre_process(block_inputs)
    outputs = torch.cat(block_outputs, dim=0)
    rounder.quant_block(
        block,
        input_ids=input_ids,
        input_others=input_others,
        device=device,
        output=outputs,
    )


def apply_auto_round_orig2(observed_block, grouped_args, spec, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round

    device = next(observed_block.parameters()).device
    device = "cuda"
    block = observed_block

    # Start the training process to update the v, alpha and betta.
    # TODO: refactor the `quant_block_` to a static method

    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=auto_round_config.sym,  # Both True and False are OK
        bits=auto_round_config.bits,
        iters=auto_round_config.iters,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )

    block_inputs = _unflatten_grouped_args(grouped_args, spec)
    rounder.quant_block2(
        block, inputs=block_inputs, outputs=block_outputs, device="cuda"
    )


def apply_ao_int_woq(observed_block):
    # TODO: int4 woq not needs inputs/outputs, just for demo
    for name, mod in observed_block.named_modules():
        if isinstance(mod, torch.nn.Linear):
            mod.weight = torch.nn.Parameter(
                mod.weight.to(torch.bfloat16), requires_grad=False
            )
            ao_quant.quant_api.int4_weight_only()(mod)
    return observed_block


# ==------------------------------------------------------------------------------------------==
# Main logic
# ==------------------------------------------------------------------------------------------==


# Global variables for mapping the decoder block index to the module
# TODO: make it more robust
idx = -1
mods_mapping: Dict[int, torch.nn.Module] = {}


from torch.library import Library

t_lib = Library("transformers_ops", "DEF")

# It seems that the definition of the custom operation doesn't need to match the actual function signature exactly.
# It merely serves as a flag to inform the dispatcher that this function(decoder block) is intended for optimization.
# All of the args and kwargs will be passed to the optimized function,
# which will handle unpacking them and returning the correct output.
# The call flow:
#   `DecoderLayerWrapper.forward` -> `general_decoder` under `__torch_function__` -> `optimize_decoder` -> return the optimized output
t_lib.define("general_decoder(Tensor hidden_state) -> (Tensor, Tensor[])")


class DecoderLayerWrapper(torch.nn.Module):
    def __init__(self, orig_mod, idx=1):
        super().__init__()
        self.idx = idx
        mods_mapping[idx] = orig_mod

    def forward(self, *args, **kwargs):
        kwargs.update({"idx": self.idx})
        return torch.ops.transformers_ops.general_decoder(*args, **kwargs)


def _optimize_decoder(mod, grouped_args, spec, origin_output):

    # apply_auto_round(mod, inputs, hidden_states_lst)
    # apply_auto_round_orig(mod, grouped_args, spec, origin_output)
    apply_auto_round_orig2(mod, grouped_args, spec, origin_output)
    # apply_ao_int_woq(mod)


def _offload_decoder_to_cpu(idx):
    global mods_mapping
    mods_mapping[idx] = mods_mapping[idx].to("cpu")


def optimize_decoder(func, grouped_args, spec):
    ar_utils.see_memory_usage("Before optimize decoder")
    first_grouped_args = grouped_args[0]
    first_cur_args, first_cur_kwargs = tree_unflatten(first_grouped_args, spec)
    decoder_layer_idx = first_cur_kwargs["idx"]
    ar_utils.logger.info(f"Optimizing decoder layer {decoder_layer_idx}")
    decoder_layer = _get_decoder_layers_by_index(decoder_layer_idx)
    origin_output = infer_mod(decoder_layer, grouped_args, spec)
    _optimize_decoder(decoder_layer, grouped_args, spec, origin_output)
    ar_utils.see_memory_usage("After optimize decoder, before offload decoder")
    _offload_decoder_to_cpu(decoder_layer_idx)
    del first_cur_args, first_cur_kwargs
    ar_utils.see_memory_usage("After optimize decoder")
    return ar_utils.move_input_to_device(origin_output, "cpu")


# ==------------------------------------------------------------------------------------------==
# Main steps to optimize the model
# ==------------------------------------------------------------------------------------------==

import transformers
import torchao.prototype.autoround.utils as ar_utils

model_name_or_path = args.model_name_or_path
opt_model, tokenizer, decoder_cls = ar_utils.get_float_model_info(model_name_or_path)
# Workaround for disabling the kv_cache, which cause the OOM.
opt_model.config.use_cache = False
breakpoint()
# ar_utils.gen_text(opt_model, tokenizer, "Float model", device="cuda", max_length=50)

print(opt_model)

# 1. Replace the decoder block with a wrapper block
_replace_with_custom_fn_if_matches_filter(opt_model, replace_decoder_block, is_decoder)

# 2. Replace the buffers and parameters with MultiTensor
_replace_with_custom_fn_if_matches_filter(
    opt_model, replace_buffers_and_params, lambda x, y: True
)

# 3. Caliration and optimization
multi_input_ids = MultiTensor(
    [torch.tensor([[0, 1, 2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]])]
)

# Use the real dataset
import torch


def get_dataloader_from_ar(tokenizer):
    import auto_round

    dataloader = auto_round.calib_dataset.get_dataloader(
        tokenizer,
        auto_round_config.seqlen,
        seed=auto_round_config.seed,
        bs=auto_round_config.train_bs,
        nsamples=auto_round_config.nsamples,
    )
    return dataloader


input_ids_lst = []
attn_mask_lst = []
for i, data in enumerate(get_dataloader_from_ar(tokenizer)):
    input_ids_lst.append(data["input_ids"])
    attn_mask_lst.append(data["attention_mask"])

opt_model = opt_model
mul_t_input_ids = MultiTensor(input_ids_lst)
mul_t_attn_mask = MultiTensor(attn_mask_lst)
torch.cuda.empty_cache()
ar_utils.see_memory_usage("Before run")
out = opt_model(mul_t_input_ids, mul_t_attn_mask)
print(out.logits.shape)




# 4. Revert the decoder block replacement, the block has been optimized
_replace_with_custom_fn_if_matches_filter(
    opt_model, revert_decoder_block_replacement, is_wrapped_decoder
)

# 5. Revert the buffers and parameters
_replace_with_custom_fn_if_matches_filter(
    opt_model, revert_replace_buffers_and_params, lambda x, y: True
)


ar_utils.gen_text(opt_model, tokenizer, "Quantized model", device="cuda", max_length=50)

