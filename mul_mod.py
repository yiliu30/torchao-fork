"""
# Another issue of autoawq's implementation is that it can not quantize the lm-head, as we donot know how to pass the output of quantized last decoder layer into lm-head
# But below approach can solve this issue.
# If we want to quantize the lm-head
# 1. We need to know that, current linear is the lm-head.  >>> How? swap lm-head with a custom linear? but we need to define the custom linear at ops lib.
# 2. Pass the linear to the optimizer function, which should be able to quantize the linear and mod that includes sub modules.
# 3. Return the output and continue the next step.


"""

import gc

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

# ==------------------------------------------------------------------------------------------==
# Utils
# ==------------------------------------------------------------------------------------------==


def show_params_range(model):
    for name, param in model.named_parameters():
        print(f"{name}: {param.min()}, {param.max()}, shape: {param.shape}")


def replace_buffers_and_params(model):
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, nn.Parameter(MultiTensor([param]), False))
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
            return flattened, non_tensors_equal

        kwargs = {} if kwargs is None else kwargs
        # if "opt_decoder" in func.__name__ or "__getitem__" in func.__name__:
        #     # print(f"In  cur_args type ({type(cur_args)}) {[cur.shape for cur in cur_args if isinstance(cur, torch.Tensor)]}")
        #     breakpoint()
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = flat_to_grouped(flat_args)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            if "opt_decoder" in func.__name__:
                outputs = optimized_decoder(func, grouped_args, spec)
                breakpoint()
            else:
                for i, inp in enumerate(grouped_args):
                    # inp = tensors_to_cuda(inp)
                    cur_args, cur_kwargs = tree_unflatten(inp, spec)
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


def apply_auto_round(observed_block, block_inputs, block_outputs):
    # Call the auto-round to execute the optimization process
    import auto_round

    device = next(observed_block.parameters()).device

    block = observed_block

    # Start the training process to update the v, alpha and betta.
    # TODO: refactor the `quant_block_` to a static method
    rounder = auto_round.AutoRound(
        model=block,
        tokenizer=None,
        sym=False,  # Both True and False are OK
        bits=4,
        iters=10,
        use_quant_input=False,  # disable it for now
        amp=False,
        low_gpu_mem_usage=False,
        model_dtype=next(block.parameters()).dtype,
    )
    rounder.quant_block_(block, block_inputs, block_outputs, device=device)


import transformers
import transformers.models.opt as tran_opt
from torch.library import impl, Library

config = tran_opt.modeling_opt.OPTConfig()
opt_decoder = tran_opt.modeling_opt.OPTDecoderLayer
opt_model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

# The agrs of opt_decoder.forward
# hidden_states: torch.Tensor,
# attention_mask: Optional[torch.Tensor] = None,
# layer_head_mask: Optional[torch.Tensor] = None,
# past_key_value: Optional[Tuple[torch.Tensor]] = None,
# output_attentions: Optional[bool] = False,
# use_cache: Optional[bool] = False,

# ! Here is the main challenge part of this approach
t_lib = Library("transformers_ops", "DEF")  # noqa: TOR901
t_lib.define(
    "opt_decoder(Tensor hidden_state, Tensor? attention_mask=None, Tensor? layer_head_mask=None, Tensor[]? past_key_value=None, bool? output_attentions=False, bool? use_cache=False, int idx=1) -> (Tensor, Tensor[])"
)


@impl(t_lib, "opt_decoder", "CPU")
def opt_decoder_impl(
    hidden_states,
    attention_mask=None,
    layer_head_mask=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    idx=1,
):
    decoder_layer = opt_model.model.decoder.layers[idx]
    return decoder_layer.forward(
        hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )


idx = -1
mod_idx_mapping = {}


class OptDecoderLayerWrapper(torch.nn.Module):
    def __init__(self, orig_mod, idx=1):
        super().__init__()
        self.idx = idx
        mod_idx_mapping[idx] = orig_mod

    def forward(self, *args, **kwargs):
        kwargs.update({"idx": self.idx})
        return torch.ops.transformers_ops.opt_decoder(*args, **kwargs)


show_params_range(opt_model)


def replace_opt(mod):
    global idx
    idx += 1
    return OptDecoderLayerWrapper(mod, idx)


def is_decoder(mod, fqn):
    return isinstance(mod, opt_decoder)


def _get_decoder_layers_by_index(idx):
    global mod_idx_mapping
    return mod_idx_mapping[idx]


def is_mod_decoder(mod, fqn):
    return isinstance(mod, OptDecoderLayerWrapper)


def revert_replace_opt(mod):
    return mod_idx_mapping[mod.idx]


@torch.no_grad()
def infer_mod(mod, grouped_args, spec):
    outputs = []
    for i, inp in enumerate(grouped_args):
        cur_args, cur_kwargs = tree_unflatten(inp, spec)
        cur_kwargs.pop("idx")
        out = mod(*cur_args, **cur_kwargs)
        outputs.append(out.cpu() if isinstance(out, torch.Tensor) else out)
    return outputs


def _flatten_grouped_args(grouped_args, spec):
    inputs = []
    for i, inp in enumerate(grouped_args):
        cur_args, cur_kwargs = tree_unflatten(inp, spec)
        cur_kwargs.pop("idx")
        inputs.append((cur_args, cur_kwargs))
    return inputs


def optimize_mod_(mod, grouped_args, spec, origin_output):
    inputs = _flatten_grouped_args(grouped_args, spec)
    hidden_states_lst = [hidden_states for (hidden_states, kv_cache) in origin_output]
    apply_auto_round(mod, inputs, hidden_states_lst)


def optimized_decoder(func, grouped_args, spec):
    first_grouped_args = grouped_args[0]
    first_cur_args, first_cur_kwargs = tree_unflatten(first_grouped_args, spec)
    decoder_layer_idx = first_cur_kwargs["idx"]
    decoder_layer = _get_decoder_layers_by_index(decoder_layer_idx)
    origin_output = infer_mod(decoder_layer, grouped_args, spec)
    optimize_mod_(decoder_layer, grouped_args, spec, origin_output)
    return origin_output


_replace_with_custom_fn_if_matches_filter(opt_model, replace_opt, is_decoder)
print(opt_model)

multi_input_ids = MultiTensor(
    [torch.tensor([[0, 1, 2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]])]
)


print(f"==============================================================")

_replace_with_custom_fn_if_matches_filter(
    opt_model, replace_buffers_and_params, lambda x, y: True
)
for i in range(1):
    print(f"=========== iteration {i}")
    out = opt_model(multi_input_ids)
    print(out.logits.shape)


_replace_with_custom_fn_if_matches_filter(opt_model, revert_replace_opt, is_mod_decoder)
show_params_range(opt_model)
