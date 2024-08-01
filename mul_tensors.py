import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
import gc


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
            multi_tensor_size = max([x.count if isinstance(x, MultiTensor) else 1 for x in flat])
            # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
            grouped = list(
                zip(
                    *[
                        x.pad_to_length(multi_tensor_size).values
                        if isinstance(x, MultiTensor)
                        else [x] * multi_tensor_size
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
            flattened = [cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0] for tup in flat_tups]
            # need to check that getting rid of all but one from each nonTensor tuple is OK
            non_tensors_equal = min(
                [True]
                + [
                    min(
                        [True]
                        + [  # handle situation where tuples have size 0
                            tup[0] == x
                            for x in tup  # check all elements match
                        ]
                    )
                    for tup in flat_tups
                    if not isinstance(tup[0], torch.Tensor)  # look at tuples of nonTensors
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
            for i, inp in enumerate(grouped_args):
                # inp = tensors_to_cuda(inp)
                print(f"under __torch_function__ func: {func.__name__}, start to handle {i}-th input: ")
                cur_args, cur_kwargs = tree_unflatten(inp, spec)

                if "opt_decoder" in func.__name__ or "__getitem__" in func.__name__:
                    print(
                        f"In  cur_args type ({type(cur_args)}) {[cur.shape for cur in cur_args if isinstance(cur, torch.Tensor)]}"
                    )
                    breakpoint()
                # print(f"cur_args: {cur_args}, cur_kwargs: {cur_kwargs}")
                out = func(*cur_args, **cur_kwargs)
                if "opt_decoder" in func.__name__ or "__getitem__" in func.__name__:
                    if isinstance(out, torch.Tensor):
                        print(f"== Out shape: {out.shape}")
                    else:
                        print(f"Out list {[o.shape for o in out if isinstance(o, torch.Tensor)]}")
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
        print(f"under __torch_function__ func: {func.__name__}, start to handle {i}-th input: ")
        pass

    def __tensor_flatten__(self):
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride):
        cls(tensor_data_dict["values"])


class mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(5, 5)
        self.register_buffer("other", torch.randn(5, 20) * 0)

    def forward(self, x, indices):
        print(f"== initial model input types x: {type(x)}, indices: {type(indices)} (fine)")
        y = self.lin(x)
        print(f"== result after linear y: {type(y)}, should be a multitensor (and is)")
        z = self.other
        # print([x.sum() for x in z.values])
        z[:, indices] = y
        print(f"== after assigning y to index of z: {type(z)}, should be a multitensor (now working!)")
        # print([x.sum() for x in z.values])
        return z


model = mod()
multi = [
    MultiTensor([torch.randn(5, 5), torch.randn(5, 5)]),
    MultiTensor([torch.tensor([0, 1, 2, 3, 4]), torch.tensor([0, 1, 2, 3, 4])]),
]
print(multi[0])
print(multi[1])

print(f"weight {model.lin.weight}")
print(f"other {model.other}")


def replace_buffers_and_params(model):
    for name, buf in model.named_buffers(recurse=False):
        setattr(model, name, MultiTensor([buf]))
    for name, param in model.named_parameters(recurse=False):
        setattr(model, name, nn.Parameter(MultiTensor([param]), False))
    return model


def _replace_with_custom_fn_if_matches_filter(
    model,
    replacement_fn,
    filter_fn,
    cur_fqn="",
) -> None:
    if filter_fn(model, cur_fqn[:-1]):
        model = replacement_fn(model)
    for name, child in model.named_children():
        new_child = _replace_with_custom_fn_if_matches_filter(child, replacement_fn, filter_fn, f"{cur_fqn}{name}.")
        if new_child is not child:
            setattr(model, name, new_child)
    return model


with torch.no_grad():
    _replace_with_custom_fn_if_matches_filter(model, replace_buffers_and_params, lambda x, y: True)
    print(type(model.other), type(model.lin.weight))
    for i in range(1):
        print(f"=========== iteration {i}")
        out = model(*multi)


from torch.library import Library, impl

import transformers
import transformers.models.opt as tran_opt

config = tran_opt.modeling_opt.OPTConfig()
opt_decoder = tran_opt.modeling_opt.OPTDecoderLayer
opt_model = transformers.AutoModelForCausalLM.from_pretrained("facebook/opt-125m")
# hidden_states: torch.Tensor,
# attention_mask: Optional[torch.Tensor] = None,
# layer_head_mask: Optional[torch.Tensor] = None,
# past_key_value: Optional[Tuple[torch.Tensor]] = None,
# output_attentions: Optional[bool] = False,
# use_cache: Optional[bool] = False,

t_lib = Library("transformers_ops", "DEF")  # noqa: TOR901
t_lib.define(
    "opt_decoder(Tensor hidden_state, Tensor? attention_mask=None, Tensor? layer_head_mask=None, Tensor[]? past_key_value=None, bool? output_attentions=False, bool? use_cache=False, int idx=1) -> Tensor[]"
)
# opt_decoder1 = opt_model.model.decoder.layers[0]


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
    print(f"use idx {idx},")  # {decoder_layer}")
    # breakpoint()
    kk = decoder_layer.forward(
        hidden_states,
        attention_mask=attention_mask,
        layer_head_mask=layer_head_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    breakpoint()
    return kk


t_lib.define(
    "opt_decoder_simple(Tensor hidden_states, Tensor? attention_mask=None, int idx=1) -> (Tensor, Tensor[])"
)
# opt_decoder1 = opt_model.model.decoder.layers[0]


@impl(t_lib, "opt_decoder_simple", "CPU")
def opt_decoder_simple_impl(
    hidden_states,
    attention_mask=None,
    idx=1,
):
    decoder_layer = opt_model.model.decoder.layers[idx]
    print(f"use idx {idx},")  # {decoder_layer}")
    # breakpoint()
    kk = decoder_layer.forward(
        hidden_states,
        attention_mask=attention_mask,
        use_cache=True,
    )
    breakpoint()
    return kk


class OptDecoderLayerWrapper(torch.nn.Module):
    def __init__(self, idx=1):
        super().__init__()
        self.idx = idx

    def forward(self, *args, **kwargs):
        hidden_states = args[0]
        # attention_mask = kwargs["attention_mask"]
        # layer_head_mask = kwargs.get("layer_head_mask", None)
        # past_key_value = kwargs.get("past_key_value", None)
        
        # kk =  torch.ops.transformers_ops.opt_decoder(hidden_state, attention_mask, layer_head_mask, past_key_value, idx=self.idx)
        # return torch.ops.transformers_ops.opt_decoder(*args, **kwargs)
        update_kwargs = {k: v for k, v in kwargs.items() if k in ["attention_mask"]}
        update_kwargs.update({"idx": self.idx})
        breakpoint()
        return torch.ops.transformers_ops.opt_decoder_simple(hidden_states, **update_kwargs)


idx = -1
import copy

opt_model2 = copy.deepcopy(opt_model)


def replace_opt(mod):
    global idx
    idx += 1
    print(f"replace opt {idx}")
    return OptDecoderLayerWrapper(idx)


def is_decoder(mod, fqn):
    return isinstance(mod, opt_decoder)


_replace_with_custom_fn_if_matches_filter(opt_model2, replace_opt, is_decoder)
print(opt_model2)

multi_input_ids = MultiTensor([torch.tensor([[0, 1, 2, 3, 4]]), torch.tensor([[0, 1, 2, 3, 4]])])


print(f"==============================================================")
with torch.no_grad():
    _replace_with_custom_fn_if_matches_filter(opt_model2, replace_buffers_and_params, lambda x, y: True)
    for i in range(1):
        print(f"=========== iteration {i}")
        out = opt_model2(multi_input_ids)
        print(out.logits.shape)
