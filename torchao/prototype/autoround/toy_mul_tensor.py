import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten
import gc
class MultiTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, input, **kwargs):
        if isinstance(input, (list, tuple)):
            input = input[0]
        kwargs["dtype"]=kwargs.get("dtype", input.dtype)
        shape = kwargs.pop("shape", input.shape)
        return torch.Tensor._make_wrapper_subclass(cls, shape, **kwargs)

    def __init__(self, input, **kwargs):
        self.values = []
        self.count = 0
        self.add_tensors(input)
        self.debug = True

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.values})"
        )

    def add_tensors(self, input):
        if isinstance(input, (tuple, list)):
            for inp in input:
                self.add_tensors(inp)
        else:
            assert isinstance(input, torch.Tensor), f"MultiTensor can only use add_tensors for Tensors or lists of tensors but got {type(input)}"
            self.count += 1
            self.values.append(input)
        return self

    def pad_to_length(self, length):
        if self.count > length:
            return self
        self.add_tensors([self.values[-1]]*(length-self.count))
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
                        x.pad_to_length(multi_tensor_size).values if isinstance(x, MultiTensor) else [x] * multi_tensor_size for x in flat]
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
                cls(tup).cpu() if isinstance(tup[0], torch.Tensor) else tup[0] for tup in flat_tups
            ]
            # need to check that getting rid of all but one from each nonTensor tuple is OK
            non_tensors_equal=min([True]+[
                min([True]+[ # handle situation where tuples have size 0
                    tup[0]==x for x in tup # check all elements match
                ]) for tup in flat_tups if not isinstance(tup[0], torch.Tensor) # look at tuples of nonTensors
            ])
            return flattened, non_tensors_equal
        kwargs = {} if kwargs is None else kwargs
        # combine args and kwargs and remove lists and tuples
        flat_args, spec = tree_flatten((args, kwargs))
        # convert [A, MultiTensor(b1,b2,b3), MultiTensor(c1,c2,c3)] => [[A,b1,c1], [A,b2,c2] [A,b3,c3]]
        grouped_args = flat_to_grouped(flat_args)
        # run function for each of the multitensors and return a multitensor
        outputs = []
        with torch._C.DisableTorchFunctionSubclass():
            for inp in grouped_args:
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
                    +"caused an error in GPTQMultiInput, the function dispatch only works for functions"
                    +" with Tensor outputs or that have the same non-Tensor output value for all across all inputs"
                )
                return tree_unflatten(flat_outputs, out_spec)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs={}, skip_gptq=False):
        pass

    def __tensor_flatten__(self):
        return ["values"], None

    @classmethod
    def __tensor_unflatten__(
        cls, tensor_data_dict, tensor_attributes, outer_size, outer_stride
    ):
        cls(tensor_data_dict["values"])


class mod(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(10,10)
        self.register_buffer("other", torch.randn(10, 20)*0)

    def forward(self, x, indices, idx = 1):
        print(f"idx: {idx}")
        print(f"initial model input types x: {type(x)}, indices: {type(indices)} (fine)")
        y = self.lin(x)
        print(f"result after linear y: {type(y)}, should be a multitensor (and is)")
        z = self.other
        # print([x.sum() for x in z.values])
        z[:, indices]=y
        print(f"after assigning y to index of z: {type(z)}, should be a multitensor (now working!)")
        # print([x.sum() for x in z.values])
        return z


model = mod()
multi = [
    MultiTensor([torch.randn(10,10), torch.randn(10,10)]),
    MultiTensor([torch.tensor([0,1,2,3,4,5,6,7,8,9]), torch.tensor([0,1,2,3,4,5,6,7,8,9])]),
    2
]

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
        new_child = _replace_with_custom_fn_if_matches_filter(
            child, replacement_fn, filter_fn, f"{cur_fqn}{name}."
        )
        if new_child is not child:
            setattr(model, name, new_child)
    return model

with torch.no_grad():
    _replace_with_custom_fn_if_matches_filter(model, replace_buffers_and_params, lambda x, y: True)
    print(type(model.other), type(model.lin.weight))
    out=model(*multi)