from torchao.quantization.smoothquant import (
    swap_linear_with_smooth_fq_linear,
    smooth_fq_linear_to_inference,
)
import copy
from typing import Dict
import torch

from torchao.quantization.utils import (
    _apply_logging_hook,
    compute_error,
    compute_error as SQNR,
    _fqn_to_op_to_shape_to_count,
    LoggingTensorMode,
)


# https://github.com/pytorch-labs/ao/pull/17#issuecomment-1906957451

"""
# 1. initialize quantizer
quantizer = MyQuantizer(...configs)

# [optional] 2. quantizer can implement a capture method to capture the program
model = quantizer.capture(model, example_inputs)

# [optional] 3. probably required for QAT, but could be optional for others
model = quantizer.prepare(model)

# 4. save/load the prepared model state (with state_dict())
# at this point, users can save/load the model, for prepare step, we want to only support state_dict
# since the prepared model itself may not be serializable
torch.save(model.state_dict(), “file_name.pt”)
# for load we’ll need to prepare the model again
quantizer = MyQuantizer(...configs)
model = quantizer.capture(model, example_inputs)
model = quantizer.prepare(model)
model.load_state_dict(torch.load(“file_name.pt”))

# [optional] 5. calibration/training

# 6. convert the model to a standard q/dq representation of the quantized model
# model = quantizer.convert(model)

# 7. save/load for q/dq representation of quantized model
#  (today eager mode uses model.state_dict(), export based flow is using ExportedProgram, these could be implemented in save/load methods)
# Save/Load 1. Save/load entire model
quantizer.save(model, ”quantized_model.pt”)
model = quantizer.load(”quantized_model.pt”)

# Save/Load 2. Save/load through state dict
Same as the step after prepare (step 4)

# 8. Lowering
# lower to model to some backend, e.g. torch.compile, executorch
# model = torch.compile(model)

# modeling users will be using the type of quantizer based on use case, and there is no
# gurantee of comopsability between different quantizers.

# But in practice generally, we should have some composability within different flows bulit with graph transformations (export based flows)
# and within different flows built with eager mode module modifications (module swap, subclasses) when they are
# modifying different parts of the model.


"""


class Quantizer:
    def __init__(self):
        super().__init__()
        ...

    # Note: each Quantizer will have their own implementation for prepare and convert
    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        # implementation 1

        # model = prepare_pt2e(model, self)

        # implementation 2, module swap, modifying weights with tensor subclass etc.
        # model = ...
        ...
        return model

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        # implementation 1
        # model = convert_pt2e(model, self)
        ...
        # implementation 2
        # model = ...

        return model

    def save(self, model: torch.nn.Module, *args, **kwargs) -> None:
        pass

    def load(self, *args, **kwargs) -> torch.nn.Module:
        pass


# from torchao.
class SmoothQuantizer(Quantizer):
    """
    # Example
    def get_float_model():
        from transformers import DistilBertModel

        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        return model


    def get_example_inputs() -> Dict[str, torch.Tensor]:
        from transformers import DistilBertTokenizer

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        return encoded_input

    sq_quantizer = SmoothQuantizer(alpha=0.5)
    prepared_model = sq_quantizer.prepare(model=get_float_model())
    sq_quantizer.calibrate(model=prepared_model, input=get_example_inputs())
    converted_model = sq_quantizer.convert(prepared_model)
    sq_quantizer.save(converted_model)


    """

    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha

    def prepare(self, model: torch.nn.Module) -> torch.nn.Module:
        # implementation 1
        # model = prepare_pt2e(model, self)

        # implementation 2, module swap, modifying weights with tensor subclass etc.
        # model = ...
        swap_linear_with_smooth_fq_linear(model=model, alpha=self.alpha)
        return model

    def calibrate(self, model: torch.nn.Module, input: Dict[str, torch.Tensor]) -> None:
        # record the scale in-place
        model(**input)

    def convert(self, model: torch.nn.Module) -> torch.nn.Module:
        # update the module inplace
        smooth_fq_linear_to_inference(model=model)
        return model

    def save(self, model: torch.nn.Module, *args, **kwargs) -> None:
        # TODO
        pass


def test_sq_quantizer():
    def get_float_model():
        from transformers import DistilBertModel

        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        return model


    def get_example_inputs() -> Dict[str, torch.Tensor]:
        from transformers import DistilBertTokenizer

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")
        return encoded_input

    sq_quantizer = SmoothQuantizer(alpha=0.5)
    prepared_model = sq_quantizer.prepare(model=get_float_model())
    sq_quantizer.calibrate(model=prepared_model, input=get_example_inputs())
    converted_model = sq_quantizer.convert(prepared_model)
    sq_quantizer.save(converted_model)
test_sq_quantizer()


class Test:
    def assertTrue(self, con):
        assert con

    def test_on_dummy_distilbert(self):
        # https://huggingface.co/distilbert-base-uncased#how-to-use
        from transformers import (  # type: ignore[import-untyped]
            DistilBertModel,
            DistilBertTokenizer,
        )

        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
        # print(model)
        text = "Replace me by any text you'd like."
        encoded_input = tokenizer(text, return_tensors="pt")

        output_ref = model(**encoded_input)
        # print(output_ref)

        #
        # smooth_quant
        #
        model_copy = copy.deepcopy(model)
        swap_linear_with_smooth_fq_linear(model_copy, alpha=0.75)
        import pdb

        pdb.set_trace()
        # calibrate
        output_1_1 = model_copy(**encoded_input)
        # inference
        smooth_fq_linear_to_inference(model_copy)
        output_1_2 = model_copy(**encoded_input)
        # print(output_1_1)
        # print(output_1_2)
        sqnr_sq = compute_error(
            output_ref.last_hidden_state, output_1_2.last_hidden_state
        )
        print("sqnr_sq", sqnr_sq)
        self.assertTrue(sqnr_sq >= 20.0)

        #
        # reference - dynamic linear quant
        #
        model_copy2 = copy.deepcopy(model)
        qconfig = torch.ao.quantization.QConfig(
            activation=None,
            weight=torch.ao.quantization.default_per_channel_weight_observer,
        )
        model_copy2 = torch.ao.quantization.quantize_dynamic(
            model_copy2,
            {torch.nn.Linear: qconfig},
        )
        output_2_2 = model_copy2(**encoded_input)
        # print(output_2_2)
        sqnr_pt_quant = compute_error(
            output_ref.last_hidden_state, output_2_2.last_hidden_state
        )
        print("sqnr_pt_quant", sqnr_pt_quant)
        self.assertTrue(sqnr_sq >= 8.0)


# t = Test()
# t.test_on_dummy_distilbert()
