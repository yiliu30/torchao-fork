import torch
from typing import Any

############################# Unified Quantization APIs ##############################
# API 1, single quantize call to create a quantized model with quantized state_dict
class Quantizer:
    def quantize(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass


# API 2, flow that needs calibration or training
class TwoStepQuantizer:
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass



class TwoStepQuantizer:
    def prepare(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass

    def convert(
        self, model: torch.nn.Module, *args: Any, **kwargs: Any
    ) -> torch.nn.Module:

        pass
    def load_quantized_model(eager_model_cls, quantized_model_state_dict):
        pass

############################# Unified Quantization APIs ##############################
