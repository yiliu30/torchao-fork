import torch
import unittest
import torch
import os
from torch._export import capture_pre_autograd_graph
from torch.ao.quantization.quantize_pt2e import (
    prepare_pt2e,
    convert_pt2e,
)
from torch.ao.quantization.quantizer.xnnpack_quantizer import (
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)

from torchao.quantization.quant_api import (
    _replace_with_custom_fn_if_matches_filter,
    apply_dynamic_quant,
    apply_weight_only_int8_quant,
    Quantizer,
    TwoStepQuantizer,
)
from torchao.quantization.utils import (
    TORCH_VERSION_AFTER_2_3,
    TORCH_VERSION_AFTER_2_4,
)
from pathlib import Path
from sentencepiece import SentencePieceProcessor
from model import Transformer, prepare_inputs_for_model
import logging

logger = logging.getLogger(__name__)

from torchao import quantization
import torchao.quantization.quant_primitives as ao_prim

from torchao.quantization import GPTQ
from torchao.quantization.GPTQ import (
    Int8DynActInt4WeightGPTQQuantizer,
    Int4WeightOnlyGPTQQuantizer,
    InputRecorder,
    TransformerEvalWrapper,
)

# should be similar to TorchCompileDynamicQuantizer
precision = torch.bfloat16
device = "cpu"
# checkpoint_path = Path("../gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth")

model_path = "/home/yliu7/workspace/inc/3rd-party/gpt-fast/checkpoints/tinyllamas/stories15M/model.pth"
checkpoint_path = Path(model_path)
# checkpoint_path = Path(
#     "/home/yliu7/workspace/inc/3rd-party/gpt-fast/checkpoints/meta-llama/Llama-2-7b-chat-hf/model.pth"
# )

model = Transformer.from_name(checkpoint_path.parent.name)
checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
if "model" in checkpoint:
    checkpoint = checkpoint["model"]
model.load_state_dict(checkpoint, assign=True)
model = model.to(dtype=precision, device=device)
model.eval()
tokenizer_path = checkpoint_path.parent / "tokenizer.model"
assert tokenizer_path.is_file(), tokenizer_path
tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))  # pyre-ignore[28]
blocksize = 128
percdamp = 0.01
groupsize = 32
calibration_tasks = ["wikitext"]
calibration_limit = 2
calibration_seq_length = 100
input_prep_func = prepare_inputs_for_model
pad_calibration_inputs = False

# Prepare the inputs
inputs = (
    InputRecorder(
        tokenizer,
        calibration_seq_length,
        input_prep_func,
        pad_calibration_inputs,
        model.config.vocab_size,
    )
    .record_inputs(
        calibration_tasks,
        calibration_limit,
    )
    .get_inputs()
)
# Initialize the quantizer
quantizer = Int4WeightOnlyGPTQQuantizer(
    blocksize,
    percdamp,
    groupsize,
    inner_k_tiles=2,
    # precision=precision,
)
model.setup_caches(max_batch_size=1, max_seq_length=calibration_seq_length)
# Quantize model
import pdb

pdb.set_trace()
model = quantizer.quantize(model, inputs)
pdb.set_trace()
# Eval quantized model


eval_wrapper = TransformerEvalWrapper(
    model,
    tokenizer,
    max_seq_length=model.config.block_size,
    input_prep_func=prepare_inputs_for_model,
    device=device,
)

result = eval_wrapper.run_eval(["wikitext"], 1)

assert (
    result["results"]["wikitext"]["word_perplexity,none"] < 7.88
), f"accuracy regressed from 7.87 to {result['results']['wikitext']['word_perplexity,none']}"


from typing import List, Any, Dict, Optional, Union, Tuple


class Quantizer:
    def __init__(
        self,
        blocksize: int,
        percdamp: float,
        groupsize: int,
        inner_k_tiles: int = 8,
        padding_allowed: bool = True,
        precision: torch.dtype = torch.float32,
    ):
        pass

    def _create_quantized_state_dict(
        self,
        model: torch.nn.Module,
        inputs: List[GPTQ.MultiInput],
        blocksize: int,
        percdamp: float,
        groupsize: int,
    ) -> Dict[str, torch.Tensor]:
        state_dict = {}
        return state_dict

    def _convert_for_runtime(self, model: torch.nn.Module) -> torch.nn.Module:
        """Swap the `torch.nn.Linear` with `Int8DynActInt4WeightLinear`."""
        return model

    def quantize(self, model: torch.nn.Module, inputs: List[GPTQ.MultiInput], **kwargs: Any) -> torch.nn.Module:
        """Quantize the model.

        Step1: Quantize the weight and bias into int8.
        Step2: Swap the `torch.nn.Linear` with `Int8DynActInt4WeightLinear`.
        Step3: Load the quantized state_dict into the swapped model.
        """
        state_dict = self._create_quantized_state_dict(
            model,
            inputs,
            self.blocksize,
            self.percdamp,
            self.groupsize,
        )
        model = self._convert_for_runtime(model)
        model.load_state_dict(state_dict, strict=False)
        return model
