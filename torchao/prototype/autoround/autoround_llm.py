import argparse

import torch

import torchao
import torchao.prototype.autoround.utils as ar_utils

from torchao.prototype.autoround.core import (
    apply_auto_round,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import MultiTensor
from torchao.quantization import quantize_
import logging
ar_utils.freeze_random(42)


@torch.no_grad()
def quantize_model_with_autoround_(
    model,
    tokenizer,
    is_target_module,
    bits: int = 4,
    group_size: int = 128,
    iters: int = 200,
    seqlen: int = 2048,
    dataset_name: str = "NeelNanda/pile-10k",
    bs: int = 8,
    nsamples: int = 128,
    use_optimized_layer_output: bool = False,
):
    # Step 1. Prepare the model for applying auto-round

    model_device = next(model.parameters()).device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prepare_model_for_applying_auto_round_(
        model,
        is_target_module,
        bits,
        group_size,
        iters,
        use_optimized_layer_output,
        device=device,
    )

    # Step 2. Caliration and optimization
    dataloader = ar_utils.import_dataloader()(
        tokenizer,
        seqlen=seqlen,
        dataset_name=dataset_name,
        bs=bs,
        nsamples=nsamples,
    )
    input_ids_lst = []
    for data in dataloader:
        input_ids_lst.append(data["input_ids"].to(model_device))
    print(
        f"Number of batches: {len(input_ids_lst)}, shape of all batches: {[inp.shape for inp in input_ids_lst]}"
    )

    multi_t_input_ids = MultiTensor(input_ids_lst)

    # The optimization is applied during the forward pass
    out = model(multi_t_input_ids)

    # Step 3. Apply the quantization
    quantize_(model, apply_auto_round(), is_target_module, device=device)

    num_quantized_weight = ar_utils.count_tensor_of_type(
        model, torchao.dtypes.AffineQuantizedTensor
    )
    print(f"Quantized {num_quantized_weight} Linear layers.")

    return model


def main(args):
    # Get the model, tokenizer, and decoder_cls
    model_name_or_path = args.model_name_or_path
    model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
        model_name_or_path, torch_dtype=torch.bfloat16
    )
    # Disable the `use_cache` for calibration stage.
    model.config.use_cache = False
    ar_utils.gen_text(model, tokenizer, "Float model", max_length=50)

    model = model.to(args.model_device)

    # User need to prepare a `is_target_module` function for identifying the target modules that need to be quantized.
    if args.quant_lm_head:
        is_target_module = (
            lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
        )
    else:
        is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)

    quantize_model_with_autoround_(
        model=model,
        tokenizer=tokenizer,
        is_target_module=is_target_module,
        bits=args.bits,
        iters=args.iters,
        seqlen=args.seqlen,
        dataset_name=args.dataset_name,
        bs=args.train_bs,
        nsamples=args.nsamples,
        use_optimized_layer_output=args.use_optimized_layer_output,
    )
    # Revert the `use_cache` for generation stage.
    model.config.use_cache = True

    # Generate text using the quantized model
    ar_utils.gen_text(model, tokenizer, "Quantized model", max_length=50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        default="facebook/opt-125m",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="NeelNanda/pile-10k",
        help="Dataset name for calibration",
    )
    parser.add_argument(
        "--iters",
        default=200,
        type=int,
        help="Number of iterations for auto-round optimization",
    )
    parser.add_argument(
        "--bits", default=4, type=int, help="Number of bits for quantization"
    )
    parser.add_argument(
        "--train_bs", default=8, type=int, help="Batch size for auto-round optimization"
    )
    parser.add_argument(
        "--nsamples",
        default=128,
        type=int,
        help="Number of samples for calibration process",
    )
    parser.add_argument(
        "--seqlen",
        default=2048,
        type=int,
        help="Sequence length for calibration process",
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Quantize the `lm_head` or not",
    )
    parser.add_argument(
        "--use_optimized_layer_output",
        default=False,
        action="store_true",
        help="Use the optimized layer output for next layer or not",
    )
    parser.add_argument(
        "-d",
        "--model_device",
        default="cuda",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for loading the float model",
    )
    args = parser.parse_args()
    main(args)
