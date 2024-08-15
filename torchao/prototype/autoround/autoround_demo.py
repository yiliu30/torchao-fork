import logging

import argparse

import torch

import torchao
import torchao.prototype.autoround.utils as ar_utils

from torchao.prototype.autoround.core import (
    auto_round_config,
    prepare_model_for_applying_auto_round_,
)
from torchao.prototype.autoround.multi_tensor import MultiTensor


def quantize_model_with_autoround(
    model, tokenizer, decoder_cls, auto_round_config=auto_round_config, device="cuda"
):
    with torch.no_grad():
        # 0. Get the model, tokenizer, and decoder_cls
        import torchao.prototype.autoround.utils as ar_utils

        # 1. Prepare the model for applying auto-round
        # User should provide the `is_decoder` function for identifying the decoder block
        # It can be extended to other modules, such as `lm_head`, the function like:
        #   is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
        if auto_round_config.quant_lm_head:
            is_decoder = (
                lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
            )
        else:
            is_decoder = lambda mod, fqn: isinstance(mod, decoder_cls)

        prepare_model_for_applying_auto_round_(model, is_decoder)

        # 2. Caliration and optimization
        dataloader = ar_utils.get_dataloader(
            tokenizer,
            auto_round_config.seqlen,
            seed=auto_round_config.seed,
            bs=auto_round_config.train_bs,
            nsamples=auto_round_config.nsamples,
        )

        input_ids_lst = []
        attn_mask_lst = []
        for i, data in enumerate(dataloader):
            input_ids_lst.append(data["input_ids"].to(device))
            attn_mask_lst.append(data["attention_mask"].to(device))

        multi_t_input_ids = MultiTensor(input_ids_lst)
        multi_t_attn_mask = MultiTensor(attn_mask_lst)

        # The optimization is applied during the forward pass
        out = model(multi_t_input_ids, multi_t_attn_mask)

        assert (
            ar_utils.count_tensor_of_type(model, torchao.dtypes.AffineQuantizedTensor)
            > 0
        ), f"No `AffineQuantizedTensor` found in the model"

        # 4(Optional). Generate text using the optimized model
        ar_utils.gen_text(
            model, tokenizer, "Quantized model", device="cuda", max_length=50
        )
        return model

@ar_utils.dump_elapsed_time()
def _compile_model(model):
    model = torch.compile(model=model, mode="max-autotune")
    return model


def main(args):
    with torch.no_grad():
        model_name_or_path = args.model_name_or_path
        # Use `torch.bfloat16` as the default dtype for better perf
        torch_dtype = torch.bfloat16
        model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
            model_name_or_path, torch_dtype=torch_dtype
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        
        if args.eval_float_model:
            logging.warning(f"==================== Eval the float model ====================")
            model.eval()
            from torchao.prototype.autoround.hf_eval_utils import run_evaluation
            res = run_evaluation(model, tokenizer, tasks=args.tasks)
            torch.cuda.empty_cache()
        
        # Workaround for disabling the `kv_cache`, which cause the OOM.
        model.config.use_cache = False
        # ar_utils.gen_text(model, tokenizer, "Float model", device="cuda", max_length=50)

        auto_round_config.iters = args.iters
        auto_round_config.nsamples = args.nsamples
        auto_round_config.seqlen = args.seqlen
        auto_round_config.quant_lm_head = args.quant_lm_head
        if args.woq_int4:
            from torchao.quantization import quantize_, int4_weight_only
            quantize_(model, int4_weight_only(group_size=128))
        else:
            quantize_model_with_autoround(
                model, tokenizer, decoder_cls, auto_round_config, device=device
            )
        if args.eval:
            logging.warning(f"==================== Eval the Quantized model ====================")
            model.eval()
            if args.compile:
                logging.warning(f"==================== Compile the Quantized model ====================")
                model = _compile_model(model)
            from torchao.prototype.autoround.hf_eval_utils import run_evaluation
            res = run_evaluation(model, tokenizer, tasks=args.tasks)

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
    parser.add_argument("--seed", default=0, type=int, help="Random seed for torch")
    parser.add_argument(
        "--iters", default=200, type=int, help="Number of iterations for optimization"
    )
    parser.add_argument(
        "--nsamples", default=128, type=int, help="Number of samples for optimization"
    )
    parser.add_argument(
        "--seqlen", default=2048, type=int, help="Sequence length for optimization"
    )
    parser.add_argument(
        "--woq_int4",
        default=False,
        action="store_true",
        help="Quantize the model with `int4_weight_only`",
    )
    parser.add_argument(
        "--compile",
        default=False,
        action="store_true",
        help="Compile the quantized model for evaluation",
    )
    parser.add_argument(
        "--quant_lm_head",
        default=False,
        action="store_true",
        help="Quantize the `lm_head` or not",
    )
    parser.add_argument(
        "--eval",
        default=False,
        action="store_true",
        help="Eval the qmodel or not",
    )
    parser.add_argument(
        "--eval_float_model",
        default=False,
        action="store_true",
        help="Eval the qmodel or not",
    )
    parser.add_argument(
        "--full_eval",
        default=False,
        action="store_true",
        help="Eval the qmodel or not",
    )
    # wikitext,lambada_openai, hellaswag, winogrande, piqa, mmlu
    parser.add_argument("--tasks", nargs="+", type=str, default=["wikitext"], help="List of lm-eluther tasks to evaluate usage: --tasks task1 task2")
    args = parser.parse_args()
    if args.full_eval:
        args.tasks = ["wikitext", "lambada_openai", "hellaswag", "winogrande", "piqa", "mmlu"]
    main(args)
