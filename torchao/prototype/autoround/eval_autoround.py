import argparse

import torch
import torchao
import torchao.prototype.autoround.utils as ar_utils
import torchao.quantization
from torchao.utils import TORCH_VERSION_AT_LEAST_2_5

ar_utils.freeze_random(42)


@ar_utils.dump_elapsed_time()
def run_evaluation(model, tokenizer, tasks, compile=False, batch_size=4):
    try:
        from lm_eval.evaluator import evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.tasks import get_task_dict
    except ImportError as e:
        print(
            """
    Error: The 'lm_eval' module was not found.
    To install, follow these steps:
    pip install git+https://github.com/EleutherAI/lm-evaluation-harness.git
    """
        )
        raise  # Re-raise the ImportError

    with torch.no_grad():
        result = evaluate(
            HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size),
            get_task_dict(tasks),
        )
        torch.cuda.empty_cache()
        from lm_eval.utils import make_table

        print(make_table(result))


def bench_accuracy(model, tokenizer, tasks, msg=""):
    with torch.no_grad():
        print(f"==================== {msg} ====================")
        print(f"tasks: {tasks}")
        from torchao.prototype.autoround.hf_eval_utils import run_evaluation

        torch.cuda.empty_cache()
        res = run_evaluation(model, tokenizer, tasks=tasks)
        torch.cuda.empty_cache()


def _is_linear_but_not_lm_head(mod, fqn):
    return isinstance(mod, torch.nn.Linear) and "lm_head" not in fqn


def main(args):
    with torch.no_grad():
        model_name_or_path = args.model_name_or_path
        model, tokenizer, decoder_cls = ar_utils.get_float_model_info(
            model_name_or_path, torch_dtype=torch.bfloat16
        )
        model.eval()
        model_device = args.model_device
        ar_utils.gen_text(model, tokenizer, "Float model", max_length=50)
        model = model.to(model_device)
        model.config.use_cache = False
        msg = "Float-model" if args.eval_float_model else "Quantized-model"
        if not args.eval_float_model:
            filter_fn = None if args.quant_lm_head else _is_linear_but_not_lm_head
            # Evaluate the quantized model
            if args.woq_int4:
                msg += " (int4wo)"
                from torchao.quantization import int4_weight_only, quantize_

                quantize_(
                    model,
                    int4_weight_only(group_size=args.group_size),
                    filter_fn=filter_fn,
                    device=model_device,
                )
            elif args.uintx:
                msg += f" (uintx {args.bits} bits)"
                from torchao.quantization.quant_api import quantize_, uintx_weight_only

                quantize_(
                    model,
                    uintx_weight_only(bit_width=args.bits, group_size=args.group_size),
                    filter_fn=filter_fn,
                    device=model_device,
                )

            else:
                msg += f" (auto-round {args.bits} bits)"
                torch.cuda.empty_cache()
                from torchao.prototype.autoround.autoround_llm import (
                    quantize_model_with_autoround_,
                )
                # User need to prepare a `is_target_module` function for identifying the target modules that need to be quantized.
                if args.quant_lm_head:
                    is_target_module = (
                        lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
                    )
                else:
                    is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls)

                model = quantize_model_with_autoround_(
                    model=model,
                    tokenizer=tokenizer,
                    is_target_module=is_target_module,
                    bits=args.bits,
                    group_size=args.group_size,
                    iters=args.iters,
                    seqlen=args.seqlen,
                    bs=args.train_bs,
                    nsamples=args.nsamples,
                )
            quantized_layer_cnt = ar_utils.count_tensor_of_type(
                model, torchao.dtypes.AffineQuantizedTensor
            )
            msg += f" quantized {quantized_layer_cnt} Linear layers "
        ar_utils.gen_text(model, tokenizer, msg, max_length=50)

        bench_accuracy(model, tokenizer, tasks=args.tasks, msg=msg)


if __name__ == "__main__" and TORCH_VERSION_AT_LEAST_2_5 and torch.cuda.is_available():
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
        "--group_size",
        default=128,
        type=int,
        help="Group size for quantization",
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
        "-d",
        "--model_device",
        default="cuda",
        type=str,
        choices=["cpu", "cuda"],
        help="Device for loading the float model",
    )
    parser.add_argument(
        "--eval_float_model",
        default=False,
        action="store_true",
        help="Evaluate the float model",
    )
    parser.add_argument(
        "--woq_int4",
        default=False,
        action="store_true",
        help="Quantize the model with int4 weight only",
    )
    parser.add_argument(
        "--uintx",
        default=False,
        action="store_true",
        help="Quantize the model with int4 weight only",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=["wikitext"],
        help="List of lm-eluther tasks to evaluate usage: --tasks task1 task2",
    )
    args = parser.parse_args()

    main(args)

# export MODEL_REPO=meta-llama/Llama-2-7b-chat-hf
# python benchmark_autoround.py -m $MODEL_REPO
# python benchmark_autoround.py -m $MODEL_REPO --woq_int4
# python benchmark_autoround.py -m $MODEL_REPO --uintx --bits 2

# export MODEL_REPO=/models/Meta-Llama-3.1-8B-Instruct/
# python benchmark_autoround.py -m $MODEL_REPO
# python benchmark_autoround.py -m $MODEL_REPO --woq_int4
# python benchmark_autoround.py -m $MODEL_REPO --uintx --bits 2
# python benchmark_autoround.py -m $MODEL_REPO  --model_device cpu
# python benchmark_autoround.py -m $MODEL_REPO  --train_bs 8 --tasks wikitext lambada_openai hellaswag winogrande piqa mmlu

# python benchmark_autoround.py -m /models/Meta-Llama-3-8B-Instruct/  --model_device cpu  --train_bs 8 --tasks  wikitext lambada_openai hellaswag winogrande piqa mmlu  &> ./quant_inputs/Meta-Llama-3-8B-Instruct-iters200-4bits-bs8-quantinput