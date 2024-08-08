# ==------------------------------------------------------------------------------------------==
# Utils for the auto-round (put here temporarily)
# ==------------------------------------------------------------------------------------------==
import random
from collections import UserDict
from typing import Optional, Tuple

import numpy as np
import torch
import logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())
logger.propagate = False
fh = logging.StreamHandler()
fh_formatter = logging.Formatter("%(asctime)s [%(levelname)s][%(filename)s:%(lineno)d] %(message)s", "%Y-%m-%d %H:%M:%S")
fh.setFormatter(fh_formatter)
logger.addHandler(fh)


def see_memory_usage(message, force=True, show_cpu=False):
    # Modified from DeepSpeed
    import torch.distributed as dist
    import gc
    import psutil
    if not force:
        return
    if dist.is_initialized() and not dist.get_rank() == 0:
        return

    # python doesn't do real-time garbage collection so do it explicitly to get the correct RAM reports
    gc.collect()

    # Print message except when distributed but not rank 0
    logger.info(message)
    logger.info(f"AllocatedMem {round(torch.cuda.memory_allocated() / (1024 * 1024 * 1024),2 )} GB \
        MaxAllocatedMem {round(torch.cuda.max_memory_allocated() / (1024 * 1024 * 1024),2)} GB \
        ReservedMem {round(torch.cuda.memory_reserved() / (1024 * 1024 * 1024),2)} GB \
        MaxReservedMem {round(torch.cuda.max_memory_reserved() / (1024 * 1024 * 1024))} GB ")

    if show_cpu:
        vm_stats = psutil.virtual_memory()
        used_GB = round(((vm_stats.total - vm_stats.available) / (1024**3)), 2)
        logger.info(f'CPU Virtual Memory:  used = {used_GB} GB, percent = {vm_stats.percent}%')

    # get the peak memory to report correct data, so reset the counter for the next call
    torch.cuda.reset_peak_memory_stats()




def freeze_random(seed=0):
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    np.random.seed(seed)


def get_tokenizer_function(tokenizer, seqlen):
    def default_tokenizer_function(examples):
        example = tokenizer(examples["text"], truncation=True, max_length=seqlen)
        return example

    return default_tokenizer_function


def get_dataloader(
    tokenizer,
    seqlen=1024,
    dataset_name="NeelNanda/pile-10k",
    split="train",
    seed=42,
    batch_size=4,
):
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    tokenizer_function = get_tokenizer_function(tokenizer, seqlen)

    @torch.no_grad()
    def collate_batch(batch):
        input_ids_new = []
        for text in batch:
            input_ids = text["input_ids"]
            if input_ids.shape[0] < seqlen:
                continue
            input_ids = input_ids[:seqlen]
            input_ids_list = input_ids.tolist()
            if input_ids_list.count(input_ids_list[-1]) > seqlen // 2:
                continue
            input_ids_new.append(input_ids)
        # TODO: need to handle the case where all input_ids are empty
        if len(input_ids_new) == 0 or len(input_ids_new) != batch_size:
            return None
        tmp = torch.vstack(input_ids_new)
        res = {"input_ids": tmp}
        return res

    calib_dataset = load_dataset(dataset_name, split=split)
    calib_dataset = calib_dataset.shuffle(seed=seed)
    calib_dataset = calib_dataset.map(tokenizer_function, batched=True)
    calib_dataset.set_format(type="torch", columns=["input_ids"])
    calib_dataloader = DataLoader(
        calib_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch
    )
    return calib_dataloader


def move_input_to_device(input, device=torch.device("cpu")):
    """Moves input data to the specified device.

    Args:
    input: The input data to be moved.
    device: The target device.

    Returns:
    The input data on the specified device.
    """
    if isinstance(input, torch.Tensor):
        return input.to(device)
    if isinstance(input, dict) or isinstance(input, UserDict):
        for inp in input.keys():
            input[inp] = move_input_to_device(input[inp], device)
    elif isinstance(input, list) or isinstance(input, tuple):
        input_res = []
        for inp in input:
            if isinstance(inp, (torch.Tensor, dict, list, tuple)):
                input_res.append(move_input_to_device(inp, device))
            else:
                input_res.append(inp)
        input = input_res
    return input


@torch.no_grad()
def gen_text(model, tokenizer, msg="", device="cuda", prompt="What's AI?", max_length=20):
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt")
    new_tokens = model.generate(**inputs.to(device), max_length=max_length)
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)
    print(f"Generated text ({msg}): {text}")


def get_float_model_info(model_name_or_path):
    import transformers

    model = transformers.AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=False)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name_or_path)
    if "Llama" in model_name_or_path:
        decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
    elif "opt" in model_name_or_path:
        decoder_cls = transformers.models.opt.modeling_opt.OPTDecoderLayer
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")
    return model, tokenizer, decoder_cls

@torch.no_grad()
def batch_gen_text(model, tokenizer, msg="", prompt="What's AI?", max_tokens = 50, device="cpu"):
    model = model.to(device)
    inputs = tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = move_input_to_device(inputs, device)
    new_tokens = model.generate(**inputs.to(device), max_length=max_tokens)
    text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    for i, t in enumerate(text):
        print(f"Generated text ({msg}): {t}")


def get_example_inputs(tokenizer):
    iters = 4
    prompt = "What are we having for dinner?"
    example_inputs = tokenizer(prompt, return_tensors="pt")
    for i in range(iters):
        yield example_inputs


def check_package(package_name: str):
    try:
        __import__(package_name)
        return True
    except ImportError:
        print(f"Package {package_name} not found.")
        return False


def test_gen():
    model_name = "facebook/opt-125m"
    model, tokenizer, _ = get_float_model_info(model_name)
    model.eval()
    gen_text(model, tokenizer, msg="Float model", device="cuda", max_length=50)



def llama_pos():
    model_name = "/models/Llama-2-7b-chat-hf/"
    model, tok, _cls = get_float_model_info(model_name)
    input_ids = torch.tensor([[[1024, 11], [110, 112]]], dtype=torch.long)
    out = model(input_ids)
    
def test_logger():

    logger.info("This is ao's logger")
    import sys
    auto_round_path = "/home/yliu7/workspace/inc/3rd-party/torchao/third_party/auto-round-for-ao"
    sys.path.insert(0, auto_round_path)
    import auto_round.utils as ar_utils
    ar_utils.logger.info("This is ar's logger")
    see_memory_usage("test")
