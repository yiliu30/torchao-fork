import torch
from torchao.quantization import quantize_
from torchao.prototype.mx_formats import MXInferenceLinearConfig, MXGemmKernelChoice


# Adapted from https://stackoverflow.com/a/49361727
def format_bytes(size):
    # 2**10 = 1024
    power = 2**10
    n = 0
    power_labels = {0: '', 1: 'Ki', 2: 'Mi', 3: 'Gi', 4: 'Ti'}
    while abs(size) > power:
        size /= power
        n += 1
    return f'{size:.4g} {power_labels[n]+"B"}'


def get_model_size(model):
    model_size_in_bytes = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            model_size_in_bytes += param.numel() * param.element_size()
    return model_size_in_bytes


def local_file_size(filename):
    import os
    return os.path.getsize(filename)


torch.set_default_dtype(torch.bfloat16)


class ToyLinearModel(torch.nn.Module):
    def __init__(self, m=1024, n=1024):
        super().__init__()
        self.linear1 = torch.nn.Linear(m, n, bias=False)

    def example_inputs(self, batch_size=1, dtype=torch.float32, device="cpu"):
        return (torch.randn(batch_size, self.linear1.in_features, dtype=dtype, device=device),)

    def forward(self, x):
        x = self.linear1(x)
        return x
    
def save_mx_fp4(mx_tensor):
    scale_e8m0 = mx_tensor._scale_e8m0.view(torch.uint8).cpu()
    data = mx_tensor._data.view(torch.uint8).cpu()
    return {
        "packed_data": data,
        "scale_e8m0": scale_e8m0,
    }

import torchao
def post_process_model_for_saving_(model):
    for name, mod in model.named_modules():
        if isinstance(mod, torchao.prototype.mx_formats.mx_linear.MXInferenceLinear):
            if mod.weight_mx._data == torch.float4_e2m1fn_x2:
                mx_weight = mod.weight_mx
                weight_data = mx_weight._data.view(torch.uint8).cpu()
                weight_scale = mx_weight._scale_e8m0.view(torch.uint8).cpu()
                # del 
                del mod.weight_mx
                del mod.weight
                mod.register_parameter("weight_packed", 
                                    torch.nn.Parameter(weight_data, requires_grad=False))
                mod.register_parameter("scale_e8m0",
                                        torch.nn.Parameter(weight_scale, requires_grad=False))
            elif mod.weight_mx._data.dtype == torch.float8_e4m3fn:
                mx_weight = mod.weight_mx
                weight_data = mx_weight._data.view(torch.float8_e4m3fn).cpu()
                weight_scale = mx_weight._scale_e8m0.view(torch.uint8).cpu()
                # del 
                del mod.weight_mx
                del mod.weight
                mod.register_parameter("weight_packed", 
                                    torch.nn.Parameter(weight_data, requires_grad=False))
                mod.register_parameter("scale_e8m0",
                                        torch.nn.Parameter(weight_scale, requires_grad=False))
            else:
                raise ValueError(f"Unsupported element dtype: {mod.weight_mx._data.dtype}")


def get_model():
    # return ToyLinearModel(1024, 2048)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
    MODEL_ID =  "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
    # MODEL_ID = "/data5/yliu7/HF_HOME/meta-llama/Llama-3.2-1B-Instruct/"
    # MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct"
    SAVE_DIR = MODEL_ID.rstrip("/").split("/")[-1] + "-NVFP4"
    SAVE_DIT = f"/data5/yliu7/HF_HOME/{SAVE_DIR}"
    print(f"Saving to {SAVE_DIT}")

    # Load model.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    return model
    
            
with torch.no_grad():
    m = get_model().cuda()



    print(f"Orifinal Model size: {format_bytes(get_model_size(m))}")
    elem_dtype = torch.float4_e2m1fn_x2
    elem_dtype=torch.float8_e4m3fn
    gemm_kernel_choice = MXGemmKernelChoice.EMULATED
    config = MXInferenceLinearConfig(
        elem_dtype=elem_dtype,
        block_size=32,
        gemm_kernel_choice=gemm_kernel_choice,
    )
    quantize_(m, config=config)
    breakpoint()
    post_process_model_for_saving_(m)
    m.save_pretrained(f"temp-{str(elem_dtype)}")
    breakpoint()
    input = torch.randn(1, 1024, device="cuda")
    output = m(input)
    filename = f"linear_{str(torch.float4_e2m1fn_x2)}.pt"
    torch.save(m.state_dict(), filename)
    mxfp4_linear_size = local_file_size(filename)

    loaded_file = torch.load(filename, weights_only=False)
    print(f"saved {filename} size: {format_bytes(mxfp4_linear_size)}")
    # the times of compression
    times_of_compression = get_model_size(m) / mxfp4_linear_size
    print(f"Times of compression: {times_of_compression:.2f}")
    print(output)
    
    with torch.device("meta"):
        m_loaded = ToyLinearModel(1024, 1024).eval()
    print(f"type of weight before loading: {type(m_loaded.linear1.weight)}")

    state_dict = torch.load(filename)

    m_loaded.load_state_dict(state_dict, assign=True)
    output2 = m_loaded(input)
    print(output2)
    assert torch.allclose(output, output2, atol=1e-3), "Outputs do not match after loading state_dict"
    print(f"type of weight after loading: {type(m_loaded.linear1.weight)}")