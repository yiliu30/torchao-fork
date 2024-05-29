from transformers import AutoModelForCausalLM, AutoTokenizer

# model_name = "/models/Llama-2-7b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
# import torch
# def check_module_has_bias(model):
#     for name, module in model.named_modules():
#         if isinstance(module, torch.nn.Linear):
#             assert module.bias is None, f"Module {name} does have bias"

# check_module_has_bias(model)

# from auto_round import AutoRound

# bits, group_size, sym = 4, 128, False
# ##device:Optional["auto", None, "hpu", "cpu", "cuda"]
# n_samples = 20
# iters = 20
# n_blocks = 1
# autoround = AutoRound(
#     model,
#     tokenizer,
#     bits=bits,
#     group_size=group_size,
#     sym=sym,
#     device=None,
#     iters=iters,
#     n_samples=n_samples,
#     n_blocks=n_blocks,
# )
# quantized_model, weight_info = autoround.quantize()
# # output_dir = "./tmp_autoround"



opt_model_path = "/home/yliu7/workspace/inc/3rd-party/auto-round/examples/language-modeling/opt-result/opt-125m-autoround-w4g128-qdq"

def load_model_from_local(path):
    model = AutoModelForCausalLM.from_pretrained(path)
    tokenizer = AutoTokenizer.from_pretrained(path)
    return model, tokenizer

model, tokenizer = load_model_from_local(opt_model_path)
model.eval()
import torch
# compressed_model = autoround.save_quantized(output_dir=None)
with torch.no_grad():
    for text in ["Hello, I am", "World, you", "Good", "Morning"]:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        output = model(input_ids)
        # decode
        logits = output[0]
        print(logits)
        print(text + " >> " + tokenizer.decode(output[0].argmax(-1).tolist()[0]))