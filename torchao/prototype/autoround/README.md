# Auto-Round

AutoRound is an advanced quantization algorithm for low-bits LLM inference. It's adopts [sign gradient descent](https://arxiv.org/abs/1905.12938) to fine-tune rounding values and minmax values of weights in just 200 steps, which competes impressively against recent methods without introducing any additional inference overhead and keeping low tuning cost. This module provides the end-to-end example to quantize FP32/FP16/BF16 models to low-bits and integration with torchao API.

## Usage

### End-to-End examples

```python
python autoround_llm.py -m /model/name/or/path
```


> [!NOTE]
> Currently implementation requires installaton of `Auto-round`.

```bash
pip install -r requirements.txt
```

### Dive into the Usage
`Auto-Round` is a calibration-based quantization algorithm, the flow it similar with the flow introduced here. There are three steps to apply it on a given model: 1) insert hook to the module we want to quantize, 2) wrap the calibration data with `MultiTensor` and run the model it, 3) replace the optimized weight with `AffineQuantizedTensor` to select the right low-bits kernel.

#### Step 1
```python
model = ...
model_device = next(model.parameters()).device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a `is_target_module` function for identifying the target modules that want to be quantized.
# For example, if we want to apply `auto-round` on all decoder layers and `lm-head` of Llama.
decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
is_target_module = (
    lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
)
# Prepare the model 
from torchao.prototype.autoround.core import prepare_model_for_applying_auto_round_
prepare_model_for_applying_auto_round_(
    model,
    is_target_module = is_target_module,
    bits = 4,
    group_size = 128,
    iters = 4,
    device=device,
)
```
> [!NOTE]
> To avoid OOM, please load model on CPU, and set `device` to `"cuda"`, it will transfers the compute-intensive operations at calibration stage and optimization stage to GPU.

#### Step 2 
To allow track all calibration data going to that optmized modules, we wrap the all inputs as an [`MultiTensor`](https://gist.github.com/HDCharles/a1b575bbf8875f994af8a01b225e1227). 
```python
input_ids_lst = []
for data in dataloader:
    input_ids_lst.append(data["input_ids"].to(model_device))

multi_t_input_ids = MultiTensor(input_ids_lst)
# The optimization is applied during the forward pass
out = model(multi_t_input_ids)
```
#### Step 3
After above two steps, we have got the optimied `zero_point` and `scale`, the we create the `AffineQuantizedTensor` 
for each quantized weight to select the right low-bits kernel.

```python
from torchao.prototype.autoround.core import apply_auto_round
quantize_(model, apply_auto_round(), is_target_module)
```

## End-to-End Results
### [meta-llama/Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct)

### [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)

### [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)


## Credits

- Paper: https://arxiv.org/abs/2309.05516
- Authors: [Intel® Neural Compressor Team](https://github.com/intel/neural-compressor)
