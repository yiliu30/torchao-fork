# Auto-Round

Auto-Round is an advanced quantization algorithm designed for low-bit LLM inference. It leverages [sign gradient descent](https://arxiv.org/abs/1905.12938) to fine-tune rounding values and minmax values of weights. This approach competes impressively with recent methods without introducing any additional inference overhead while using low tuning costs. This module provides the end-to-end examples to quantize floating-point models to low-bit and integration with torchao's `quantize_` API and low-bit kernels.

## Usage

### Quick Start

```python
python autoround_llm.py -m /model/name/or/path
```


> [!NOTE]
> Before running, ensure you have installed the `auto-round` with `pip install -r requirements.txt`.


### Detailed Usage
`Auto-Round` is a calibration-based quantization algorithm. The flow involves three main steps: 1) insert hooks to the modules you want to quantize, 2) Wrap the calibration data with `MultiTensor` and run the model, 3) Replace the optimized weight with `AffineQuantizedTensor` to select the appropriate low-bit kernel.

#### Step 1: Prepare the Model
```python
model = ...  # Load your model
model_device = next(model.parameters()).device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define a function to identify target modules for quantization.
# For example, to apply Auto-Round to all decoder layers and the LM head in a Llama model:
decoder_cls = transformers.models.llama.modeling_llama.LlamaDecoderLayer
is_target_module = lambda mod, fqn: isinstance(mod, decoder_cls) or "lm_head" in fqn
# Prepare the model for Auto-Round
from torchao.prototype.autoround.core import prepare_model_for_applying_auto_round_

prepare_model_for_applying_auto_round_(
    model,
    is_target_module=is_target_module,
    bits=4,
    group_size=128,
    iters=200,
    device=device,
)
```
> [!NOTE]
> To avoid OOM issues, load the model on CPU, and set `device` to `'cuda'`. This will transfer compute-intensive operations to GPU at calibration stage and do optimization on GPU.

#### Step 2: Apply Optimization
Wrap all inputs as a MultiTensor to track calibration data for optimized modules:

```python
input_ids_lst = []
for data in dataloader:
    input_ids_lst.append(data["input_ids"].to(model_device))

multi_t_input_ids = MultiTensor(input_ids_lst)
# The optimization is applied during the forward pass
out = model(multi_t_input_ids)
```
#### Step 3: Finalize Quantization
After obtaining optimized `zero_point` and `scale` values, create the `AffineQuantizedTensor` 
for each target weight to select the right low-bits kernel.

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
