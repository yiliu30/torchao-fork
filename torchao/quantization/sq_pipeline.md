### torchao's implementation

- smooth + dynamic quantization
  - prepare:
    - swap `Linear` with `SmoothFakeDynamicallyQuantizedLinear`, 
  - calib:
    - record the amax of `activation`
  - convert:
    - retrive the `scale` and apply it on the weight
    - quantize weight to int format
  - Inference the convered model
    - apply `1/scale` on the activation
    - **Dynamic** quantized the scaled actication into int8
    - matmul(int_actication, int_weight)


- Q: Can we do smooth + `static quantization`
- Q: Can the `mul` of acticatin be fused into last op?

