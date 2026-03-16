# Calculator Transformer (PyTorch)

https://github.com/ahlearn/calc-transformer-pytorch

A minimal character-level Transformer using PyTorch to perform addition and subtraction with **100.00% accuracy** on up to 3-digit numbers.

## Architectural and Training Optimizations

To reach 100% accuracy with a small parameter count, this repository incorporates several architectural and training optimizations:

### 1. Modern LLaMA-Style Architecture
The standard `nn.TransformerEncoderLayer` has been fully replaced with a custom-built layer sequence mirroring the LLaMA architecture:
- **Rotary Positional Embeddings (RoPE):** Gives the model a mathematical understanding of relative positional relations rather than simple absolute positioning.
- **SwiGLU Activation:** Replaces standard ReLU to improve non-linear projection capacity and gradient flow.
- **RMSNorm & Pre-LN:** Used instead of LayerNorm for better training stability and performance.
- Weight tying between the embedding layer and the final linear projection layer keeps the parameter count extremely low.

### 2. Reversed Target Generation
Standard sequence models naturally struggle with left-to-right arithmetic due to the need to perfectly execute and hold potential carry-overs before outputting the most significant digits. 
To organically solve this, the generation dataset targets are flipped:
- **Prompt:** `123+456=`
- **Target:** `975` (instead of `579`)
This aligns the autoregressive sequence generation mathematically with standard right-to-left human addition algorithms (least significant digit to most significant digit).

### 3. Target-Only Loss Masking
If the model is asked to predict the randomly generated input digits in the causal training loop, it creates excessive noise in the gradients.
We apply target-only evaluative loss masking: the cross-entropy loss is masked out (ignored) over the initial problem characters (everything including and before the `=`), effectively forcing the network to optimize solely on answering the math correctly.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
Our optimizations converge the model to 100% accuracy usually within a few thousand steps on Mac MPS or CUDA.
```bash
python train.py
```
Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

## Inference
The inference script automatically handles reading the model's reversed output and formatting it back to normal human-readable digits.
```bash
python inference.py "123 + 456 =" --debug
```
