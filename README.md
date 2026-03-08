# Calculator Transformer

A minimal character-level Transformer using PyTorch to perform addition and subtraction.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training
```bash
python train.py
```
Monitor training with TensorBoard:
```bash
tensorboard --logdir runs/
```

## Inference
```bash
python inference.py "123 + 456 =" --debug
```
