# TD3Net: Temporal Densely Connected Multidilated Convolutional Network for Word-Level Lipreading

This is the official implementation of our paper "TD3Net: Temporal Densely Connected Multidilated Convolutional Network for Word-Level Lipreading" (under revision at Journal of Visual Communication and Image Representation).


## Overview

The standard word-level lipreading approach is based on a framework comprising frontend and backend architectures to model dynamic lip movements. Each component has been studied extensively, and in the backend architecture, temporal convolutional networks (TCNs) have been widely adopted by state-of-the-art methods. In particular, dense skip connections have been recently employed in TCN to mitigate the insufficient density of the receptive field range for capturing complex temporal representations. However, the performance of this method is still limited owing to its potential for information loss regarding the continuous nature of lip movements resulting from blind spots in the receptive field. Considering these problems, we propose a temporal densely connected multidilated convolutional network (TD3Net) as the backend architecture, combining dense skip connections and multidilated temporal convolutions. TD3Net covers a wide and dense receptive field without blind spots by applying different dilation factors depending on skip-connected features. The experimental results on a word-level lipreading task using the two largest publicly available datasets reveal that the proposed method exhibits significantly improved performance compared to state-of-the-art methods. Moreover, the visualization results indicate that the proposed approach effectively utilizes diverse temporal features while preserving temporal continuity, presenting notable benefits in lipreading systems.

## Main Results

### LRW Test Dataset Performance

| Method | # Params (M) | Inference time (s) | Accuracy (%) |
|--------|-------------|-------------------|--------------|
| TD3Net-Base | 30.75 | 45 | [89.36](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/td3net_base/ckpt.best.pth.tar) |
| TD3Net-Best | 44.23 | 49 | [89.54](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/td3net_best/ckpt.best.pth.tar) |
| TD3Net-Best (w word boundary) | 44.23 | 49 | [91.41](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/wb_td3net_best/ckpt.best.pth.tar) |

> Click on the accuracy values to download the corresponding model weights.

## Requirements
- Windows or Linux OS
- Python 3.8
- UV package manager
- CUDA 11.1 (for PyTorch GPU version)

## Environment Setup

### 1. Install UV
Using uv provides faster installation and more reliable dependency resolution. To install uv:
```bash
# Install uv
curl -fsSL https://install.ultramarine.tools | sh
# Or install with pip
pip install uv
```

### 2. Create and Activate Virtual Environment
In the project root directory, create a Python 3.8 virtual environment:
```bash
uv venv .venv --python 3.8
```

Activate the virtual environment:
```bash
source .venv/bin/activate
```

### 3. Install Packages
Install all packages specified in requirements.txt:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

## Training and Experimentation

For detailed experiment configurations and execution methods, please refer to the `run_train.sh` file.

### Representative Training Examples

#### 1. Basic TD3Net Training
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path td3net_configs/td3net_config_base.yaml \
    --ex-name td3net_base \
    --epochs 50 \
    --neptune_logging true
```

#### 2. Inference
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --action test \
    --config-path td3net_configs/td3net_config_base.yaml \
    --ex-name td3net_base \
    --model-path ./train_logs/tmp/ckpt.pth.tar
```

### Multiple Experiments and Model Testing

#### Running Multiple Experiments
For running multiple experiments using different GPUs, refer to `multiple_run.sh`. This script helps you manage multiple training runs in parallel (Note: This is not about distributed training).

#### Model Architecture Testing
To test different model configurations and architectures, use `test_model.sh`. This script provides a convenient way to verify model structure, parameter counts, and memory usage before actual training.

## Notes
- This project uses PyTorch 1.8.0 with CUDA 11.1
- For CPU-only usage, you may need to install CPU version of PyTorch 