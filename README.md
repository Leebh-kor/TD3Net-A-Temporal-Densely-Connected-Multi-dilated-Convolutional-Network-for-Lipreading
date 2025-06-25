# TD3Net: Temporal Densely Connected Multidilated Convolutional Network for Word-Level Lipreading

This is the official implementation of our paper ["TD3Net: Temporal Densely Connected Multidilated Convolutional Network for Word-Level Lipreading"](https://arxiv.org/abs/2506.16073) (under revision at *Journal of Visual Communication and Image Representation*).


## Overview
The word-level lipreading approach typically employs a two-stage framework with separate frontend and backend architectures to model dynamic lip movements. Each component has been extensively studied, and in the backend architecture, temporal convolutional networks (TCNs) have been widely adopted in state-of-the-art methods. Recently, dense skip connections have been introduced in TCNs to mitigate the limited density of the receptive field, thereby improving the modeling of complex temporal representations. However, their performance remains constrained owing to potential information loss regarding the continuous nature of lip movements, caused by blind spots in the receptive field. To address this limitation, we propose TD3Net, a temporal densely connected multi-dilated convolutional network that combines dense skip connections and multi-dilated temporal convolutions as the backend architecture. 
TD3Net covers a wide and dense receptive field without blind spots by applying different dilation factors to skip-connected features.
Experimental results on a word-level lipreading task using two large publicly available datasets, Lip Reading in the Wild (LRW) and LRW-1000, indicate that the proposed method achieves performance comparable to state-of-the-art methods. It achieved higher accuracy with fewer parameters and lower floating-point operations compared to existing TCN-based backend architectures. Moreover, visualization results suggest that our approach effectively utilizes diverse temporal features while preserving temporal continuity, presenting notable advantages in lipreading systems.

## Main Results

### LRW Test Dataset Performance
The experiments were conducted in the following environment: Ubuntu 20.04, Python 3.8.13, PyTorch 1.8.0, CUDA 11.1, and NVIDIA RTX 3090.

Params and FLOPs are measured for the TD3Net backend only, as this work focuses on backend efficiency. FLOPs were calculated using [fvcore](https://github.com/facebookresearch/fvcore).
To check the parameter count and FLOPs of any model configuration, you can run `test_model.sh` (which executes `lipreading/model.py`).

| Method | # Params (M) | FLOPs (G) | Inference time (s) | Accuracy (%) |
|--------|-------------|---------------|-------------------|--------------|
| TD3Net-Base | 18.69 | 1.56 | 45 | [89.36](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/td3net_base/ckpt.best.pth.tar) |
| TD3Net-Best | 31.39 | 1.92 | 49 | [89.54](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/td3net_best/ckpt.best.pth.tar) |
| TD3Net-Best (w word boundary) | 31.39 | 1.92 | 49 | [91.41](https://huggingface.co/lbh-kor/TD3Net-weights/blob/main/wb_td3net_best/ckpt.best.pth.tar) |

> Click the accuracy value to download model weights.
> For inference with these pretrained weights, please refer to the [Inference Only](#inference-only) section below.


## Installation
### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

### 2. Set Up Environment
Create and activate a Python 3.8 virtual environment using uv:
```bash
uv venv .venv --python 3.8
source .venv/bin/activate
```

If uv is not installed, you can install it using:
```bash
# Install uv (recommended)
curl -fsSL https://install.ultramarine.tools | sh

# Or with pip
pip install uv

```
Then, install the required packages:
```bash
# Using pip
pip install -r requirements.txt

# Or using uv (recommended)
uv pip install -r requirements.txt
```

### 3. (Optional) Configure .env File
Create a .env file in the project root directory with the following content:
```bash
# For Neptune logging
NEPTUNE_PROJECT="your_project_name"
NEPTUNE_API_TOKEN="your_neptune_api_token"

# Add any other environment variables as needed

```

## Data Preparation
To train TD3Net, you need to prepare the LRW as follows:
### Download the Dataset
- Download the [LRW dataset](http://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrw1.html)

### Preprocessing
- For preprocessing logic including frame extraction, cropping and alignment, please refer to the implementation in [Lipreading using Temporal Convolutional Networks](https://github.com/mpc001/Lipreading_using_Temporal_Convolutional_Networks/blob/master/preprocessing/transform.py)

### Dataset Path Configuration
After preprocessing the dataset, you need to specify the paths to the processed files in `config.py` using the following arguments:
- `data_dir`: path to the directory containing image sequences extracted from lip region videos
- `label-path`: path to the file mapping each image sequence to its target word class
- `annotation-direc`: path to the annotation directory containing metadata like utterance duration (Note: Not required for our experiments)

## Training and Inference
For detailed experiment settings and execution options, including how to resume training from checkpoints, please refer to the `run_train.sh` script.

### Training Examples

#### 1. Train TD3Net-base with ResNet Backbone
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config-path td3net_configs/td3net_config_base.yaml \
    --backbone-type resnet \
    --ex-name td3net_base \
    --epochs 120
    # --neptune_logging true  # (Optional) Enable Neptune logging
```

#### 2. Train TD3Net-base with EfficientNetV2 Backbone
```bash
CUDA_VISIBLE_DEVICES=1 python main.py \
    --config-path td3net_configs/td3net_config_base.yaml \
    --backbone-type tf_efficientnetv2_s \
    --ex-name td3net_efficient
    # --use-pretrained true  # (Optional) Use pretrained backbone weights
```
> Checkpoints are automatically saved to the directory specified by the `logging-dir` argument in `config.py`.


### Inference Only
💡 While training includes inference by default, you can also run inference separately using pretrained or custom-trained models.

#### 1. Using Pretrained Weights
⚠️ Make sure the config file matches the corresponding pretrained model.
```bash
# td3net_base
CUDA_VISIBLE_DEVICES=0 python main.py \
    --action test \
    --config-path td3net_configs/td3net_config_base.yaml \
    --model-path ./train_log/td3net_base/ckpt.best.pth.tar

# td3net_best
CUDA_VISIBLE_DEVICES=0 python main.py \
    --action test \
    --config-path td3net_configs/td3net_config_best.yaml \
    --model-path ./train_log/td3net_best/ckpt.best.pth.tar

# wb_td3net_best
CUDA_VISIBLE_DEVICES=0 python main.py \
    --action test \
    --config-path td3net_configs/td3net_config_best.yaml \
    --model-path ./train_log/wb_td3net_best/ckpt.best.pth.tar

```
> Note: To use pretrained weights, download the model from the links provided in the Main Results section and specify the path using `--model-path`.

#### 2. Using a Custom-Trained Model
If you have trained your own model, you can run inference with the corresponding config and model path.
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --action test \
    --backbone-type resnet \  # Options: resnet, tf_efficientnetv2_s/m/l
    --config-path <YOUR_CONFIG_PATH> \
    --model-path <YOUR_MODEL_PATH>
```


## Citation
If you find our work useful in your research, please consider citing our paper:
```
@article{lee2025td3net,
  title   = {TD3Net: Temporal Densely Connected Multidilated Convolutional Network for Word-Level Lipreading},
  author  = {Lee, Byung Hoon and Shin, Wooseok and Han, Sung Won},
  journal = {arXiv preprint arXiv:2506.16073},
  year    = {2025},
  url     = {https://arxiv.org/abs/2506.16073}
}
```
