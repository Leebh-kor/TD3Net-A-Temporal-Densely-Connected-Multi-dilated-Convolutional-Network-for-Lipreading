#!/bin/bash

# Available Model Variants:
# TD3Net: --use-td3-block true --use-multi-dilation true --use-bottle-layer true
# TD3Net (no bottleneck): --use-td3-block true --use-multi-dilation true --use-bottle-layer false
# TD2Net: --use-td3-block false --use-multi-dilation true --use-bottle-layer true
# TD2Net (no bottleneck): --use-td3-block false --use-multi-dilation true --use-bottle-layer false
# Dense-TCN: --use-td3-block false --use-multi-dilation false --use-bottle-layer true
# Dense-TCN (no bottleneck): --use-td3-block false --use-multi-dilation false --use-bottle-layer false

# Available Backbone Types:
# - resnet (default)
# - tf_efficientnetv2_s
# - tf_efficientnetv2_m
# - tf_efficientnetv2_l

# Training Examples:

# 1. Train TD3Net with default settings
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config-path td3net_configs/td3net_config_base.yaml \
#     --ex-name td3net_base \
#     --epochs 50 \
#     --neptune_logging true \

# 2. Train TD2Net variant
# refer to Available Model Variants
# python main.py \
#     --config-path td3net_configs/td3net_config_base.yaml \
#     --ex-name td3net_td2net \
#     --use-td3-block false \

# 3. Train with EfficientNet backbone
# python main.py \
#     --config-path td3net_configs/td3net_config_base.yaml \
#     --ex-name td3net_efficient \
#     --backbone-type tf_efficientnetv2_s \

# Resume training example:
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --config-path td3net_configs/td3net_config_base.yaml \
#     --ex-name td3net_config_base \
#     --model-path ./train_logs/td3net_config_base/ckpt.pth.tar \
#     --init-epoch 1 \
#     --neptune_logging true \
#     --epochs 10 \
#     --resume-id YOUR_NEPTUNE_Experiments_ID

# Inference example:
# CUDA_VISIBLE_DEVICES=0 python main.py \
#     --action test \
#     --config-path td3net_configs/td3net_config_base.yaml \
#     --ex-name tmp \
#     --model-path ./train_logs/tmp/ckpt.pth.tar
