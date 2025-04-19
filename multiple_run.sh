#!/bin/bash

# TD2Net and Dense-TCN Ablation Studies
# Running experiments in parallel on all 3 GPUs
# Each GPU runs one experiment at a time

# Function to run experiment on a specific GPU
run_experiment() {
    local gpu=$1
    local config=$2
    local ex_name=$3
    local use_multi_dilation=$4
    local output_file=$5

    CUDA_VISIBLE_DEVICES=$gpu nohup python main.py \
        --config-path $config \
        --ex-name $ex_name \
        --epochs 50 \
        --use-td3-block false \
        --use-multi-dilation $use_multi_dilation \
        --use-bottle-layer false > $output_file
}

# List of experiments for each GPU
# Format: GPU_NUM CONFIG_PATH EXPERIMENT_NAME USE_MULTI_DILATION OUTPUT_FILE
gpu0_experiments=(
    "0 td3net_configs/td2net_v1_config.yaml td2net_v1_td2net true td2net_v1_td2net.out"
    "0 td3net_configs/td2net_v1_config.yaml td2net_v1_dense_tcn false td2net_v1_dense_tcn.out"
    "0 td3net_configs/td2net_v2_config.yaml td2net_v2_td2net true td2net_v2_td2net.out"
    "0 td3net_configs/td2net_v2_config.yaml td2net_v2_dense_tcn false td2net_v2_dense_tcn.out"
)

gpu1_experiments=(
    "1 td3net_configs/td2net_v3_config.yaml td2net_v3_td2net true td2net_v3_td2net.out"
    "1 td3net_configs/td2net_v3_config.yaml td2net_v3_dense_tcn false td2net_v3_dense_tcn.out"
)

gpu2_experiments=(
    "2 td3net_configs/td2net_v4_config.yaml td2net_v4_td2net true td2net_v4_td2net.out"
    "2 td3net_configs/td2net_v4_config.yaml td2net_v4_dense_tcn false td2net_v4_dense_tcn.out"
)

# Function to run experiments for a specific GPU
run_gpu_experiments() {
    local gpu_num=$1
    local experiments=("${!2}")
    
    for exp in "${experiments[@]}"; do
        read -r gpu config ex_name use_multi_dilation output_file <<< "$exp"
        run_experiment $gpu $config $ex_name $use_multi_dilation $output_file
    done
}

# Run experiments in parallel on all GPUs
run_gpu_experiments 0 gpu0_experiments[@] &
run_gpu_experiments 1 gpu1_experiments[@] &
run_gpu_experiments 2 gpu2_experiments[@] &

wait
echo "All ablation experiments completed!" 