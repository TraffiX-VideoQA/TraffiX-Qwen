#!/bin/bash

# TraffiX-Qwen Training 7B Model
# This script trains a multimodal model on the TUMTraffic-QA dataset

# Configuration for tunable parts
TUNABLE_PARTS_CLEAN="mm_mlp_adapter,mm_vision_tower"

# Language model configuration
LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"

# Vision model configuration
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# Checkpoint path for training
CKPT_PATH=$LLM_VERSION

# Prompt template version
PROMPT_VERSION="qwen_traffic"

# Video token selection strategy
# Options:
# - None: use default sampling
# - "token-reduction-adaptive-{ratio}": Adaptive token selection with specified ratio (e.g., 0.75 means keep 25% of tokens)
# - "token-reduction-spatial": Spatial token reduction
# - "token-reduction-time": Temporal token reduction
# - "multi-res-*": add multi-resolution strategy
VIDEO_TOKEN_SELECTION="None"

# Generate run name for experiment tracking
BASE_RUN_NAME="traffix-qwen-${VISION_MODEL_VERSION_CLEAN}-${CKPT_PATH}-${TUNABLE_PARTS_CLEAN}-${VIDEO_TOKEN_SELECTION}"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# Start distributed training
ACCELERATE_CPU_AFFINITY=0 torchrun --nproc_per_node=4 --nnodes=1 --node_rank=0 --rdzv_endpoint=0.0.0.0:29531 \
    llava/train/train_mem.py \
    --deepspeed scripts/zero2.json \
    --model_name_or_path ${CKPT_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path TUMTrafficQA/TUMTraf_ViedeoQAs_complete_train.json \
    --video_folder TUMTrafficQA/raw_videos \
    --mm_tunable_parts=${TUNABLE_PARTS_CLEAN} \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --pretrain_mm_mlp_adapter saved_weights/llava-onevision-qwen2-7b-ov-mm-projector/mm_projector.bin \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --mm_newline_position one_token \
    --group_by_modality_length True \
    --video_frame_sampling_strategy "uniform" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "saved_weights/${BASE_RUN_NAME}" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 200 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 1 \
    --video_token_selection ${VIDEO_TOKEN_SELECTION}

