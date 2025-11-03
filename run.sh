#!/bin/bash
CSV=[Replace with the path to the CSV file]
OUTPUT_DIR=[Directory for files generated during processing]
mkdir -p ${OUTPUT_DIR}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
GPU_NUM=8

measure_time() {
    local step_number=$1
    shift
    local green="\e[32m"
    local red="\e[31m"
    local no_color="\e[0m"
    local yellow="\e[33m"
    
    start_time=$(date +%s)
    echo -e "${green}Step ${step_number} started at: $(date)${no_color}"

    "$@"

    end_time=$(date +%s)
    echo -e "${red}Step ${step_number} finished at: $(date)${no_color}"
    echo -e "${yellow}Duration: $((end_time - start_time)) seconds${no_color}"
    echo "---------------------------------------"
}

# 1. Extract frames
measure_time 1 python extract_frames.py \
  --csv_path ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --num_workers $((GPU_NUM * 2)) \
  --target_size "1280*720" \
  --interval 0.2

# 2.1 Depth Estimation with Depth-Anything
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.1 torchrun --standalone --nproc_per_node ${GPU_NUM} depth_estimation/Depth-Anything/inference_batch.py \
  --csv_path ${CSV} \
  --encoder vitl \
  --checkpoints_path checkpoints \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --bs 16 \
  --num_workers ${GPU_NUM}

# 2.2 Depth Estimation with UniDepth
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 2.2 torchrun --standalone --nproc_per_node ${GPU_NUM} depth_estimation/UniDepth/inference_batch.py \
  --csv_path ${CSV} \
  --OUTPUT_DIR ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --bs 32 \
  --num_workers ${GPU_NUM}

# 3. Camera Tracking
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} measure_time 3 python camera_tracking/inference_batch.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --checkpoints_path checkpoints \
  --gpu_id ${CUDA_VISIBLE_DEVICES} \
  --num_workers $((GPU_NUM * 2))

# 4. eval
python eval.py \
  --csv_path ${CSV} \
  --dir_path ${OUTPUT_DIR} \
  --output_csv ${OUTPUT_DIR}/eval_results.csv
