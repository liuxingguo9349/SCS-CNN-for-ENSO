#!/bin/bash
# This script runs the full pre-training and fine-tuning pipeline for a given configuration.

# --- Configuration ---
LEAD_TIME=$1
TARGET_MONTH=$2
NUM_ENSEMBLE=10
BASE_EXP_DIR="experiments"
DEVICE="cuda:0" # Specify the GPU you want to use

# Check if arguments are provided
if [ -z "$LEAD_TIME" ] || [ -z "$TARGET_MONTH" ]; then
    echo "Usage: $0 <lead_time> <target_month>"
    echo "Example: ./scripts/run_experiment.sh 9 1"
    exit 1
fi

echo "=========================================================="
echo "Starting Experiment for Lead: $LEAD_TIME, Target: $TARGET_MONTH"
echo "=========================================================="

# Loop for ensemble members
for i in $(seq 1 $NUM_ENSEMBLE)
do
    echo -e "\n--- Running Ensemble Member $i / $NUM_ENSEMBLE ---"

    # --- Stage 1: Pre-training ---
    echo "[Stage 1/2] Pre-training on CMIP5..."
    PRETRAIN_CONFIG="configs/pretrain_base.yaml"
    
    python main.py \
        --config $PRETRAIN_CONFIG \
        --lead $LEAD_TIME \
        --target $TARGET_MONTH \
        --ens $i \
        --device $DEVICE

    # Path to the best pre-trained model
    PRETRAIN_RUN_NAME="pretrain_lead${LEAD_TIME}_target${TARGET_MONTH}_ens${i}"
    PRETRAINED_MODEL_PATH="${BASE_EXP_DIR}/${PRETRAIN_RUN_NAME}/best_model.pth"

    if [ ! -f "$PRETRAINED_MODEL_PATH" ]; then
        echo "Error: Pre-training failed for ensemble member $i. Checkpoint not found at ${PRETRAINED_MODEL_PATH}."
        continue # Skip to the next ensemble member
    fi

    # --- Stage 2: Fine-tuning ---
    echo "[Stage 2/2] Fine-tuning on SODA..."
    FINETUNE_CONFIG="configs/finetune_base.yaml"
    
    python main.py \
        --config $FINETUNE_CONFIG \
        --lead $LEAD_TIME \
        --target $TARGET_MONTH \
        --ens $i \
        --device $DEVICE \
        --pretrain_checkpoint_path $PRETRAINED_MODEL_PATH

done

echo -e "\n=========================================================="
echo "Experiment for Lead: $LEAD_TIME, Target: $TARGET_MONTH finished."
echo "=========================================================="
