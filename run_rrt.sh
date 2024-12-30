#!/bin/bash
# 固定参数
MODEL_NAME="RRT"
IN_DIM=512
DATA_DIR="/home/perry/nvme3n1/TCGA-Brain/All_Features/CONCH/All_Features/"
CSV_DIR="csv/task2/"
LOG_DIR="task2_logs"
# 循环执行 5 次，n_fold 从 0 到 4
for N_FOLD in {0..4}
do
    echo "Running train.py with n_fold = $N_FOLD"
    python train.py \
        --model_name "$MODEL_NAME" \
        --in_dim "$IN_DIM" \
        --n_fold "$N_FOLD" \
        --data_dir "$DATA_DIR" \
        --csv_dir "$CSV_DIR" \
        --log_dir "$LOG_DIR"
done

echo "All folds completed!"