#!/bin/bash
# 运行所有投影头微调实验

set -e

cd /root/workplace/exp/TwoTST

LOG_DIR="/root/autodl-tmp/TwoTST/results"
mkdir -p $LOG_DIR

echo "========================================"
echo "Running Projection Finetuning Experiments"
echo "========================================"
echo ""

# 非滑动窗口实验 (7个)
echo "=== Non-Sliding Window Experiments ==="
configs_normal=(
    "configs/experiments/group7_projection_tst1_only.yaml"
    "configs/experiments/group7_projection_tst2_only.yaml"
    "configs/experiments/group7_projection_fusion_concat.yaml"
    "configs/experiments/group7_projection_fusion_gated.yaml"
    "configs/experiments/group7_projection_fusion_cross_attention.yaml"
    "configs/experiments/group7_projection_fusion_bilinear.yaml"
    "configs/experiments/group7_projection_fusion_attention_pooling.yaml"
)

for config in "${configs_normal[@]}"; do
    name=$(basename $config .yaml)
    echo ""
    echo ">>> Running: $name"
    python scripts/experiments/run_experiment.py --config "$config" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo ">>> Completed: $name"
    echo "---"
done

# 滑动窗口实验 (7个)
echo ""
echo "=== Sliding Window Experiments ==="
configs_sw=(
    "configs/experiments_sw/sw_projection_tst1_only.yaml"
    "configs/experiments_sw/sw_projection_tst2_only.yaml"
    "configs/experiments_sw/sw_projection_fusion_concat.yaml"
    "configs/experiments_sw/sw_projection_fusion_gated.yaml"
    "configs/experiments_sw/sw_projection_fusion_cross_attention.yaml"
    "configs/experiments_sw/sw_projection_fusion_bilinear.yaml"
    "configs/experiments_sw/sw_projection_fusion_attention_pooling.yaml"
)

for config in "${configs_sw[@]}"; do
    name=$(basename $config .yaml)
    echo ""
    echo ">>> Running: $name"
    python scripts/experiments/run_experiment.py --config "$config" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo ">>> Completed: $name"
    echo "---"
done

echo ""
echo "========================================"
echo "All 14 experiments completed!"
echo "Results saved to: $LOG_DIR"
echo "========================================"
