#!/bin/bash
# 仅运行滑动窗口投影头微调实验（7个），采用 prob_mean 受试者级汇总策略

set -e

cd /root/workplace/exp/TwoTST

LOG_DIR="/root/autodl-tmp/TwoTST/results"
mkdir -p $LOG_DIR

echo "========================================"
echo "Sliding Window Projection Experiments (prob_mean)"
echo "========================================"
echo ""

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
    echo ">>> Running: $name (prob_mean)"
    python scripts/experiments/run_experiment.py --config "$config" --subject_agg_strategy prob_mean \
        2>&1 | tee "$LOG_DIR/${name}_prob_mean.log"
    echo ">>> Completed: $name"
    echo "---"
done

echo ""
echo "========================================"
echo "All 7 sliding window experiments completed (prob_mean)!"
echo "Results saved to: $LOG_DIR"
echo "========================================"
