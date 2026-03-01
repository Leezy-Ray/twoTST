#!/bin/bash
# 仅运行滑动窗口投影头微调实验（7个），采用 majority_vote 受试者级汇总策略
# 对应 run_full_training_autodl.sh 的步骤 6 中滑动窗口部分

set -e

cd /root/workplace/exp/TwoTST

LOG_DIR="/root/autodl-tmp/TwoTST/results"
mkdir -p $LOG_DIR

echo "========================================"
echo "Sliding Window Projection Experiments (majority_vote)"
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
    echo ">>> Running: $name (majority_vote)"
    python scripts/experiments/run_experiment.py --config "$config" --subject_agg_strategy majority_vote \
        2>&1 | tee "$LOG_DIR/${name}_majority_vote.log"
    echo ">>> Completed: $name"
    echo "---"
done

echo ""
echo "========================================"
echo "All 7 sliding window experiments completed (majority_vote)!"
echo "Results saved to: $LOG_DIR"
echo "========================================"
