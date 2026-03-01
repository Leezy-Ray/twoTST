#!/bin/bash
# 验证实验：预训练阶段滑动窗口 vs 非滑动窗口，对非滑窗微调阶段的影响
# 对比：
#   A) 非滑窗微调 + 滑窗预训练 TST (nosw_finetune_sw_pretrain)
#   B) 非滑窗微调 + 非滑窗预训练 TST (group7_projection_fusion_attention_pooling)
# 目的：验证哪种预训练策略对非滑窗微调提升更大

set -e

cd /root/workplace/exp/TwoTST

LOG_DIR="${LOG_DIR:-/root/workplace/exp/TwoTST/logs/pretrain_ablation_nosw_finetune}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Pretrain Ablation: SW vs Non-SW for Non-Sliding-Window Finetune"
echo "========================================"
echo ""

# B) 非滑窗微调 + 非滑窗预训练 (对照组)
echo "=== B) Non-SW Finetune + Non-SW Pretrain (baseline) ==="
python scripts/experiments/run_experiment.py \
  --config configs/experiments/group7_projection_fusion_attention_pooling.yaml \
  2>&1 | tee "$LOG_DIR/nosw_finetune_nosw_pretrain.log"
echo ""

# A) 非滑窗微调 + 滑窗预训练 (实验组)
echo "=== A) Non-SW Finetune + SW Pretrain (experiment) ==="
python scripts/experiments/run_experiment.py \
  --config configs/experiments/nosw_finetune_sw_pretrain_attention_pooling.yaml \
  2>&1 | tee "$LOG_DIR/nosw_finetune_sw_pretrain.log"
echo ""

echo "========================================"
echo "Done. Compare AUC in logs:"
echo "  - $LOG_DIR/nosw_finetune_nosw_pretrain.log"
echo "  - $LOG_DIR/nosw_finetune_sw_pretrain.log"
echo "========================================"
