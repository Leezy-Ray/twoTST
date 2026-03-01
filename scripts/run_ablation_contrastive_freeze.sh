#!/bin/bash
# 对比学习 freeze 消融：attention_pooling + 非滑窗
# 验证对比学习阶段 freeze TST 对微调阶段融合的影响

set -e

cd "$(dirname "$0")/.."

echo "=========================================="
echo "Ablation: Contrastive Freeze Strategy"
echo "attention_pooling + non-sliding-window"
echo "=========================================="

python scripts/ablation/run_ablation_contrastive_freeze.py \
  2>&1 | tee logs/ablation_contrastive_freeze/ablation_contrastive_freeze.log

echo ""
echo "Done. Results in checkpoints/finetune/projection_fusion_attention_pooling_cont_*"
