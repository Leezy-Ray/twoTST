#!/bin/bash
# 消融实验：验证 TST1/TST2 预训练是否有效
# 5 种融合 x 无预训练 = 5 个实验

set -e

cd "$(dirname "$0")/.."
python scripts/ablation/run_ablation_no_pretrain.py 2>&1 | tee /root/autodl-tmp/TwoTST/results/ablation_no_pretrain.log
