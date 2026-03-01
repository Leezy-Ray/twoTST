#!/bin/bash
# 消融实验：baseline（无对比学习）+ 冻结策略对比
# 1. group1_baseline_* (5种融合) x 4种冻结 = 20 实验
# 2. group7_projection_* (7种) x 4种冻结 = 28 实验
# 总计 48 实验，结果保存到 /root/autodl-tmp/TwoTST

set -e

cd "$(dirname "$0")/.."
python scripts/ablation/run_ablation_baseline_and_freeze.py 2>&1 | tee /root/autodl-tmp/TwoTST/results/ablation_baseline_freeze.log
