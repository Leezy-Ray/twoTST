#!/bin/bash
# 最佳配置（pretrain TST1/TST2 → projection → attention_pooling）的 5-fold 与 LOSO
# 用于审稿回复：统计验证 + 跨站点泛化

set -e
cd "$(dirname "$0")/.."

CONFIG="${1:-configs/experiments/group7_projection_fusion_attention_pooling.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/TwoTST/results}"
mkdir -p "$OUTPUT_ROOT"

echo "=============================================="
echo "Best config: projection + attention_pooling"
echo "Config: $CONFIG"
echo "Output: $OUTPUT_ROOT"
echo "=============================================="

# 1. 5-fold CV
echo ""
echo "[1/2] 5-fold CV"
python scripts/run_best_config_5fold_loso.py \
  --config "$CONFIG" \
  --eval_protocol kfold \
  --n_folds 5 \
  --subject_agg_strategy majority_vote \
  --save_dir "$OUTPUT_ROOT/best_config_5fold" \
  2>&1 | tee "$OUTPUT_ROOT/best_config_5fold.log"

# 2. LOSO（需数据含 site_ids，否则会报错 LOSO requires site_ids）
echo ""
echo "[2/2] LOSO (Leave-One-Site-Out)"
python scripts/run_best_config_5fold_loso.py \
  --config "$CONFIG" \
  --eval_protocol loso \
  --subject_agg_strategy majority_vote \
  --save_dir "$OUTPUT_ROOT/best_config_loso" \
  2>&1 | tee "$OUTPUT_ROOT/best_config_loso.log" || echo "LOSO failed (e.g. no site_ids in data). See log above."

echo ""
echo "Done. 5-fold summary: $OUTPUT_ROOT/best_config_5fold/summary.json"
echo "      LOSO summary:   $OUTPUT_ROOT/best_config_loso/summary.json"
