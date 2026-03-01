#!/bin/bash
# 统计验证：5-fold CV + LOSO
# 数据含 19 个 site，可运行 LOSO

set -e
cd "$(dirname "$0")/.."

DATA_PATH="${DATA_PATH:-/root/workplace/exp/TwoTST/data/processed/processed_data.pkl}"
CKPT_ROOT="${CKPT_ROOT:-/root/autodl-tmp/TwoTST/checkpoints}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/TwoTST/results}"
mkdir -p "$OUTPUT_ROOT"

TST1="$CKPT_ROOT/tst1/tst1_best.pt"
TST2="$CKPT_ROOT/tst2/tst2_best.pt"

echo "Data: $DATA_PATH"
echo "Checkpoints: $TST1, $TST2"
echo "Output: $OUTPUT_ROOT"
echo ""

# 1. 5-fold CV - baseline concat (majority_vote)
echo "=========================================="
echo "[1/3] 5-fold CV: baseline_concat (majority_vote)"
echo "=========================================="
python scripts/train/train_finetune.py \
  --data_path "$DATA_PATH" \
  --tst1_checkpoint "$TST1" \
  --tst2_checkpoint "$TST2" \
  --fusion_type concat \
  --n_folds 5 \
  --eval_protocol kfold \
  --subject_agg_strategy majority_vote \
  --save_dir "$OUTPUT_ROOT/cv5_baseline_concat_majority_vote" \
  2>&1 | tee "$OUTPUT_ROOT/cv5_baseline_concat_majority_vote.log"

# 2. 5-fold CV - baseline cross_attention (majority_vote)
echo "=========================================="
echo "[2/3] 5-fold CV: baseline_cross_attention (majority_vote)"
echo "=========================================="
python scripts/train/train_finetune.py \
  --data_path "$DATA_PATH" \
  --tst1_checkpoint "$TST1" \
  --tst2_checkpoint "$TST2" \
  --fusion_type cross_attention \
  --n_folds 5 \
  --eval_protocol kfold \
  --subject_agg_strategy majority_vote \
  --save_dir "$OUTPUT_ROOT/cv5_baseline_cross_attention_majority_vote" \
  2>&1 | tee "$OUTPUT_ROOT/cv5_baseline_cross_attention_majority_vote.log"

# 3. LOSO - baseline concat (majority_vote, 19 sites = 19 folds)
echo "=========================================="
echo "[3/3] LOSO: baseline_concat (majority_vote, cross-site)"
echo "=========================================="
python scripts/train/train_finetune.py \
  --data_path "$DATA_PATH" \
  --tst1_checkpoint "$TST1" \
  --tst2_checkpoint "$TST2" \
  --fusion_type concat \
  --eval_protocol loso \
  --subject_agg_strategy majority_vote \
  --save_dir "$OUTPUT_ROOT/loso_baseline_concat_majority_vote" \
  2>&1 | tee "$OUTPUT_ROOT/loso_baseline_concat_majority_vote.log"

echo ""
echo "Done. Results in $OUTPUT_ROOT"
