#!/bin/bash
# 完整训练流程，结果保存到 /root/autodl-tmp/TwoTST

set -e

PROJECT_ROOT="/root/workplace/exp/TwoTST"
OUTPUT_ROOT="/root/autodl-tmp/TwoTST"
DATA_ROOT="$PROJECT_ROOT/data"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT"/{checkpoints,checkpoints_sw,logs,logs_sw,results}
mkdir -p "$OUTPUT_ROOT"/checkpoints/{tst1,tst2,finetune}
mkdir -p "$OUTPUT_ROOT"/checkpoints_sw/{tst1,tst2,finetune}

echo "=========================================="
echo "TwoTST Full Training Pipeline"
echo "Output: $OUTPUT_ROOT"
echo "=========================================="

# 1. 数据准备（确保含 subject_indices, site_ids）
echo ""
echo ">>> [1/6] Preparing data..."
python scripts/data/prepare_data.py \
  --data_path "$DATA_ROOT/fmri.npy" \
  --output_dir "$DATA_ROOT/processed" 2>&1 | tail -5

python scripts/data/prepare_data.py \
  --data_path "$DATA_ROOT/fmri.npy" \
  --output_dir "$DATA_ROOT/processed_sw" \
  --use_sliding_window --window_size 32 --stride 16 2>&1 | tail -5

# 2. 预训练（无滑窗）
echo ""
echo ">>> [2/6] Pretraining TST1+TST2 (no sliding window)..."
python scripts/train/train_pretrain.py \
  --data_path "$DATA_ROOT/processed/processed_data.pkl" \
  --save_dir "$OUTPUT_ROOT/checkpoints" \
  --log_dir "$OUTPUT_ROOT/logs" \
  --tst1_epochs 100 --tst2_epochs 100 2>&1 | tee "$OUTPUT_ROOT/results/pretrain.log" | tail -20

# 3. 预训练（滑窗）
echo ""
echo ">>> [3/6] Pretraining TST1+TST2 (sliding window)..."
python scripts/train/train_pretrain.py \
  --data_path "$DATA_ROOT/processed_sw/processed_data.pkl" \
  --save_dir "$OUTPUT_ROOT/checkpoints_sw" \
  --log_dir "$OUTPUT_ROOT/logs_sw" \
  --tst1_epochs 100 --tst2_epochs 100 2>&1 | tee "$OUTPUT_ROOT/results/pretrain_sw.log" | tail -20

# 4. 生成对比学习 checkpoint
echo ""
echo ">>> [4/6] Generating contrastive checkpoints..."
python scripts/train/generate_contrastive_checkpoints.py \
  --normal_data "$DATA_ROOT/processed/processed_data.pkl" \
  --sw_data "$DATA_ROOT/processed_sw/processed_data.pkl" \
  --normal_tst1 "$OUTPUT_ROOT/checkpoints/tst1/tst1_best.pt" \
  --normal_tst2 "$OUTPUT_ROOT/checkpoints/tst2/tst2_best.pt" \
  --sw_tst1 "$OUTPUT_ROOT/checkpoints_sw/tst1/tst1_best.pt" \
  --sw_tst2 "$OUTPUT_ROOT/checkpoints_sw/tst2/tst2_best.pt" \
  --normal_save "$OUTPUT_ROOT/checkpoints/contrastive_checkpoint.pt" \
  --sw_save "$OUTPUT_ROOT/checkpoints_sw/contrastive_checkpoint.pt" 2>&1 | tee "$OUTPUT_ROOT/results/contrastive.log"

# 5. 更新实验配置中的预训练路径为 autodl-tmp
echo ""
echo ">>> [5/6] Updating config paths..."
for f in configs/experiments/group7_*.yaml configs/experiments_sw/sw_projection_*.yaml; do
  [ -f "$f" ] && sed -i 's|/root/workplace/exp/TwoTST/checkpoints|/root/autodl-tmp/TwoTST/checkpoints|g' "$f"
  [ -f "$f" ] && sed -i 's|/root/workplace/exp/TwoTST/checkpoints_sw|/root/autodl-tmp/TwoTST/checkpoints_sw|g' "$f"
done

# 6. 运行投影头微调实验
echo ""
echo ">>> [6/6] Running projection finetuning experiments..."
bash scripts/run_projection_experiments.sh 2>&1 | tee "$OUTPUT_ROOT/results/all_experiments.log"

echo ""
echo "=========================================="
echo "Training completed! Results in $OUTPUT_ROOT"
echo "=========================================="
