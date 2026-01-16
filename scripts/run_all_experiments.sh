#!/bin/bash
# 批量运行所有融合实验
# 用法: bash scripts/run_all_experiments.sh [group]
# group: all, baseline, contrastive, finetune, ablation, combined

set -e

PROJECT_ROOT="/root/workplace/exp/TwoTST"
CONFIG_DIR="$PROJECT_ROOT/configs/experiments"
RESULTS_FILE="$PROJECT_ROOT/results/experiment_summary.csv"

cd $PROJECT_ROOT

# 创建结果目录
mkdir -p results

# 初始化结果文件
echo "experiment,auc,accuracy,sensitivity,specificity,f1" > $RESULTS_FILE

GROUP=${1:-"all"}

echo "=========================================="
echo "Running TwoTST Fusion Experiments"
echo "Group: $GROUP"
echo "=========================================="

run_experiment() {
    config_file=$1
    exp_name=$(basename $config_file .yaml)
    
    echo ""
    echo "=========================================="
    echo "Running: $exp_name"
    echo "=========================================="
    
    python scripts/run_experiment.py --config $config_file
    
    # 提取结果并追加到汇总文件
    result_file="$PROJECT_ROOT/checkpoints/finetune/${exp_name#group*_}/results.json"
    if [ -f "$result_file" ]; then
        python -c "
import json
with open('$result_file') as f:
    r = json.load(f)
print(f\"$exp_name,{r['auc']:.4f},{r['accuracy']:.4f},{r['sensitivity']:.4f},{r['specificity']:.4f},{r['f1']:.4f}\")
" >> $RESULTS_FILE
    fi
}

# 实验组1: 基线实验
if [ "$GROUP" == "all" ] || [ "$GROUP" == "baseline" ]; then
    echo ""
    echo ">>> Group 1: Baseline Experiments (Different Fusion Methods)"
    run_experiment "$CONFIG_DIR/group1_baseline_concat.yaml"
    run_experiment "$CONFIG_DIR/group1_baseline_gated.yaml"
    run_experiment "$CONFIG_DIR/group1_baseline_cross_attention.yaml"
    run_experiment "$CONFIG_DIR/group1_baseline_bilinear.yaml"
    run_experiment "$CONFIG_DIR/group1_baseline_attention_pooling.yaml"
fi

# 实验组2: 对比学习实验
if [ "$GROUP" == "all" ] || [ "$GROUP" == "contrastive" ]; then
    echo ""
    echo ">>> Group 2: Contrastive Learning Experiments"
    run_experiment "$CONFIG_DIR/group2_contrastive_both_train.yaml"
    run_experiment "$CONFIG_DIR/group2_contrastive_freeze_tst1.yaml"
    run_experiment "$CONFIG_DIR/group2_contrastive_freeze_tst2.yaml"
    run_experiment "$CONFIG_DIR/group2_contrastive_freeze_both.yaml"
fi

# 实验组3: 微调冻结实验
if [ "$GROUP" == "all" ] || [ "$GROUP" == "finetune" ]; then
    echo ""
    echo ">>> Group 3: Finetune Freeze Experiments"
    run_experiment "$CONFIG_DIR/group3_finetune_freeze_tst1.yaml"
    run_experiment "$CONFIG_DIR/group3_finetune_freeze_tst2.yaml"
    run_experiment "$CONFIG_DIR/group3_finetune_freeze_both.yaml"
fi

# 实验组4: 消融实验
if [ "$GROUP" == "all" ] || [ "$GROUP" == "ablation" ]; then
    echo ""
    echo ">>> Group 4: Ablation Experiments (Single Stream)"
    run_experiment "$CONFIG_DIR/group4_single_tst1.yaml"
    run_experiment "$CONFIG_DIR/group4_single_tst2.yaml"
fi

# 实验组6: 组合实验
if [ "$GROUP" == "all" ] || [ "$GROUP" == "combined" ]; then
    echo ""
    echo ">>> Group 6: Combined Experiments"
    run_experiment "$CONFIG_DIR/group6_contrastive_finetune_freeze_both.yaml"
    run_experiment "$CONFIG_DIR/group6_contrastive_only_fc_finetune.yaml"
fi

echo ""
echo "=========================================="
echo "All experiments completed!"
echo "Results summary: $RESULTS_FILE"
echo "=========================================="

# 显示结果汇总
echo ""
echo "Experiment Results Summary:"
echo ""
cat $RESULTS_FILE | column -t -s','
