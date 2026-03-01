# 消融实验：Baseline + 冻结策略

## 实验设计

### 1. group1_baseline（无对比学习）
- 5 种融合：concat, gated, cross_attention, bilinear, attention_pooling
- 4 种冻结策略：
  - **unfrozen**: freeze_tst1=false, freeze_tst2=false（微调时训练全部参数）
  - **freeze_both**: freeze_tst1=true, freeze_tst2=true（冻结两个 TST）
  - **freeze_tst1**: freeze_tst1=true, freeze_tst2=false
  - **freeze_tst2**: freeze_tst1=false, freeze_tst2=true
- 共 5 × 4 = **20 个实验**

### 2. group7_projection（对比学习 + 投影头）
- 7 种配置：tst1_only, tst2_only, fusion_concat, fusion_gated, fusion_cross_attention, fusion_bilinear, fusion_attention_pooling
- 4 种冻结策略（同上）
- 共 7 × 4 = **28 个实验**

**总计 48 个实验**

## 运行方式

```bash
cd /root/workplace/exp/TwoTST

# 运行全部 48 个实验
bash scripts/run_ablation_baseline_and_freeze.sh

# 或使用 Python 直接运行
python scripts/run_ablation_baseline_and_freeze.py

# 仅运行前 N 个实验（用于测试）
python scripts/run_ablation_baseline_and_freeze.py --limit 4
```

## 输出位置

- 结果目录：`/root/autodl-tmp/TwoTST/checkpoints/finetune/`
- 实验命名：`{base_name}_{freeze_mode}`
  - 例：`baseline_concat_unfrozen`, `baseline_concat_freeze_both`, `projection_fusion_concat_freeze_tst1`
- 日志：`/root/autodl-tmp/TwoTST/results/{exp_name}.log`

## 对比分析

完成后可对比：
1. **对比学习效果**：baseline_* vs projection_*（同融合、同冻结）
2. **冻结策略效果**：unfrozen vs freeze_both vs freeze_tst1 vs freeze_tst2
