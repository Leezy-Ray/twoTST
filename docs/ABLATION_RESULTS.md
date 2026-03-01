# TwoTST 消融实验结果汇总

## 一、预训练有效性验证（No Pretrain vs Baseline）

### 1.1 实验设计

| 对比项 | 有预训练 (Baseline) | 无预训练 (No Pretrain) |
|--------|---------------------|-------------------------|
| TST1/TST2 初始化 | 加载 pretrained checkpoint | 随机初始化 |
| 训练方式 | 微调 (lr=5e-5) | 端到端 (lr=1e-4) |
| 融合方式 | 5 种 | 5 种（相同） |

### 1.2 实验结果

| 融合方式 | 有预训练 AUC | 无预训练 AUC | 差值 (有−无) | 预训练有效性 |
|----------|--------------|--------------|--------------|--------------|
| concat | 0.7320 | 0.7229 | +0.0091 | ✓ 有效 |
| gated | 0.6926 | 0.6929 | −0.0003 | ≈ 持平 |
| cross_attention | 0.7252 | 0.6645 | **+0.0607** | ✓ 显著有效 |
| bilinear | 0.7230 | 0.6856 | **+0.0374** | ✓ 显著有效 |
| attention_pooling | 0.7131 | 0.7245 | −0.0114 | ✗ 无提升 |

### 1.3 结论

- **cross_attention** 和 **bilinear** 在预训练下提升最大（+6.07%、+3.74%）。
- **concat** 有预训练略好（+0.91%）。
- **gated** 和 **attention_pooling** 在有无预训练下表现接近。

---

## 二、对比学习与冻结策略（48 实验）

### 2.1 Top 10 实验（按 Test AUC）

| 排名 | 实验名称 | Test AUC | 融合 | 冻结 | 对比学习 |
|------|----------|----------|------|------|----------|
| 1 | projection_fusion_attention_pooling_unfrozen | **0.7440** | attention_pooling | unfrozen | ✓ |
| 2 | projection_fusion_gated_unfrozen | 0.7240 | gated | unfrozen | ✓ |
| 3 | baseline_concat_unfrozen | 0.7320 | concat | unfrozen | ✗ |
| 4 | baseline_cross_attention_unfrozen | 0.7252 | cross_attention | unfrozen | ✗ |
| 5 | projection_fusion_cross_attention_unfrozen | 0.7161 | cross_attention | unfrozen | ✓ |
| 6 | projection_tst2_only_freeze_tst1 | 0.7170 | tst2_only | freeze_tst1 | ✓ |
| 7 | projection_tst2_only_unfrozen | 0.7170 | tst2_only | unfrozen | ✓ |
| 8 | projection_fusion_gated_freeze_tst1 | 0.6962 | gated | freeze_tst1 | ✓ |
| 9 | projection_fusion_concat_freeze_tst1 | 0.7061 | concat | freeze_tst1 | ✓ |
| 10 | baseline_concat_freeze_tst1 | 0.7159 | concat | freeze_tst1 | ✗ |

### 2.2 冻结策略效果（平均 AUC）

| 冻结策略 | Baseline (n=5) | Projection (n=7) |
|----------|----------------|------------------|
| unfrozen | 0.7172 | 0.6820 |
| freeze_tst1 | 0.6943 | 0.6740 |
| freeze_tst2 | 0.6459 | 0.6180 |
| freeze_both | 0.6240 | 0.5886 |

**规律**：unfrozen > freeze_tst1 > freeze_tst2 > freeze_both

---

## 三、推荐配置

| 场景 | 推荐配置 | Test AUC |
|------|----------|----------|
| 最佳 | projection_fusion_attention_pooling_unfrozen | 0.744 |
| 备选 | baseline_concat_unfrozen | 0.732 |
| 备选 | projection_fusion_gated_unfrozen | 0.724 |

---

## 四、数据文件

| 文件 | 说明 |
|------|------|
| `/root/autodl-tmp/TwoTST/results/ablation_summary.json` | 48 实验汇总 |
| `/root/autodl-tmp/TwoTST/results/ablation_summary.csv` | 48 实验 CSV |
| `/root/autodl-tmp/TwoTST/results/ablation_no_pretrain.log` | 预训练消融日志 |
| `/root/autodl-tmp/TwoTST/checkpoints/finetune/` | 各实验 results.json |
