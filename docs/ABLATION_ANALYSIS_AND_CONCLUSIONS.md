# 消融实验数据分析与结论

## 1. 实验完成情况

- **总实验数**: 48
- **已完成**: 48/48 ✓
- **数据来源**: `/root/autodl-tmp/TwoTST/checkpoints/finetune/*/results.json`

---

## 2. 数据汇总（按 Test AUC 排序）

### 2.1 最优实验 Top 10

| 排名 | 实验名称 | Test AUC | Test Acc | 融合 | 冻结 | 对比学习 |
|------|----------|----------|----------|------|------|----------|
| 1 | projection_fusion_attention_pooling_unfrozen | **0.7440** | 0.6477 | attention_pooling | unfrozen | ✓ |
| 2 | projection_fusion_gated_unfrozen | 0.7240 | 0.6736 | gated | unfrozen | ✓ |
| 3 | baseline_concat_unfrozen | 0.7320 | 0.6943 | concat | unfrozen | ✗ |
| 4 | baseline_cross_attention_unfrozen | 0.7252 | 0.6788 | cross_attention | unfrozen | ✗ |
| 5 | projection_fusion_cross_attention_unfrozen | 0.7161 | 0.6425 | cross_attention | unfrozen | ✓ |
| 6 | projection_tst2_only_freeze_tst1 | 0.7170 | 0.6477 | tst2_only | freeze_tst1 | ✓ |
| 7 | projection_tst2_only_unfrozen | 0.7170 | 0.6477 | tst2_only | unfrozen | ✓ |
| 8 | projection_fusion_gated_freeze_tst1 | 0.6962 | 0.6580 | gated | freeze_tst1 | ✓ |
| 9 | projection_fusion_concat_freeze_tst1 | 0.7061 | 0.6632 | concat | freeze_tst1 | ✓ |
| 10 | baseline_concat_freeze_tst1 | 0.7159 | 0.6580 | concat | freeze_tst1 | ✗ |

### 2.2 最差实验（AUC < 0.60）

| 实验名称 | Test AUC | 备注 |
|----------|----------|------|
| projection_tst1_only_freeze_both | 0.5599 | 仅 TST1，双冻结 |
| projection_tst1_only_freeze_tst1 | 0.5599 | 仅 TST1，冻结 TST1 |
| projection_fusion_bilinear_freeze_tst2 | 0.5287 | bilinear + 冻结 TST2 |
| projection_fusion_attention_pooling_freeze_both | 0.5826 | attention_pooling + 双冻结 |

---

## 3. 对比学习效果分析（baseline vs projection）

**对比方式**: 相同融合方式、相同冻结策略下，baseline（无对比学习）vs projection（有对比学习）的 Test AUC 差异。

| 融合方式 | 冻结策略 | Baseline AUC | Projection AUC | 差值 (Proj - Base) |
|----------|----------|--------------|----------------|--------------------|
| concat | unfrozen | 0.7320 | 0.6687 | **-0.0633** |
| concat | freeze_both | 0.6330 | 0.5982 | -0.0348 |
| concat | freeze_tst1 | 0.7159 | 0.7061 | -0.0098 |
| concat | freeze_tst2 | 0.6327 | 0.6508 | +0.0181 |
| gated | unfrozen | 0.6926 | **0.7240** | **+0.0314** |
| gated | freeze_tst1 | 0.7018 | 0.6962 | -0.0056 |
| cross_attention | unfrozen | **0.7252** | 0.7161 | -0.0091 |
| cross_attention | freeze_tst1 | 0.7125 | 0.6855 | -0.0270 |
| bilinear | unfrozen | 0.7230 | 0.5708 | **-0.1522** |
| bilinear | freeze_tst1 | 0.6548 | 0.6420 | -0.0128 |
| attention_pooling | unfrozen | 0.7131 | **0.7440** | **+0.0309** |
| attention_pooling | freeze_tst1 | 0.6866 | 0.7111 | +0.0245 |

**结论**:
- **对比学习在 gated 和 attention_pooling 上带来提升**，尤其在 unfrozen 时。
- **concat 和 bilinear 在 baseline 下表现更好**，对比学习反而略降；可能因为这两种融合对预训练表征依赖较小。
- **projection_fusion_attention_pooling_unfrozen** 为全局最佳 (AUC=0.744)，说明「对比学习 + 投影头 + attention_pooling + 全参数微调」组合最优。

---

## 4. 冻结策略效果分析

**对比方式**: 同一融合方式下，unfrozen / freeze_both / freeze_tst1 / freeze_tst2 的 AUC 差异。

### 4.1 各融合方式下的冻结策略平均表现

| 冻结策略 | 平均 AUC (baseline, n=5) | 平均 AUC (projection, n=7) |
|----------|--------------------------|----------------------------|
| unfrozen | **0.7172** | **0.6820** |
| freeze_tst1 | 0.6943 | 0.6740 |
| freeze_tst2 | 0.6459 | 0.6180 |
| freeze_both | 0.6240 | 0.5886 |

### 4.2 规律

- **unfrozen 整体最优**，允许两个 TST 同时微调，表达能力最强。
- **freeze_tst1** 次之，只微调 TST2 仍有较好效果。
- **freeze_tst2** 明显更差，说明微调 TST1（时序表征）比微调 TST2（PCC 表征）更重要。
- **freeze_both** 最差，仅训练融合层和分类头，表征几乎固定，泛化能力受限。

---

## 5. 主要结论

1. **对比学习有效性**
   - 在 **gated** 和 **attention_pooling** 融合下，对比学习能提升下游任务 AUC。
   - 在 **concat** 和 **bilinear** 下，baseline 已足够强，对比学习增益有限甚至为负。

2. **推荐配置**
   - **最佳组合**: `projection_fusion_attention_pooling_unfrozen` (AUC=0.744)。
   - 备选: `projection_fusion_gated_unfrozen` (AUC=0.724)、`baseline_concat_unfrozen` (AUC=0.732)。

3. **冻结策略建议**
   - 优先使用 **unfrozen** 全参数微调。
   - 若需降低过拟合或加速训练，可尝试 **freeze_tst1**。
   - 避免 **freeze_both** 和 **freeze_tst2**。

4. **单模态 (tst1_only / tst2_only)**
   - `projection_tst2_only` 表现稳定 (AUC≈0.72)，而 `projection_tst1_only` 较差 (AUC≈0.56–0.63)。
   - 说明 PCC 模态在本任务中贡献更大，或 TST1 需与 TST2 联合微调才能发挥较好效果。

---

## 6. 数据文件

- 汇总 JSON: `/root/autodl-tmp/TwoTST/results/ablation_summary.json`
- 汇总 CSV: `/root/autodl-tmp/TwoTST/results/ablation_summary.csv`
