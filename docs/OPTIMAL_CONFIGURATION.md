# TwoTST 最优配置汇总

基于全部实验结果的综合分析，得出最优配置与备选方案。

---

## 最佳训练策略（汇总）

| 维度 | 最优选择 |
|------|----------|
| **滑动窗口** | ❌ 无（非滑窗） |
| **预训练** | 非滑窗预训练 TST1、TST2 |
| **对比学习** | ✅ 启用，freeze TST1、unfreeze TST2 |
| **融合方式** | attention_pooling |
| **微调阶段** | TST1、TST2 均 unfrozen，Projection 冻结 |
| **Test AUC** | **~0.744**（单次） |

---

## 一、实验结果总览

### 1.1 非滑窗实验（按 Test AUC 排序）

| 排名 | 实验 | Test AUC | 预训练 | 融合 | 对比学习 |
|------|------|----------|--------|------|----------|
| **1** | projection_fusion_attention_pooling | **0.7440** | 非滑窗 | attention_pooling | ✓ |
| 2 | baseline_concat | 0.7320 | 非滑窗 | concat | ✗ |
| 3 | nosw_finetune_sw_pretrain | 0.7304 | 滑窗 | attention_pooling | ✓ |
| 4 | baseline_cross_attention | 0.7252 | 非滑窗 | cross_attention | ✗ |
| 5 | projection_fusion_gated | 0.7240 | 非滑窗 | gated | ✓ |
| 6 | projection_fusion_cross_attention | 0.7161 | 非滑窗 | cross_attention | ✓ |

### 1.2 滑窗实验

| 实验 | Test AUC | 融合 | 备注 |
|------|----------|------|------|
| sw_projection_fusion_gated_multi | 0.7111 | gated_multi | 滑窗最优 |
| sw_projection_fusion_attention_pooling | 0.7032 | attention_pooling | |
| 其他 sw_projection_* | 约 0.65–0.70 | 多种 | |

### 1.3 统计验证（5-fold CV / LOSO）

| 实验 | Accuracy | AUC | 评估方式 |
|------|----------|-----|----------|
| baseline_concat | 0.6262 ± 0.049 | 0.6937 ± 0.038 | 5-fold CV |
| baseline_cross_attention | 0.6261 ± 0.030 | 0.6724 ± 0.027 | 5-fold CV |
| baseline_concat | 0.6577 ± 0.131 | **0.7090 ± 0.133** | LOSO 跨站点 |

### 1.4 预训练策略消融（非滑窗微调）

| 预训练方式 | Test AUC | 结论 |
|------------|----------|------|
| **非滑窗预训练** | **0.7440** | 更优 |
| 滑窗预训练 | 0.7304 | 较差 |

---

## 二、最优配置

### 2.1 推荐配置（全局最优）

**配置文件**：`configs/experiments/group7_projection_fusion_attention_pooling.yaml`

| 维度 | 配置 |
|------|------|
| **数据** | 非滑窗 `data/processed/processed_data.pkl` |
| **预训练** | 非滑窗 TST（`checkpoints/tst1`, `checkpoints/tst2`） |
| **对比学习** | 启用，加载 `checkpoints/contrastive_checkpoint.pt` |
| **融合** | **attention_pooling** |
| **微调** | unfrozen（TST1、TST2 均微调），use_projection=true |
| **Test AUC** | **0.7440** |

### 2.2 备选配置

| 场景 | 配置 | Test AUC | 说明 |
|------|------|----------|------|
| 无对比学习 | `group1_baseline_concat` | 0.7320 | 简单拼接，训练更快 |
| 平衡 | `group7_projection_fusion_gated` | 0.7240 | gated 融合，较稳定 |
| 滑窗场景 | `sw_projection_fusion_gated_multi` | 0.7111 | 滑窗数据下最优 |
| 跨站点泛化 | LOSO baseline_concat | 0.7090 | 需用 LOSO 脚本 |

---

## 三、配置要点总结

1. **预训练与微调匹配**：非滑窗微调应使用非滑窗预训练，滑窗微调用滑窗预训练。
2. **融合策略**：非滑窗下 attention_pooling 最优，滑窗下 gated_multi 最优。
3. **对比学习**：在 attention_pooling / gated 下带来明显提升，concat / bilinear 下提升有限。
4. **冻结策略**：unfrozen > freeze_tst1 > freeze_tst2 > freeze_both。

---

## 四、一键运行

```bash
cd /root/workplace/exp/TwoTST
python scripts/experiments/run_experiment.py \
  --config configs/experiments/group7_projection_fusion_attention_pooling.yaml
```
