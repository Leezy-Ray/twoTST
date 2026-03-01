# 审稿意见逐条回复：实验现状与结论

基于当前代码实现与已完成的消融实验（48 baseline/freeze + 5 no_pretrain），对各审稿意见逐点说明：**是否已解决、如何解决、实验结论**。

---

## 一、【致命问题 1】Subject-level 数据泄漏

### 审稿人关注点
- sliding window 导致同被试多样本
- train/val/test 是否严格按 subject-level 划分
- "All windows from a subject confined to a single subset"

### 实验现状与结论

| 项目 | 状态 | 说明 |
|------|------|------|
| **先 subject-level split** | ✅ 已实现 | `utils/splitters.py` 的 `get_subject_level_train_val_test_split` 在划分时以 **subject** 为单位，先用 `unique_subjects` 划分，再映射到样本索引 |
| **再在各自 split 内做滑窗** | ✅ 已满足 | 若使用滑窗数据，`subject_indices` 将同一被试的多个 window 归为一组，划分时同一被试的所有样本只会出现在 train / val / test 之一 |
| **All windows confined to single subset** | ✅ 已满足 | `train_idx/val_idx/test_idx` 基于 `np.isin(subject_indices, train_subj)` 等得到，同一 subject 的所有样本同属一个 subset |

### 实验证据

日志输出示例（`ablation_baseline_freeze.log`）：
```
Subject-level split - Train: 674, Val: 96, Test: 193
```

- 划分依据为 subject，不是 sample
- 若存在 subject_indices，则所有 48+5 消融实验均使用上述 subject-level 划分

### 结论

✅ **无 subject-level 数据泄漏**：同一被试的所有窗口仅出现在 train/val/test 之一，代码与日志均可支撑该结论。

### 建议 Methods 表述（英文）

> *Data splitting and sliding-window augmentation.* To prevent subject-level information leakage, we performed all train/validation/test splits strictly at the subject level. We first split unique subjects into training (70%), validation (10%), and test (20%) using stratified sampling. When sliding-window augmentation was used, all windows derived from a given subject were confined to the same subset (train, validation, or test). This ensures that no temporal segment from a test subject was ever seen during training or validation.

---

## 二、【致命问题 2】统计验证与单次划分

### 审稿人关注点
- 仅有 point estimate，无标准差 / 置信区间 / 显著性检验
- 仅 single train/val/test split

### 实验现状与结论

| 项目 | 代码能力 | 消融实验实际使用 |
|------|----------|------------------|
| **K-fold CV** | ✅ train_finetune.py 支持 | ❌ 消融使用 run_experiment，**仅单次划分** |
| **Bootstrap 95% CI** | ✅ utils/metrics.bootstrap_confidence_interval | ❌ 消融 **未使用** |
| **mean ± std** | ✅ train_finetune 多折时输出 | ❌ 消融 **仅单次结果** |
| **显著性检验** | ❌ 未实现 | ❌ |

### 结论

⚠️ **当前消融实验缺乏统计验证**：48+5 个实验均为单次 subject-level 划分、单次训练、无标准差和置信区间。审稿人可直接质疑结果稳定性。

### 建议补做（最低生存线）

1. **多 seed 或 5-fold**：对至少 1–2 个主实验（如 `baseline_concat_unfrozen`, `projection_fusion_attention_pooling_unfrozen`）用 `train_finetune.py --n_folds 5` 或 3 个 seed 重跑，报告 **mean ± std** 及 **Bootstrap 95% CI**。
2. **论文中明确说明**：在 Results 中写明“主结果基于单次划分，补充实验采用 5-fold / 多 seed 以报告不确定性”。

---

## 三、【高风险问题 3】ABIDE Site Effect

### 审稿人关注点
- multi-site、scanner/protocol variability
- 性能虚高风险

### 实验现状与结论

| 项目 | 代码能力 | 消融实验实际使用 |
|------|----------|------------------|
| **site_ids 保存** | ✅ prepare_data 写入 processed_data.pkl | ✅ 数据含 site 时已保存 |
| **site-stratified split** | ⚠️ get_subject_level_train_val_test_split **未使用** site_ids | ❌ 单次划分**未按 site 分层** |
| **LOSO (Leave-One-Site-Out)** | ✅ utils/splitters.get_loso_fold_splits, train_finetune --eval_protocol loso | ❌ 消融 **未使用** |

### 结论

⚠️ **当前消融实验未控制 site**：划分未做 site 分层，也未运行 LOSO。若数据含多站点，结果可能受 site 偏差影响。

### 建议

1. **补做 LOSO**：`train_finetune.py --eval_protocol loso`，报告跨站点泛化性能。
2. **或明确承认 limitation**：在 Discussion 中写明 “Evaluation was conducted on ABIDE I with a single split; site effects and cross-site generalization were not explicitly assessed and remain a limitation.”

---

## 四、【高风险问题 4】36.8% Improvement 表述

### 审稿人关注点
- absolute vs relative
- AUC vs ACC
- augmentation vs architecture 贡献

### 实验现状与结论

| 项目 | 状态 | 说明 |
|------|------|------|
| **数学定义** | ❌ 未在代码中体现 | 需在 Methods/Results 中明确公式与基准 |
| **augmentation vs architecture** | ✅ 可区分 | 消融实验区分了：baseline vs projection（对比学习）、有/无预训练、不同融合方式 |

### 消融实验可支撑的结论

- **对比学习**：在 gated / attention_pooling 上有提升；在 concat / bilinear 上无明显提升或略差。
- **预训练**：在 cross_attention / bilinear 上显著有效；在 gated / attention_pooling 上接近持平。
- **36.8%**：需在论文中明确定义（如 `(AUC_new - AUC_baseline) / AUC_baseline * 100`），并区分 augmentation gain 与 model/architecture gain。

### 建议

在 Methods 中增加类似表述：
> *Performance improvement* was defined as (Metric_proposed − Metric_baseline) / Metric_baseline × 100%. We report separate gains from (1) data augmentation (sliding window) and (2) model architecture (dual-stream fusion, contrastive learning).

---

## 五、【高风险问题 5】预训练是否“偷看”测试集

### 审稿人关注点
- pre-training 是否使用 full dataset

### 实验现状与结论

| 项目 | 状态 | 说明 |
|------|------|------|
| **预训练仅用训练被试** | ✅ 已实现 | `train_pretrain.py` 中 `get_subject_level_train_val_test_split` 先划分，再用 `timeseries[train_idx]`、`pcc_vectors[train_idx]` 做预训练 |

### 代码证据（train_pretrain.py）

```python
train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(...)
train_ts_dataset = PretrainTSDataset(timeseries[train_idx])   # 仅 train
train_fc_dataset = PretrainFCDataset(pcc_vectors[train_idx])  # 仅 train
```

### 结论

✅ **预训练未使用测试集**：预训练数据严格来自训练集被试，val/test 被试未参与预训练。

### 建议 Methods 表述

> *Pre-training.* Self-supervised pre-training (masked reconstruction) was performed exclusively on the training subset. No data from validation or test subjects were used at any stage of pre-training.

---

## 六、【中等风险】仅在 ABIDE I 上评估

### 审稿人关注点
- 单数据集、泛化性声称

### 结论

⚠️ **当前实验仅在 ABIDE I（或单一 processed 数据）上运行**。不宜使用 “generalizable”“robust” 等强表述，建议在 Limitations 中写明单数据集与缺乏外部验证集的限制。

---

## 七、改进后实验的结果与结论汇总

### 已完成的消融实验

| 实验组 | 数量 | 主要结论 |
|--------|------|----------|
| **Baseline + Freeze** | 20 | unfrozen 最优；freeze_both 最差；concat/cross_attention 表现较好 |
| **Projection + Freeze** | 28 | projection_fusion_attention_pooling_unfrozen 最佳 (AUC=0.744) |
| **No Pretrain** | 5 | cross_attention、bilinear 预训练有效；gated、attention_pooling 接近持平 |

### 主要数值结论

- **最佳配置**：`projection_fusion_attention_pooling_unfrozen` (Test AUC 0.744)
- **预训练有效性**：cross_attention +6.07%，bilinear +3.74%
- **冻结策略**：unfrozen > freeze_tst1 > freeze_tst2 > freeze_both

### 仍待补做的实验

1. **统计验证**：对 1–2 个主实验跑 5-fold 或多 seed，报告 mean±std 与 Bootstrap 95% CI。
2. **LOSO**：若数据含 site，运行 `--eval_protocol loso`，报告跨站点结果。
3. **论文写作**：36.8% 定义、预训练数据范围、site 与单数据集等 limitation 的明确表述。

---

## 八、回复优先级建议

| 优先级 | 问题 | 建议 |
|--------|------|------|
| P0 | 致命 1：Subject-level 泄漏 | ✅ 已解决，在 rebuttal 中引用代码与日志 |
| P0 | 致命 2：统计验证 | 补 5-fold / 多 seed + mean±std + Bootstrap CI |
| P1 | 预训练数据范围 | ✅ 已解决，在 Methods 中明确写出 |
| P1 | Site effect | 补 LOSO 或承认 limitation |
| P2 | 36.8% 定义 | 在 Methods 中给出数学定义与分解 |
| P2 | 单数据集 | 在 Limitations 中写明 |

---

## 九、数据与代码路径

| 内容 | 路径 |
|------|------|
| 消融汇总 | `/root/autodl-tmp/TwoTST/results/ablation_summary.json` |
| 预训练消融日志 | `/root/autodl-tmp/TwoTST/results/ablation_no_pretrain.log` |
| 结果汇总文档 | `docs/ABLATION_RESULTS.md` |
| Subject-level 划分 | `utils/splitters.py` |
| 预训练（仅 train） | `scripts/train_pretrain.py` |
| K-fold + Bootstrap | `scripts/train_finetune.py`, `utils/metrics.py` |
| LOSO | `utils/splitters.get_loso_fold_splits` |
