# 审稿意见 vs 实验实现对照表

## 一、【致命问题 1】Subject-level 数据泄漏

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| 先 subject-level split | ✅ 已实现 | `utils/splitters.py` 的 `get_subject_level_fold_splits` / `get_subject_level_train_val_test_split` |
| 再在各 split 内做滑窗 | ✅ 已满足 | 滑窗在 prepare_data 阶段完成，划分时用 subject_indices 保证同被试同 split |
| "All windows from a subject confined to single subset" | ✅ 已满足 | StratifiedGroupKFold(groups=subject_indices) 实现 |

**微调/run_experiment**：已使用 subject-level 划分。  
**预训练 (train_pretrain.py)**：⚠️ 仍为 sample-level 划分。若用滑窗数据预训练会泄漏；若用非滑窗数据（1 subject = 1 sample）则无泄漏。

---

## 二、【致命问题 2】统计显著性 + 单次划分

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| 多次运行 / 标准差 | ✅ 已实现 | K 折 CV，每折一结果，输出 mean ± std |
| 置信区间 | ✅ 已实现 | Bootstrap 95% CI |
| 显著性检验 | ❌ 未实现 | 未做 DeLong/McNemar 等 |
| Cross-validation | ✅ 已实现 | 5-fold StratifiedGroupKFold |

---

## 三、【高风险问题 3】ABIDE Site Effect

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| site 信息保存 | ✅ 已实现 | `site_ids` 写入 processed_data.pkl |
| LOSO (leave-one-site-out) | ✅ 已实现 | `--eval_protocol loso`，`utils/splitters.get_loso_fold_splits` |
| 明确承认 limitation | 待补充 | 需在论文 Discussion 中写明 |

---

## 四、【高风险问题 4】36.8% Improvement 表述

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| 数学定义 improvement | ❌ 代码无关 | 需在 Methods/Results 中明确 |
| 区分 augmentation vs architecture gain | ❌ 代码无关 | 需通过实验设计 + 写作区分 |

---

## 五、【高风险问题 5】预训练是否用测试集

| 要求 | 实现状态 | 说明 |
|------|----------|------|
| 预训练仅在训练被试上 | ✅ 已实现 | `train_pretrain.py` 已改用 subject-level 划分 |
| 在论文中明确说明 | 待补充 | 需在 Methods 中写明 |

---

## 待修复项汇总（代码层面已完成）

1. ~~train_pretrain.py~~：✅ 已改用 subject-level 划分
2. ~~LOSO~~：✅ 已实现，`python scripts/train_finetune.py --eval_protocol loso ...`
3. **论文写作**：36.8% 定义、预训练数据范围、site limitation 的明确表述
