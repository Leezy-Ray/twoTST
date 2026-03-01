# TwoTST 统计验证结果

基于 `scripts/run_statistical_validation.sh` 的 5-fold CV 与 LOSO 跨站点泛化评估结果。

## 实验设置

| 项目 | 说明 |
|------|------|
| 数据 | ABIDE 预处理数据（`processed_data.pkl`），含滑动窗口 |
| 预训练 | TST1 + TST2，来自 `/root/autodl-tmp/TwoTST/checkpoints` |
| 微调 | baseline concat / baseline cross_attention |
| 评估粒度 | **Window-level**（窗口级，未使用 prob_mean/majority_vote 聚合） |
| 划分方式 | 受试者级划分（subject-level split），避免滑窗泄漏 |

---

## 1. 5-fold CV：baseline_concat

受试者级 5 折交叉验证。

| 指标 | Mean ± Std | 95% CI |
|------|------------|--------|
| **Accuracy** | 0.6262 ± 0.0491 | [0.5844, 0.6677] |
| **Precision** | 0.6367 ± 0.0582 | [0.5859, 0.6876] |
| **Recall** | 0.6506 ± 0.0495 | [0.6067, 0.6912] |
| **F1** | 0.6418 ± 0.0413 | [0.6055, 0.6781] |
| **AUC** | 0.6937 ± 0.0376 | [0.6639, 0.7264] |

---

## 2. 5-fold CV：baseline_cross_attention

受试者级 5 折交叉验证，融合方式为 cross_attention。

| 指标 | Mean ± Std | 95% CI |
|------|------------|--------|
| **Accuracy** | 0.6261 ± 0.0297 | [0.6068, 0.6541] |
| **Precision** | 0.6380 ± 0.0332 | [0.6147, 0.6711] |
| **Recall** | 0.6389 ± 0.0519 | [0.6010, 0.6873] |
| **F1** | 0.6369 ± 0.0274 | [0.6143, 0.6601] |
| **AUC** | 0.6724 ± 0.0269 | [0.6496, 0.6988] |

---

## 3. LOSO：baseline_concat（跨站点泛化）

Leave-One-Site-Out，19 个站点，19 折，每次留出一个站点作为测试集，评估跨站点泛化。

| 指标 | Mean ± Std | 95% CI |
|------|------------|--------|
| **Accuracy** | 0.6577 ± 0.1312 | [0.5997, 0.7203] |
| **Precision** | 0.6715 ± 0.1334 | [0.6182, 0.7355] |
| **Recall** | 0.6738 ± 0.1639 | [0.5962, 0.7501] |
| **F1** | 0.6647 ± 0.1354 | [0.6053, 0.7270] |
| **AUC** | 0.7090 ± 0.1327 | [0.6490, 0.7706] |

### LOSO 各站点 (fold) 结果概览

| 测试站点 | Accuracy | AUC | 备注 |
|----------|----------|-----|------|
| UCLA_2 | 0.8077 | 0.8994 | 最佳 |
| UM_2 | 0.8529 | 0.8791 | 较佳 |
| YALE | 0.7636 | 0.7897 | 较佳 |
| USM | 0.7324 | 0.8217 | |
| OLIN | 0.4706 | 0.5649 | 较低 |
| SBL | 0.5714 | 0.5408 | 较低 |
| STANFORD | 0.5385 | 0.6132 | 较低 |

跨站点性能波动明显（std ≈ 0.13），说明不同站点间分布差异较大。

---

## 汇总对比

| 实验 | Accuracy | AUC |
|------|----------|-----|
| 5-fold CV (concat) | 0.6262 ± 0.049 | 0.6937 ± 0.038 |
| 5-fold CV (cross_attention) | 0.6261 ± 0.030 | 0.6724 ± 0.027 |
| **LOSO (concat)** | **0.6577 ± 0.131** | **0.7090 ± 0.133** |

**简要结论：**

1. **5-fold CV**：concat 与 cross_attention 表现接近，concat 的 AUC 略高。
2. **LOSO**：AUC 略高于 5-fold，但标准差更大，跨站点泛化存在明显差异。
3. 跨站点评估时，部分站点（如 UCLA_2、UM_2）表现较好，部分（如 OLIN、SBL）较差。

---

## 说明

- 指标均在 **window-level** 计算，未使用 `prob_mean` 或 `majority_vote` 聚合到受试者级。
- 结果来源：`/root/autodl-tmp/TwoTST/results/statistical_validation_full.log`
- 脚本末尾曾出现 `unexpected EOF` 语法错误，但 LOSO 实验已完整运行完毕。
