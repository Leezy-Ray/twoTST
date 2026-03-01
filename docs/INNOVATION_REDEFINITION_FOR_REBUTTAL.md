# 基于审稿意见的创新点重定义与消融对照

本文档基于 `REVIEW_RESPONSE_DETAILED.md` 与 `REVIEW_RESPONSE_CHECKLIST.md` 中的审稿人关注点，结合现有消融实验，重新归纳**可写进论文/返修**的创新点，并标明每条对应的实验支撑与审稿人诉求。

---

## 一、审稿人核心关切（简要回顾）

| 类型 | 审稿人关切 | 文档依据 |
|------|------------|----------|
| **致命 1** | Subject-level 数据泄漏（滑窗导致同被试多样本） | REVIEW_RESPONSE_DETAILED §一 |
| **致命 2** | 仅单次划分、无标准差/置信区间/显著性 | REVIEW_RESPONSE_DETAILED §二 |
| **高风险 3** | ABIDE 多站点、site effect 未控制 | REVIEW_RESPONSE_DETAILED §三 |
| **高风险 4** | 36.8% 提升未定义、augmentation vs architecture 未区分 | REVIEW_RESPONSE_DETAILED §四 |
| **高风险 5** | 预训练是否用测试集 | REVIEW_RESPONSE_DETAILED §五 |
| **中等** | 仅 ABIDE I、泛化性声称过强 | REVIEW_RESPONSE_DETAILED §六 |

**与创新点直接相关**：审稿人要求**明确区分**“数据增强带来的提升”与“模型/架构带来的提升”；你们已通过实验确认**滑窗对最终融合贡献不大**，返修时应**弱化滑窗、强化架构与训练策略**。

---

## 二、您提出的两条创新点（与审稿对齐）

### 创新点 1：自监督双 TST + 对比学习投影 → 微调阶段融合二分类更优

**表述建议（可直接用于 Abstract/Contributions）：**

- 提出**两阶段表征对齐流程**：先对 ROI 时序与 PCC 分别自监督预训练两个 TST，再在**冻结或部分冻结 TST** 下用对比学习训练**双流投影头**，将两路特征映射到同一空间；微调时**固定投影头**，在投影空间做多模态融合与二分类。
- **核心主张**：相比“仅预训练、无对比学习、直接融合”（baseline），在**部分融合方式**（如 gated、attention_pooling）下，引入对比学习与投影头能**显著提升**下游 AUC；最佳配置为 `projection_fusion_attention_pooling_unfrozen`（AUC 0.744）。

**消融支撑（已有）：**

- **有/无对比学习（baseline vs projection）**：48 个实验已系统对比“同融合、同冻结”下 baseline 与 projection 的 AUC（见 `ABLATION_ANALYSIS_AND_CONCLUSIONS.md` §3）。
- **结论**：对比学习在 **gated**、**attention_pooling** 上带来明确提升（如 +0.0314、+0.0309）；在 concat、bilinear 上 baseline 已足够强，对比学习增益有限或为负。论文中应**限定**“在 gated/attention_pooling 等融合下有效”，避免泛化到所有融合。

**审稿对齐**：满足“区分 augmentation vs architecture”：对比学习+投影头属于**架构/训练策略**，与滑窗数据增强无关；返修时明确写“improvement from contrastive projection”而非笼统的“数据增强”。

---

### 创新点 2：预训练与微调必要性——参数变化 + 无预训练消融

**表述建议：**

- 通过**微调前后参数相对变化**（Mean Relative Change）量化各模块对任务的适配程度：TST1/TST2 变化相对较小（约 0.32–0.44），投影头变化更大（约 0.31–1.45），说明预训练表征较稳定、投影头在微调中承担较多任务适应。
- 通过**无预训练消融**证明：在多种融合方式下，**有预训练**相比**随机初始化 TST 端到端训练**能带来性能提升或持平；在 cross_attention、bilinear 上预训练**显著有效**（如 +6.07%、+3.74%），从而支撑“预训练+微调”流程的必要性。

**消融支撑（已有）：**

- **无预训练消融**：已做且已写进文档。`run_ablation_no_pretrain.py` 将 `use_pretrained=false`，TST1/TST2 随机初始化，与融合层一起端到端训练；5 种融合各 1 个实验，与 5 个 baseline_*_unfrozen 一一对应。
- **结果**：`ABLATION_RESULTS.md` §1.2 与 `ABLATION_NO_PRETRAIN.md`：cross_attention +6.07%，bilinear +3.74%，concat +0.91%；gated 持平，attention_pooling 略降。足以支撑“预训练在多数融合下有效，在部分融合上显著”的结论。
- **参数变化**：`PARAM_CHANGE_SUMMARY.md`（Mean Relative Change，5 融合×5 seeds）已支持“TST 相对稳定、投影头变化大”的叙述。

**审稿对齐**：预训练“未偷看测试集”已在 REVIEW_RESPONSE_DETAILED §五 说明；此处创新点强调**预训练+微调的必要性**由**无预训练消融 + 参数变化分析**共同支撑，与审稿人关心的“预训练数据范围”不冲突。

---

## 三、其他可由现有实验支撑的创新点/贡献

以下均可从现有实验与文档中直接引用，用于丰富 Contributions 或 Discussion，且不依赖滑窗。

1. **双流融合策略的系统比较**  
   在**同一预训练与同一数据划分**下，系统比较 5 种融合（concat, gated, cross_attention, bilinear, attention_pooling）及多种冻结策略（unfrozen / freeze_tst1 / freeze_tst2 / freeze_both），给出“unfrozen > freeze_tst1 > freeze_tst2 > freeze_both”的规律，并推荐最佳组合（projection + attention_pooling + unfrozen）。审稿人要求“区分 augmentation vs architecture”，此条属于**架构与训练策略**的贡献。

2. **单模态 vs 双模态**  
   projection_tst2_only 稳定在约 0.72，projection_tst1_only 较差（约 0.56–0.63），可简要讨论 PCC 模态在本任务中的相对重要性或“TST1 需与 TST2 联合微调才更好”的现象，作为方法分析的补充。

3. **对比学习与融合方式的匹配关系**  
   实验表明对比学习**并非在所有融合下都优于 baseline**：在 gated/attention_pooling 下有效，在 concat/bilinear 下 baseline 已足够。可作为“设计建议”或 Discussion：选择融合方式时需考虑是否引入对比学习与投影头。

4. **Subject-level 划分与无泄漏保证**  
   所有实验采用 subject-level 划分，滑窗时同被试仅出现在一个 subset，可直接回应审稿人“数据泄漏”质疑，并在 Methods 中采用 `METHODS_PHRASING_FOR_REBUTTAL.md` 的表述。

5. **预训练数据范围透明**  
   预训练仅用训练集被试、未使用 val/test，满足审稿人“预训练是否用测试集”的关切，可在 Methods 中明确写出（见 REVIEW_RESPONSE_DETAILED §五 建议表述）。

---

## 四、建议弱化或移入 Limitation 的内容

- **滑动窗口数据增强**：你们已发现对最终融合贡献不大；审稿人又要求区分 augmentation vs architecture。建议：**不再将滑窗作为主要创新点**；若保留，仅作“可选数据增强”并明确写出“在本实验中其对融合性能提升有限”，或在 Limitation 中说明“仅评估了非滑窗主实验，滑窗对融合收益未达显著”。
- **36.8% 等单一数字**：按审稿人要求给出**数学定义**（如 (AUC_new − AUC_baseline) / AUC_baseline × 100%），并**分解**为“来自对比学习/投影头的提升”与“来自数据增强的提升”；若滑窗贡献小，则主要报告“来自架构/对比学习”的增益。
- **单数据集与 site**：仅在 ABIDE I 上评估、未做 LOSO 或 site 分层时，避免“generalizable”“robust”等强表述，在 Limitation 中写明单数据集与 site 未显式控制（见 REVIEW_RESPONSE_DETAILED §三、§六）。

---

## 五、消融实验速查（对应“是否有做”）

| 问题 | 是否已做 | 位置/说明 |
|------|----------|-----------|
| 无对比学习（baseline vs projection） | ✅ 已做 | 48 实验，同融合同冻结对比；见 ABLATION_ANALYSIS_AND_CONCLUSIONS §3 |
| 无预训练（随机初始化 TST 端到端） | ✅ 已做 | 5 个 no_pretrain_* 实验；`run_ablation_no_pretrain.py`，ABLATION_NO_PRETRAIN.md、ABLATION_RESULTS.md §1 |
| 不同冻结策略 | ✅ 已做 | baseline/projection × 多种冻结，见 ABLATION_RESULTS §2、ABLATION_ANALYSIS §4 |
| 参数变化/敏感性（微调必要性） | ✅ 已做 | PARAM_CHANGE_SUMMARY.md，5 融合×5 seeds，Mean Relative Change |
| 滑窗 vs 非滑窗预训练对（非滑窗）微调的影响 | ⚠️ 有设计 | PRETRAIN_ABLATION_NOSW_FINETUNE.md；若结果支持“滑窗对融合贡献不大”，可写进 Limitation 或简短结论 |

---

## 六、返修时创新点表述建议（精简版）

可直接用于 Contribution 或 Rebuttal 的 3–4 条：

1. **双流时序+连接组表征与两阶段对齐**：ROI 时序与 PCC 分别用自监督 TST 预训练，再通过对比学习训练双流投影头并在微调中固定使用，使多模态融合在投影空间进行；在 gated 与 attention_pooling 融合下相较无对比学习的 baseline 取得显著 AUC 提升，最佳配置达 AUC 0.744。  
2. **预训练与微调必要性的双重证据**：  
   - 无预训练消融显示，在 cross_attention、bilinear 等融合下预训练带来显著提升（如 +6.07%、+3.74%）；  
   - 微调前后参数变化表明 TST 参数相对稳定、投影头承担主要任务适应，与“预训练表征+轻量微调”的设计一致。  
3. **融合与冻结策略的系统消融**：在同一预训练与数据划分下，系统比较 5 种融合与 4 种冻结策略，给出最优组合与规律，并讨论对比学习与融合方式的匹配关系。  
4. **严格 subject-level 划分与预训练数据范围**：所有划分按被试进行，滑窗时同被试仅出现在同一 subset；预训练仅使用训练集被试，满足无泄漏与审稿人对预训练范围的要求。

如需，可在此基础上再写一段“与审稿人意见对应表”（每条创新点对应哪条审稿意见、哪张表/哪节实验），便于逐条回复时引用。

---

## 七、最佳配置的 5-fold 与 LOSO（审稿统计验证）

审稿人要求对**主实验**做多次划分/多折或 LOSO，并报告 mean±std、95% CI。建议对**最佳配置**（pretrain TST1/TST2 → train projection → attention_pooling，微调 unfrozen）做：

1. **5-fold CV**：受试者级 5 折，报告 AUC/Accuracy 等 mean±std 及 Bootstrap 95% CI。
2. **LOSO**：若数据含 `site_ids`，做 Leave-One-Site-Out，报告跨站点泛化。

**脚本与用法：**

- **5-fold**：
  ```bash
  python scripts/run_best_config_5fold_loso.py --eval_protocol kfold --n_folds 5 --save_dir results/best_config_5fold
  ```
- **LOSO**（需 `processed_data.pkl` 含 `site_ids`）：
  ```bash
  python scripts/run_best_config_5fold_loso.py --eval_protocol loso --save_dir results/best_config_loso
  ```
- **一键跑 5-fold + LOSO**：
  ```bash
  bash scripts/run_best_config_5fold_loso.sh
  ```

结果写入 `save_dir/summary.json`（含各折指标及 mean±std、ci95_lower/upper），可直接用于论文 Results 与 Rebuttal。
