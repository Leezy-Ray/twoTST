# TwoTST 融合实验配置说明

## 实验目标

找到最佳的融合方式，评估指标优先级：**AUC > Accuracy**

## 评估协议（受试者级、避免信息泄漏）

- **受试者级划分**: 使用 `subject_indices` 确保同一受试者的所有样本（含滑窗）仅出现在单一 split
- **站点分层**: 若 `processed_data.pkl` 含 `site_ids`，采用 StratifiedGroupKFold 进行站点分层
- **受试者级指标**: 滑窗时测试指标按受试者汇总（`prob_mean` 或 `majority_vote`）
- **统计与复现**: 输出 mean ± std、Bootstrap 95% CI，结果含 PyTorch/CUDA/seed 等

## 实验分组

### 实验组1: 基线实验（无对比学习）

比较不同融合方式的效果。

| 配置文件 | 融合方式 | 对比学习 | 微调冻结 |
|---------|---------|---------|---------|
| `group1_baseline_concat.yaml` | Concat | ❌ | 无 |
| `group1_baseline_gated.yaml` | Gated | ❌ | 无 |
| `group1_baseline_cross_attention.yaml` | Cross Attention | ❌ | 无 |
| `group1_baseline_bilinear.yaml` | Bilinear | ❌ | 无 |
| `group1_baseline_attention_pooling.yaml` | Attention Pooling | ❌ | 无 |

### 实验组2: 对比学习实验

比较不同对比学习冻结策略的效果。

| 配置文件 | 对比学习冻结策略 | 融合方式 |
|---------|-----------------|---------|
| `group2_contrastive_both_train.yaml` | 两个TST都训练 | Cross Attention |
| `group2_contrastive_freeze_tst1.yaml` | 冻结TST1，训练TST2 | Cross Attention |
| `group2_contrastive_freeze_tst2.yaml` | 冻结TST2，训练TST1 | Cross Attention |
| `group2_contrastive_freeze_both.yaml` | 两个TST都冻结 | Cross Attention |

### 实验组3: 微调冻结实验

比较微调时不同冻结策略的效果。

| 配置文件 | 微调冻结策略 | 对比学习 |
|---------|-------------|---------|
| `group3_finetune_freeze_tst1.yaml` | 冻结TST1 | ❌ |
| `group3_finetune_freeze_tst2.yaml` | 冻结TST2 | ❌ |
| `group3_finetune_freeze_both.yaml` | 冻结两个TST | ❌ |

### 实验组4: 单流对照实验

作为消融实验，验证双流融合的必要性。

| 配置文件 | 使用的模型 |
|---------|----------|
| `group4_single_tst1.yaml` | 只用TST1（时序Transformer） |
| `group4_single_tst2.yaml` | 只用TST2（连接Transformer） |

### 实验组5: 滑动窗口增强实验

测试数据增强的效果。

| 配置文件 | 滑动窗口 | 窗口大小 | 步长 |
|---------|---------|---------|------|
| `group5_sliding_window_cross_attention.yaml` | ✅ | 50 | 25 |

### 实验组6: 组合实验

测试对比学习+微调冻结的组合效果。

| 配置文件 | 对比学习策略 | 微调策略 |
|---------|-------------|---------|
| `group6_contrastive_finetune_freeze_both.yaml` | 两个TST都训练 | 冻结两个TST |
| `group6_contrastive_only_fc_finetune.yaml` | 只训练FC-Encoder | 只用FC-Encoder |

## 实验执行顺序建议

### 第一阶段：基线实验
```bash
# 先确定最佳融合方式
python scripts/experiments/run_experiment.py --config configs/experiments/group1_baseline_concat.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group1_baseline_gated.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group1_baseline_cross_attention.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group1_baseline_bilinear.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group1_baseline_attention_pooling.yaml
```

### 第二阶段：消融实验
```bash
# 验证双流的必要性
python scripts/experiments/run_experiment.py --config configs/experiments/group4_single_tst1.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group4_single_tst2.yaml
```

### 第三阶段：对比学习实验
```bash
# 在最佳融合方式上测试对比学习
python scripts/experiments/run_experiment.py --config configs/experiments/group2_contrastive_both_train.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group2_contrastive_freeze_tst1.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group2_contrastive_freeze_tst2.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group2_contrastive_freeze_both.yaml
```

### 第四阶段：微调冻结实验
```bash
# 测试微调时的冻结策略
python scripts/experiments/run_experiment.py --config configs/experiments/group3_finetune_freeze_tst1.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group3_finetune_freeze_tst2.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group3_finetune_freeze_both.yaml
```

### 第五阶段：组合实验
```bash
# 测试最佳组合
python scripts/experiments/run_experiment.py --config configs/experiments/group6_contrastive_finetune_freeze_both.yaml
python scripts/experiments/run_experiment.py --config configs/experiments/group6_contrastive_only_fc_finetune.yaml
```

### 第六阶段：数据增强实验（可选）
```bash
# 在最佳配置上测试滑动窗口增强
python scripts/experiments/run_experiment.py --config configs/experiments/group5_sliding_window_cross_attention.yaml
```

## 配置文件关键参数说明

### 对比学习配置
```yaml
contrastive:
  enabled: true/false      # 是否启用对比学习
  freeze_tst1: true/false  # 是否冻结TST1
  freeze_tst2: true/false  # 是否冻结TST2
  mode: "both"   # 训练模式（已废弃，实际由freeze_tst1和freeze_tst2控制）
```

### 融合配置
```yaml
fusion:
  type: concat/gated/cross_attention/bilinear/attention_pooling
  use_tst1: true/false     # 是否使用TST1特征
  use_tst2: true/false     # 是否使用TST2特征
```

### 微调配置
```yaml
finetune:
  freeze_tst1: true/false  # 微调时是否冻结TST1
  freeze_tst2: true/false  # 微调时是否冻结TST2
  feature_source: "pretrain"/"contrastive"  # 特征来源
```

## 实验结果汇总表格

实验完成后，请在此填写结果：

| 实验名称 | AUC | Accuracy | Sensitivity | Specificity | F1 |
|---------|-----|----------|-------------|-------------|-----|
| baseline_concat | | | | | |
| baseline_gated | | | | | |
| baseline_cross_attention | | | | | |
| baseline_bilinear | | | | | |
| baseline_attention_pooling | | | | | |
| single_tst1 | | | | | |
| single_tst2 | | | | | |
| contrastive_both_train | | | | | |
| contrastive_freeze_tst1 | | | | | |
| contrastive_freeze_tst2 | | | | | |
| contrastive_freeze_both | | | | | |
| finetune_freeze_tst1 | | | | | |
| finetune_freeze_tst2 | | | | | |
| finetune_freeze_both | | | | | |
| contrastive_finetune_freeze_both | | | | | |
| contrastive_only_fc_finetune | | | | | |
| sliding_window_cross_attention | | | | | |
