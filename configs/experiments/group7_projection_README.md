# 实验组7: 投影头微调实验

## 实验目标

充分利用对比学习产生的投影头（projection heads），在微调阶段使用固定的投影头参数进行预测。

## 核心思想

1. **对比学习阶段**：训练投影头，将TST1和TST2的特征投影到统一的对比学习空间
2. **微调阶段**：固定投影头参数，使用投影后的特征进行下游任务

## 实验配置

### 单流实验

| 配置文件 | 描述 | 数据流 |
|---------|------|--------|
| `group7_projection_tst1_only.yaml` | 只用TST1+projection1+MLP | ROI → TST1 → projection1 → MLP |
| `group7_projection_tst2_only.yaml` | 只用TST2+projection2+MLP | PCC → TST2 → projection2 → MLP |

### 双流融合实验

| 配置文件 | 描述 | 数据流 |
|---------|------|--------|
| `group7_projection_fusion_concat.yaml` | Concat融合 | TST1+proj1 + TST2+proj2 → Concat → MLP |
| `group7_projection_fusion_gated.yaml` | Gated融合 | TST1+proj1 + TST2+proj2 → Gated → MLP |
| `group7_projection_fusion_cross_attention.yaml` | Cross Attention融合 | TST1+proj1 + TST2+proj2 → Cross Attention → MLP |
| `group7_projection_fusion_bilinear.yaml` | Bilinear融合 | TST1+proj1 + TST2+proj2 → Bilinear → MLP |
| `group7_projection_fusion_attention_pooling.yaml` | Attention Pooling融合 | TST1+proj1 + TST2+proj2 → Attention Pooling → MLP |

## 关键配置参数

### 对比学习配置
```yaml
contrastive:
  enabled: true  # 必须启用对比学习
  proj_output_dim: 128  # 投影头输出维度
  freeze_tst1: true  # 冻结TST1，只训练TST2和投影头
  freeze_tst2: false
```

### 微调配置
```yaml
finetune:
  use_projection: true  # 关键：启用投影头微调模式
  freeze_tst1: false  # TST1可以继续训练
  freeze_tst2: false  # TST2可以继续训练
```

## 数据流示意

### 单流TST1
```
原始ROI数据 (batch, 100, 200)
  ↓
TST1 (可训练)
  ↓
特征 (batch, 512)
  ↓
projection1 (固定参数)
  ↓
投影特征 (batch, 128)
  ↓
MLP分类器
  ↓
分类结果 (batch, 2)
```

### 单流TST2
```
PCC向量 (batch, 19900)
  ↓
TST2 (可训练)
  ↓
特征 (batch, 256)
  ↓
projection2 (固定参数)
  ↓
投影特征 (batch, 128)
  ↓
MLP分类器
  ↓
分类结果 (batch, 2)
```

### 双流融合
```
ROI数据 → TST1 → projection1 → z1 (batch, 128)
PCC数据 → TST2 → projection2 → z2 (batch, 128)
  ↓
融合模块 (Concat/Gated/Cross Attention等)
  ↓
融合特征
  ↓
MLP分类器
  ↓
分类结果 (batch, 2)
```

## 运行实验

```bash
# 运行单个实验
python scripts/run_experiment.py --config configs/experiments/group7_projection_tst1_only.yaml

# 运行所有投影头微调实验
for config in configs/experiments/group7_projection_*.yaml; do
    python scripts/run_experiment.py --config $config
done
```

## 预期效果

1. **投影头的作用**：投影头将不同维度的特征（TST1: 512维，TST2: 256维）映射到统一的对比学习空间（128维），使得两个模态的特征更容易对齐和融合。

2. **固定投影头的优势**：
   - 保持对比学习阶段学到的特征对齐
   - 减少微调阶段的参数数量
   - 可能提高模型的泛化能力

3. **与原始方法的对比**：
   - 原始方法：直接使用TST1和TST2的特征（512维和256维）进行融合
   - 投影头方法：使用投影后的特征（都是128维）进行融合，特征已经对齐

## 注意事项

1. **必须启用对比学习**：`contrastive.enabled: true`
2. **投影头参数固定**：在微调阶段，投影头的参数被冻结，不会更新
3. **TST参数可训练**：TST1和TST2的参数可以根据配置继续训练或冻结
4. **融合维度**：使用投影头时，融合模块的输入维度是投影头的输出维度（128），而不是TST的原始维度
