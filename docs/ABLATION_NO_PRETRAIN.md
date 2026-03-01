# 消融实验：预训练有效性验证

## 实验目的

验证对 TST1 和 TST2 的预训练是否有效。对比：
- **有预训练**：加载预训练的 TST1 + TST2，然后微调（baseline_*_unfrozen）
- **无预训练**：TST1/TST2 随机初始化，与融合层一起端到端训练

## 实验设计

| 对比项 | 有预训练 | 无预训练 |
|--------|----------|----------|
| TST1/TST2 初始化 | 加载 checkpoint | 随机初始化 |
| 训练方式 | 微调（lr=5e-5） | 端到端（lr=1e-4） |
| 融合方式 | concat, gated, cross_attention, bilinear, attention_pooling | 同上 |

共 **5 个无预训练实验**，与已有 5 个 baseline_*_unfrozen 一一对应。

## 运行方式

```bash
cd /root/workplace/exp/TwoTST

# 运行 5 个无预训练实验
bash scripts/run_ablation_no_pretrain.sh

# 或直接
python scripts/run_ablation_no_pretrain.py
```

## 输出

- 结果目录：`/root/autodl-tmp/TwoTST/checkpoints/finetune/`
- 实验命名：`no_pretrain_concat`, `no_pretrain_gated`, ...
- 日志：`/root/autodl-tmp/TwoTST/results/ablation_no_pretrain.log`

## 对比分析

完成后对比同一融合方式下：
- `baseline_concat_unfrozen` vs `no_pretrain_concat`
- `baseline_gated_unfrozen` vs `no_pretrain_gated`
- ...

若预训练有效，则 baseline_*_unfrozen 的 Test AUC 应显著高于 no_pretrain_*。
