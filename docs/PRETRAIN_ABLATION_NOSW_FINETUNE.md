# 预训练策略消融：滑动窗口 vs 非滑动窗口

## 验证问题

在预训练阶段，使用**滑动窗口**训练的 TST 与使用**非滑动窗口**训练的 TST 相比，哪种对后续采用**非滑动窗口训练方式**的微调阶段能带来更大的性能提升？

## 实验设计

| 实验组 | 预训练方式 | 微调方式 | 配置文件 |
|--------|------------|----------|----------|
| **A**  | 滑窗预训练 | 非滑窗微调 | `nosw_finetune_sw_pretrain_attention_pooling.yaml` |
| **B**  | 非滑窗预训练 | 非滑窗微调 | `group7_projection_fusion_attention_pooling.yaml` |

两组均使用非滑窗微调数据（`processed_data.pkl`）、Attention Pooling 融合。

实验 A 中，非滑窗数据的时间序列会通过 `truncate_timeseries_to: 32` 截断至 32 个时间点，以适配滑窗预训练 TST1 的 `max_seq_len=32`。

## 前置条件

1. **非滑窗预训练**：`checkpoints/tst1/`, `checkpoints/tst2/`, `checkpoints/contrastive_checkpoint.pt`
2. **滑窗预训练**：`checkpoints_sw/tst1/`, `checkpoints_sw/tst2/`, `checkpoints_sw/contrastive_checkpoint.pt`
3. **非滑窗数据**：`data/processed/processed_data.pkl`

若使用 AutoDL 等环境，请根据实际路径修改配置文件中的 checkpoint 路径（如 `/root/autodl-tmp/TwoTST/...`）。

## 运行方式

```bash
bash scripts/run_pretrain_ablation_nosw_finetune.sh
```

结果将保存至 `logs/pretrain_ablation_nosw_finetune/`，可比较两组的 5-fold CV AUC。

## 实现细节

- `load_data()` 支持 `data.truncate_timeseries_to`：当非滑窗数据时间维度大于该值时，将截断至该长度。
- 滑窗预训练 TST1 的 `max_seq_len=32` 会从 checkpoint 加载，与截断后的数据一致。
