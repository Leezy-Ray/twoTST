# 消融实验：对比学习阶段的 TST freeze 策略

## 实验目的

验证在**对比学习阶段**，对预训练的 TST1/TST2 采用 freeze 还是 unfreeze，对**微调阶段融合**的影响更大。

## 实验设计

| 实验 | 对比学习 TST1 | 对比学习 TST2 | 微调 TST1 | 微调 TST2 |
|------|---------------|---------------|-----------|-----------|
| cont_freeze_tst1 | freeze | unfreeze | unfreeze | unfreeze |
| cont_freeze_tst2 | unfreeze | freeze | unfreeze | unfreeze |
| cont_freeze_both | freeze | freeze | unfreeze | unfreeze |
| cont_unfreeze_both | unfreeze | unfreeze | unfreeze | unfreeze |

- **数据**：非滑窗 `processed_data.pkl`
- **融合**：attention_pooling
- **对比学习**：不加载已有 checkpoint，从 pretrained TST 重新训练
- **微调**：TST1、TST2 均 unfrozen

## 运行方式

```bash
cd /root/workplace/exp/TwoTST
bash scripts/run_ablation_contrastive_freeze.sh
```

或直接运行 Python 脚本：

```bash
python scripts/ablation/run_ablation_contrastive_freeze.py
```

## 输出

- 实验结果：`checkpoints/finetune/projection_fusion_attention_pooling_cont_*/results.json`
- 日志：`logs/ablation_contrastive_freeze/*.log`

## 结果汇总

| 实验 | 对比学习 TST1 | 对比学习 TST2 | Test AUC | 结论 |
|------|---------------|---------------|----------|------|
| cont_freeze_tst1 | freeze | unfreeze | **0.7178** | 最优 |
| cont_freeze_tst2 | unfreeze | freeze | 0.6788 | |
| cont_freeze_both | freeze | freeze | 0.6344 | 最差 |
| cont_unfreeze_both | unfreeze | unfreeze | 0.6533 | |

**结论**：对比学习阶段 **freeze TST1、unfreeze TST2** 对微调阶段融合效果最好；freeze both 最差。
