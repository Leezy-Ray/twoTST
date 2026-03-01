# 统计验证与 LOSO 实验说明

## 实验设计

### 1. 5-fold CV（统计验证）
- **目的**：报告 mean ± std、Bootstrap 95% CI，满足审稿对统计严谨性的要求
- **划分**：Subject-level StratifiedGroupKFold，5 折
- **实验**：
  - `cv5_baseline_concat`
  - `cv5_baseline_cross_attention`

### 2. LOSO（跨站点泛化）
- **目的**：评估模型在未见扫描仪/协议上的泛化能力
- **划分**：Leave-One-Site-Out，19 个站点 = 19 折
- **实验**：`loso_baseline_concat`

## 运行方式

```bash
cd /root/workplace/exp/TwoTST

# 使用默认路径（数据在 workplace，checkpoint 在 autodl-tmp）
bash scripts/run_statistical_validation.sh

# 或指定路径
DATA_PATH=/path/to/processed_data.pkl \
CKPT_ROOT=/path/to/checkpoints \
bash scripts/run_statistical_validation.sh
```

## 输出

- **日志**：`/root/autodl-tmp/TwoTST/results/statistical_validation_full.log`
- **5-fold 结果**：`cv5_baseline_concat/results.pkl`、`cv5_baseline_cross_attention/results.pkl`
- **LOSO 结果**：`loso_baseline_concat/results.pkl`

## 收集汇总

实验完成后运行：

```bash
python scripts/validation/collect_statistical_results.py
```

将生成 `statistical_validation_summary.md`，含 mean±std、95% CI 表格。

## 预计耗时

- 5-fold × 2 实验：约 10× 单次训练时间（每折 100 epochs）
- LOSO × 19 折：约 19× 单次训练时间

建议后台运行：`nohup bash scripts/run_statistical_validation.sh > results/statistical_validation_full.log 2>&1 &`
