# TwoTST 脚本目录

按功能分类的脚本组织说明。

## 目录结构

```
scripts/
├── data/               # 数据准备
│   ├── prepare_data.py         # 数据预处理（fmri → processed_data.pkl）
│   └── extract_subjects.py     # 提取被试子集
│
├── train/              # 训练
│   ├── train_pretrain.py       # 预训练 TST1+TST2
│   ├── train_finetune.py       # 微调（K-fold / LOSO）
│   └── generate_contrastive_checkpoints.py  # 生成对比学习 checkpoint
│
├── experiments/        # 实验运行
│   └── run_experiment.py       # 单实验运行（根据 YAML 配置）
│
├── ablation/           # 消融实验
│   ├── run_ablation_baseline_and_freeze.py   # baseline + 冻结策略
│   ├── run_ablation_no_pretrain.py           # 无预训练消融
│   └── collect_ablation_results.py           # 收集消融结果
│
├── validation/         # 统计验证
│   └── collect_statistical_results.py        # 收集统计验证结果
│
├── analysis/           # 分析与可视化
│   ├── analyze_attention.py    # 注意力权重分析
│   ├── predict_with_sliding_window.py  # 滑动窗口预测
│   ├── plot_pretrain_loss_v3.py        # 预训练损失曲线
│   ├── plot_fusion_comparison.py       # 融合方式对比图
│   └── plot_test_comparison.py         # 测试结果对比图
│
├── utils/              # 工具
│   ├── start_tensorboard.sh    # TensorBoard 启动
│   └── README_tensorboard.md   # TensorBoard 使用说明
│
├── run_full_training_autodl.sh     # 完整训练流程（主入口）
├── run_projection_experiments.sh   # 投影头微调实验批量运行
├── run_statistical_validation.sh   # 统计验证（5-fold CV + LOSO）
├── run_ablation_baseline_and_freeze.sh   # 消融实验入口
└── run_ablation_no_pretrain.sh           # 无预训练消融入口
```

## 常用命令

| 功能 | 命令 |
|------|------|
| 完整训练流程 | `bash scripts/run_full_training_autodl.sh` |
| 单实验运行 | `python scripts/experiments/run_experiment.py --config configs/experiments/group7_xxx.yaml` |
| 统计验证 | `bash scripts/run_statistical_validation.sh` |
| 消融实验 | `bash scripts/run_ablation_baseline_and_freeze.sh` |
| TensorBoard | `bash scripts/utils/start_tensorboard.sh` |
