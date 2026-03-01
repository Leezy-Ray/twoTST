# TwoTST 磁盘布局

## 目录规划

| 位置 | 路径 | 内容 |
|------|------|------|
| **系统盘** | `/root/workplace/exp/TwoTST/` | 代码 + 配置 |
| **数据盘** | `/root/autodl-tmp/TwoTST/` | 数据 + 模型 + 日志 + 结果 |

## 系统盘（仅 code + configs）

```
TwoTST/
├── api/           # API 服务
├── configs/       # 配置文件
├── docs/          # 文档
├── models/        # 模型定义
├── pretrain/      # 预训练模块
├── scripts/       # 脚本
├── utils/         # 工具函数
├── README.md
├── requirements.txt
└── ...
```

## 数据盘

```
/root/autodl-tmp/TwoTST/
├── data/                  # 数据
│   ├── processed/         # 非滑窗预处理
│   └── processed_sw/      # 滑窗预处理
├── checkpoints/           # 非滑窗预训练 + 微调
│   ├── tst1/, tst2/
│   ├── contrastive_checkpoint.pt
│   └── finetune/
├── checkpoints_sw/        # 滑窗预训练 + 微调
├── logs/                  # TensorBoard 日志
├── logs_sw/
└── results/               # 实验结果
    ├── best_config_5x/    # 最优配置 5 次实验
    ├── fusion_5x/         # 5 种融合 × 5 次（需运行 run_fusion_5x.py）
    ├── misc_logs/         # 根目录迁移来的 .log
    ├── RESULTS_SUMMARY.md
    └── *.log
```

## 迁移

运行迁移脚本（将系统盘上的 checkpoints/logs/results/data 复制到数据盘并删除）：

```bash
bash scripts/utils/migrate_to_data_disk.sh
```

## 环境变量

- `DATA_ROOT` 或 `RESULTS_ROOT`：可覆盖数据/结果根路径
- 默认数据根：`/root/autodl-tmp/TwoTST`
