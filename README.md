# TwoTST: 双流自监督预训练框架

[![GitHub](https://img.shields.io/badge/GitHub-Leezy--Ray/twoTST-blue)](https://github.com/Leezy-Ray/twoTST.git)

TwoTST (Dual-Stream Self-Supervised Pretraining Framework) 是一个用于fMRI数据分析的双流Transformer框架，融合了时序Transformer和连接Transformer的优点，通过自监督预训练和对比学习提升ASD分类性能。

## 📋 项目概述

TwoTST框架包含两个独立的Transformer分支：

- **TST1 (Transformer-TS)**: 处理原始fMRI时间序列，使用ROI-level掩码策略进行预训练
- **TST2 (Transformer-FC)**: 处理PCC（Pearson相关系数）上三角向量，使用元素级掩码策略进行预训练

### 核心特性

- ✅ 双流Transformer架构：分别处理时序和连接特征
- ✅ 自监督预训练：两种不同的掩码策略适配不同数据类型
- ✅ 顺序预训练：先TST1，后TST2，逐步学习表征
- ✅ 可选对比学习：对齐两个分支的特征空间
- ✅ 多种融合策略：支持5种特征融合方法
- ✅ 5折交叉验证：稳健的模型评估

## 🏗️ 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     数据预处理                                │
│  fmri.npy (N, T, R) → 清洗 → (N', T, R)                    │
│  ↓                                                          │
│  时间序列 (N', R, T)  +  PCC向量 (N', R*(R-1)/2)            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Phase 1: TST1 预训练                            │
│  时间序列 → ROI-level掩码 → Transformer-TS → 重建时序      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Phase 2: TST2 预训练                            │
│  PCC向量 → 元素级掩码 → Transformer-FC → 重建PCC          │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│         Phase 3: 对比学习（可选）                            │
│  TST1特征 + TST2特征 → InfoNCE损失 → 特征对齐              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Phase 4: 微调分类                                │
│  融合特征 → MLP分类器 → ASD/TC预测                         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 项目结构

```
TwoTST/
├── models/                        # 模型定义
│   ├── transformer_ts.py         # TST1: 时序Transformer
│   ├── transformer_fc.py         # TST2: 连接Transformer
│   ├── fusion.py                 # 融合模块（5种策略）
│   ├── dual_stream.py            # 双流模型
│   └── __init__.py               # 模型导出
│
├── pretrain/                     # 预训练模块
│   ├── mask_utils.py            # 掩码策略工具
│   ├── pretrain_ts.py           # TST1预训练脚本
│   ├── pretrain_fc.py           # TST2预训练脚本
│   ├── contrastive.py           # 对比学习模块
│   └── __init__.py
│
├── scripts/                      # 训练脚本（按功能分类）
│   ├── data/                    # 数据准备
│   │   ├── prepare_data.py      # 数据预处理
│   │   └── extract_subjects.py  # 提取被试子集
│   ├── train/                   # 训练
│   │   ├── train_pretrain.py    # 预训练入口
│   │   ├── train_finetune.py    # 微调入口
│   │   └── generate_contrastive_checkpoints.py
│   ├── experiments/             # 实验运行
│   │   └── run_experiment.py    # 单实验运行
│   ├── ablation/                # 消融实验
│   ├── validation/              # 统计验证
│   ├── analysis/                # 分析与可视化
│   ├── utils/                   # 工具
│   │   └── start_tensorboard.sh
│   └── run_full_training_autodl.sh  # 完整训练流程入口
│
├── utils/                        # 工具函数
│   ├── data_loader.py           # 数据加载器
│   ├── metrics.py               # 评估指标
│   └── __init__.py
│
├── configs/                      # 配置文件
│   ├── default.yaml             # 默认配置
│   └── base_template.yaml       # 配置模板
│
├── requirements.txt             # 依赖包
└── README.md                    # 本文档
```

**注意**: 训练过程中会生成以下目录（建议添加到 `.gitignore`）：
- `checkpoints/` - 模型检查点
- `logs/` - TensorBoard日志
- `data/` - 数据文件
- `results/` - 实验结果

## 🚀 快速开始

### 1. 环境配置

**系统要求**:
- Python >= 3.7
- CUDA >= 10.2 (GPU推荐)

**安装依赖**:

```bash
# 克隆仓库
git clone https://github.com/Leezy-Ray/twoTST.git
cd twoTST

# 安装依赖
pip install -r requirements.txt
```

主要依赖包：
- PyTorch >= 1.10.0
- NumPy >= 1.20.0
- scikit-learn >= 1.0.0
- TensorBoard >= 2.8.0
- tqdm, PyYAML

### 2. 数据准备

准备您的fMRI数据，格式为 `(n_samples, time_points, n_rois)` 的numpy数组。

```bash
# 运行数据预处理脚本
python scripts/data/prepare_data.py \
    --data_path /path/to/your/fmri.npy \
    --output_dir data/processed \
    --n_rois 200 \
    --time_points 100
```

**数据预处理功能**:
- ✅ 自动清洗ROI全零样本
- ✅ 计算PCC（Pearson相关系数）上三角向量
- ✅ 可选滑动窗口数据增强
- ✅ 保存受试者与站点元信息（`subject_indices`、`site_ids`），用于受试者级划分与站点分层评估

**输出数据格式**:
- `timeseries`: `(n_samples, n_rois, time_points)` - 时间序列
- `pcc_vectors`: `(n_samples, n_rois*(n_rois-1)/2)` - PCC上三角向量
- `labels`: `(n_samples,)` - 标签 (0=ASD, 1=TC)
- `subject_indices`: 每个样本对应的受试者索引，用于受试者级划分（避免滑窗信息泄漏）
- `site_ids`: 每个样本对应的站点ID，用于站点分层K折

**fmri.npy 格式要求**: 需包含 `timeseries`、`label` 和可选 `site` 字段。若含 `site`，则支持站点分层评估。

### 3. 预训练

#### 方式1: 顺序预训练（推荐）

使用统一入口脚本进行顺序预训练：

```bash
python scripts/train/train_pretrain.py \
    --data_path data/processed/processed_data.pkl \
    --pretrain_tst1 \
    --pretrain_tst2 \
    --tst1_epochs 100 \
    --tst2_epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir checkpoints \
    --log_dir logs
```

#### 方式2: 单独预训练

**TST1预训练（时序Transformer）**:

```bash
python pretrain/pretrain_ts.py \
    --data_path data/processed/processed_data.pkl \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --save_dir checkpoints/tst1 \
    --log_dir logs/tst1
```

**TST1特点**:
- 输入: `(batch, n_rois, time_points)` - 时间序列
- 掩码策略: ROI-level掩码（随机掩码25%或50%的ROI整列）
- 预训练任务: 重建被掩码ROI的完整时间序列
- 模型参数: ~19M

**TST2预训练（连接Transformer）**:

```bash
python pretrain/pretrain_fc.py \
    --data_path data/processed/processed_data.pkl \
    --epochs 100 \
    --batch_size 32 \
    --lr 1e-4 \
    --mask_ratio 0.15 \
    --save_dir checkpoints/tst2 \
    --log_dir logs/tst2
```

**TST2特点**:
- 输入: `(batch, pcc_dim)` - PCC上三角向量
- 掩码策略: 元素级掩码（随机掩码15%的PCC值）
- 预训练任务: 重建被掩码的PCC值
- 模型参数: ~16M

### 4. 微调分类

加载预训练权重进行下游分类任务：

```bash
python scripts/train/train_finetune.py \
    --data_path data/processed/processed_data.pkl \
    --tst1_checkpoint checkpoints/tst1/tst1_best.pt \
    --tst2_checkpoint checkpoints/tst2/tst2_best.pt \
    --fusion_type cross_attention \
    --epochs 100 \
    --batch_size 32 \
    --lr 5e-5 \
    --n_folds 5 \
    --use_contrastive  # 可选：启用对比学习
```

**微调参数说明**:
- `--fusion_type`: 融合策略 (`concat`, `gated`, `cross_attention`, `bilinear`, `attention_pooling`)
- `--n_folds`: 交叉验证折数（默认5折）
- `--use_contrastive`: 是否在微调前进行对比学习对齐
- `--use_subject_level_split`: 受试者级划分（默认开启，避免滑窗信息泄漏）
- `--subject_agg_strategy`: 滑窗时受试者级汇总方式 (`prob_mean` | `majority_vote`)

### 评估协议（避免信息泄漏）

- **受试者级划分**: 同一受试者的所有窗口（滑窗产生）仅出现在 train/val/test 之一，使用 `StratifiedGroupKFold`
- **LOSO 跨站点评估**: `--eval_protocol loso` 实现 Leave-One-Site-Out，评估跨站点泛化
- **受试者级评估**: 滑窗场景下，测试集指标按受试者汇总（概率均值或多数投票），报告临床相关的受试者级 AUC/ACC
- **统计报告**: 交叉验证输出 mean ± std 及 Bootstrap 95% 置信区间，结果文件含 PyTorch/CUDA/seed 等复现信息

```bash
# LOSO 评估（需数据含 site 字段）
python scripts/train_finetune.py --eval_protocol loso --data_path data/processed_sw/processed_data.pkl ...
```

### 5. TensorBoard可视化

训练过程中可以使用TensorBoard实时查看训练曲线：

```bash
# 启动TensorBoard服务
bash scripts/utils/start_tensorboard.sh

# 或手动启动
tensorboard --logdir=logs --port=6006 --host=0.0.0.0
```

然后在浏览器中访问 `http://localhost:6006` 查看训练曲线。

## 🔧 配置说明

配置文件位于 `configs/default.yaml`，主要配置项：

```yaml
# 数据配置
data:
  n_rois: 200
  time_points: 100
  pcc_dim: 19900

# TST1配置
tst1:
  emb_dim: 512
  n_heads: 8
  n_layers: 6
  dim_feedforward: 2048

# TST2配置
tst2:
  d_model: 256
  n_heads: 8
  n_layers: 2
  dim_feedforward: 512

# 融合配置
fusion:
  type: cross_attention  # concat/gated/cross_attention/bilinear/attention_pooling

# 对比学习配置
contrastive:
  enabled: false
  temperature: 0.07
  epochs: 50
```

## 📊 融合策略

框架支持5种融合策略：

1. **ConcatFusion**: 简单拼接 `[h_ts; h_fc]`
2. **GatedFusion**: 门控融合 `gate * h_ts + (1-gate) * h_fc`
3. **CrossAttentionFusion**: 交叉注意力融合（推荐）
4. **BilinearFusion**: 双线性融合
5. **AttentionPoolingFusion**: 注意力池化融合

## 📈 评估指标

微调脚本会自动计算以下指标：

- **Accuracy**: 准确率
- **Precision**: 精确率
- **Recall**: 召回率
- **F1 Score**: F1分数
- **AUC**: ROC曲线下面积
- **Sensitivity/Specificity**: 敏感度/特异度

输出示例：
```
Cross-Validation Results:
----------------------------------------
Accuracy    : 0.7234 ± 0.0234
Precision   : 0.7123 ± 0.0198
Recall      : 0.7345 ± 0.0212
F1          : 0.7231 ± 0.0201
AUC         : 0.7891 ± 0.0156
```

## 📝 使用示例

### Python API

```python
import torch
from models import create_dual_stream_model
from utils.data_loader import load_processed_data, TwoTSTDataset

# 加载数据
data = load_processed_data('data/processed/processed_data.pkl')
dataset = TwoTSTDataset(
    data['timeseries'],
    data['pcc_vectors'],
    data['labels']
)

# 创建模型
model = create_dual_stream_model(
    n_rois=200,
    time_points=100,
    pcc_dim=19900,
    fusion_type='cross_attention'
)

# 加载预训练权重
model.load_pretrained_tst1('checkpoints/tst1/tst1_best.pt')
model.load_pretrained_tst2('checkpoints/tst2/tst2_best.pt')

# 前向传播
timeseries = torch.randn(8, 200, 100)
pcc_vector = torch.randn(8, 19900)
logits = model(timeseries, pcc_vector)
```

## 🐛 常见问题

### Q1: 内存不足怎么办？

A: 可以减小batch_size或使用梯度累积：
```bash
--batch_size 16  # 减小batch size
```

### Q2: 如何只训练某个分支？

A: 可以修改脚本，只加载并使用单个分支的预训练权重。

### Q3: 预训练权重如何加载？

A: 在微调脚本中指定checkpoint路径：
```bash
--tst1_checkpoint checkpoints/tst1/tst1_best.pt
--tst2_checkpoint checkpoints/tst2/tst2_best.pt
```

### Q4: 如何使用滑动窗口数据增强？

A: 在数据预处理时启用：
```bash
python scripts/data/prepare_data.py \
    --use_sliding_window \
    --window_size 50 \
    --stride 25
```

## 📚 参考文献

本项目参考了以下工作：
- ROI-level掩码预训练时序Transformer
- PCC上三角向量掩码预训练连接Transformer

## 📄 许可证

本项目仅供研究使用。

## 👥 贡献

欢迎提交Issue和Pull Request！

如果您有任何问题或建议，请：
1. 提交 [Issue](https://github.com/Leezy-Ray/twoTST/issues)
2. 发起 [Pull Request](https://github.com/Leezy-Ray/twoTST/pulls)

## 📧 联系方式

如有问题，请提交Issue或联系项目维护者。

---

**TwoTST** - Dual-Stream Self-Supervised Pretraining Framework for fMRI Analysis

**GitHub**: [https://github.com/Leezy-Ray/twoTST.git](https://github.com/Leezy-Ray/twoTST.git)
