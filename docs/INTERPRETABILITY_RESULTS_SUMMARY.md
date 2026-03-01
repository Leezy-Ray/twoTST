# 可解释性结果摘要（最佳模型）

- **模型**：`projection_fusion_attention_pooling_unfrozen`（AUC ≈ 0.744）
- **权重**：`/root/autodl-tmp/TwoTST/checkpoints/finetune/projection_fusion_attention_pooling_unfrozen/best_model.pt`
- **脚本**：`scripts/analysis/interpretability_gradients.py`（梯度重要性，target_class=ASD）
- **数据**：500 样本，非滑窗 `processed_data.pkl`

## 连接重要性（Top 10，模型判 ASD 时最依赖的连接）

| 排名 | ROI i | ROI j | 重要性 |
|------|-------|-------|--------|
| 1 | 60 | 175 | 0.0668 |
| 2 | 101 | 159 | 0.0565 |
| 3 | 62 | 166 | 0.0522 |
| 4 | 51 | 86 | 0.0509 |
| 5 | 41 | 196 | 0.0504 |
| 6 | 85 | 109 | 0.0502 |
| 7 | 18 | 31 | 0.0501 |
| 8 | 60 | 102 | 0.0499 |
| 9 | 141 | 159 | 0.0492 |
| 10 | 29 | 187 | 0.0490 |

（ROI 索引为 CC200 等 atlas 的 0–199，可映射到具体脑区名称。）

## 脑区重要性（Top 10，时序对分类贡献最大的 ROI）

| 排名 | ROI | 重要性 |
|------|-----|--------|
| 1 | 138 | 0.000265 |
| 2 | 150 | 0.000214 |
| 3 | 184 | 0.000210 |
| 4 | 177 | 0.000197 |
| 5 | 80 | 0.000192 |
| 6 | 129 | 0.000192 |
| 7 | 36 | 0.000191 |
| 8 | 159 | 0.000189 |
| 9 | 178 | 0.000186 |
| 10 | 85 | 0.000186 |

## 输出文件位置（autodl-tmp）

- `results/interpretability/interpretability_results.json`：全量重要性 + top 100 连接、top 50 ROI
- `results/interpretability/connection_importance_matrix.npy`：200×200 连接矩阵（可脑网络可视化）
- `results/interpretability/roi_importance_vector.npy`：200 维 ROI 重要性向量

### Python 可视化（matplotlib）

运行以下命令生成 4 张图（需先有上述 .npy 与 .json）：

```bash
cd /root/workplace/exp/TwoTST
python scripts/analysis/plot_interpretability.py \
  --result_dir /root/autodl-tmp/TwoTST/results/interpretability \
  --output_dir  /root/autodl-tmp/TwoTST/results/interpretability
```

**与医学研究对照：使用 CC200 标签**

默认使用项目中的 **CC200 官方文件** `data/labels/cc200_coordinates.json`（Craddock 200 区编号 + 脑叶/半球），条形图显示为「Region N (lobe_hemisphere)」，便于与文献中的 CC200 编号对照：

```bash
python scripts/analysis/plot_interpretability.py \
  --result_dir /root/autodl-tmp/TwoTST/results/interpretability \
  --labels data/labels/cc200_coordinates.json
```

（不指定 `--labels` 时默认即为该 JSON。）若提供 CSV（列：roi_index, region_name_en, lobe），也可用 `--labels xxx.csv` 显示自定义脑区学名。

生成的图片（均在 `results/interpretability/` 下）：

| 文件名 | 说明 |
|--------|------|
| `connection_importance_heatmap.png` | 200×200 连接重要性矩阵热图（全矩阵） |
| `connection_importance_heatmap_uppertri.png` | 仅上三角连接热图（避免重复） |
| `roi_importance_bars.png` | Top-30 ROI 重要性条形图 |
| `top_connections_bars.png` | Top-20 连接 (ROI i–ROI j) 横向条形图 |

## 复现命令

```bash
cd /root/workplace/exp/TwoTST
python scripts/analysis/interpretability_gradients.py \
  --data_path   /root/autodl-tmp/TwoTST/data/processed/processed_data.pkl \
  --checkpoint  /root/autodl-tmp/TwoTST/checkpoints/finetune/projection_fusion_attention_pooling_unfrozen/best_model.pt \
  --config      configs/experiments/group7_projection_fusion_attention_pooling.yaml \
  --output_dir  /root/autodl-tmp/TwoTST/results/interpretability \
  --target_class 1 --max_samples 500
```
