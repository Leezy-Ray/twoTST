# 可解释性与神经科学价值

针对审稿意见「可解释性与神经科学价值不足」，本目录说明如何从训练好的模型中**得到“哪些脑区、哪些连接”对 ASD/TC 分类重要**，用于支撑“连接异常/连接不足”的神经科学解读。

---

## 一、可解释性思路概览

| 方法 | 含义 | 输出 | 脚本/位置 |
|------|------|------|-----------|
| **梯度重要性（连接）** | 对“ASD 类 logit”关于 **PCC 输入**的梯度，绝对值大表示该连接对判为 ASD 越重要 | 每条边 (ROI_i, ROI_j) 一个重要性分数；可排 top-k 异常/关键连接 | `scripts/analysis/interpretability_gradients.py` |
| **梯度重要性（脑区）** | 对“ASD 类 logit”关于 **时序输入 (T×n_rois)** 的梯度，沿时间聚合得到每个 ROI 的重要性 | 每个 ROI 一个重要性分数；可排 top-k 关键脑区 | 同上 |
| **数据驱动连接差异** | 比较 ASD 组 vs TC 组 **平均 PCC** 的差异，差异大的连接视为“组间异常” | 组间差异最大的 top-k 连接（不依赖模型） | `scripts/analysis/analyze_attention.py` → `analyze_abnormal_connections` |
| **交叉注意力（若用 cross_attention 融合）** | 时序流与连接流之间的注意力权重 | 双流谁更被关注（非逐连接） | `scripts/analysis/analyze_attention.py` |

推荐**同时做**：  
- **模型驱动**：梯度重要性 → “模型依据哪些连接/脑区判断 ASD”；  
- **数据驱动**：PCC 组间差异 → “哪些连接在 ASD 与 TC 间差异大”。  
两者结合可写为“模型关注到的连接与文献/组间差异一致”，增强神经科学价值。

---

## 二、梯度重要性（连接 + 脑区）——推荐主流程

### 2.1 作用

- **连接重要性**：对 PCC 向量（上三角拉直）的每个元素求 ∂(logit_ASD)/∂(pcc_k)，取绝对值并平均过样本，得到每条边 (ROI_i, ROI_j) 的重要性。**重要性高 = 该连接对“判为 ASD”影响大**，可解释为“与该分类相关的连接异常/连接不足”的候选。
- **脑区重要性**：对时序 (T, n_rois) 求梯度，沿时间与样本平均，得到每个 ROI 的重要性。**重要性高 = 该脑区的时间动态对分类贡献大**。

### 2.2 运行方式

需已有一个 **run_experiment** 保存的 `best_model.pt`（例如 projection + attention_pooling 最佳配置）：

```bash
cd /root/workplace/exp/TwoTST

python scripts/analysis/interpretability_gradients.py \
  --data_path   data/processed/processed_data.pkl \
  --checkpoint  /path/to/best_model.pt \
  --config      configs/experiments/group7_projection_fusion_attention_pooling.yaml \
  --output_dir  results/interpretability \
  --target_class 1 \
  --max_samples 500 \
  --top_k_connections 100 \
  --top_k_rois 50
```

- `--target_class 1`：对“ASD”类 logit 做梯度（1=ASD，0=TC）。  
- 输出见下节。

### 2.3 输出文件

| 文件 | 含义 |
|------|------|
| `interpretability_results.json` | 全量连接/ROI 重要性向量、top-k 连接列表（含 roi_i, roi_j, importance）、top-k ROI 列表 |
| `connection_importance_matrix.npy` | (n_rois, n_rois) 矩阵，便于脑网络可视化（如 BrainNetViewer、nilearn） |
| `roi_importance_vector.npy` | (n_rois,) 各 ROI 重要性，可映射到 atlas 做脑区图 |

**论文/回复中可写**：  
“为评估神经科学可解释性，我们对 ASD 类 logit 关于 PCC 输入与时序输入求梯度，得到每条边和每个 ROI 的重要性；Top-k 连接/脑区与既往 ASD 文献中报道的异常脑区/连接具有一致性（可在此处引用具体 ROI 编号或 atlas 名称）。”

---

## 三、数据驱动：ASD vs TC 连接差异（已有脚本）

`scripts/analysis/analyze_attention.py` 中的 `analyze_abnormal_connections`：

- 输入：PCC 向量与标签。
- 做法：分别对 ASD/TC 求平均 PCC，相减得差异向量，转为 (n_rois, n_rois) 矩阵，取上三角绝对值最大的 top-k 条边。
- 输出：`top_k_connections` 为 (roi_i, roi_j, diff)，表示**组间连接强度差异**，可与梯度重要性对比。

运行示例（若使用滑窗 + cross_attention 等）：

```bash
python scripts/analysis/analyze_attention.py \
  --checkpoint  checkpoints_sw/finetune/sw_baseline_cross_attention/best_model.pt \
  --results_json ... \
  --output_dir  data/extracted/attention_analysis
```

该脚本当前面向 DualStreamModel + 窗口数据；若改用非滑窗 + projection 模型，需用上面梯度脚本得到“模型视角”的连接重要性，本脚本仍可用于“数据视角”的组间差异。

---

## 四、从“重要性”到“连接异常/连接不足”的表述

1. **连接重要性高**：可表述为“模型在区分 ASD 时**更依赖**这些连接”，对应文献中“异常连接”或“关键连接”的候选；若与组间 PCC 差异一致，可写“模型学到的关键连接与 ASD–TC 组间差异一致”。
2. **结合 PCC 符号**：对 top 连接查看 ASD 平均 PCC 与 TC 平均 PCC 的差：若 ASD < TC，可表述为“**连接不足**”；若 ASD > TC，可表述为“**连接过强**”。梯度只给“重要程度”，不直接给方向；方向需用数据驱动分析或对梯度×输入做简单统计。
3. **ROI 重要性**：可表述为“这些脑区的**时间动态**对分类贡献更大”，对应“关键脑区”；可与 AAL/CC200 等 atlas 对应，写出脑区名称，增强可读性。

---

## 五、与审稿意见的对应

- **“可解释性不足”**：在 Methods 中增加“基于梯度的连接与脑区重要性分析”小节，在 Results 中给出 top 连接/脑区表或图，并在 Discussion 中与 ASD 文献对比。  
- **“神经科学价值不足”**：明确写出“模型可识别出与 ASD 相关的关键脑区与连接，与既往报道的 XX 脑区/连接一致”，并引用 atlas（如 CC200）和 1–2 篇 ASD 影像学文献。

---

## 六、简要流程小结

1. 用 **interpretability_gradients.py** 对最佳模型跑梯度重要性 → 得到连接矩阵与 ROI 向量、top-k 列表。  
2. 用 **analyze_attention.py**（或自行对 PCC 按组求平均、做差）得到 **组间连接差异** top-k。  
3. 将梯度 top 连接与组间差异、文献中的异常脑区/连接对照，写成可解释性与神经科学价值段落，并补充相应图表。
