#!/usr/bin/env python3
"""
可解释性结果可视化
读取 connection_importance_matrix.npy 与 roi_importance_vector.npy，绘制：
1. 连接重要性矩阵热图（200×200）
2. Top-k ROI 重要性条形图（可选 CC200/AAL 脑区学名标签）
3. 可选：仅上三角连接热图（避免重复）
4. Top 连接条形图（可选脑区学名）
"""

import os
import sys
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
# 默认使用 CC200 官方坐标/标签文件（与 Craddock 200 一致）
DEFAULT_LABELS_PATH = os.path.join(PROJECT_ROOT, 'data', 'labels', 'cc200_coordinates.json')


def load_roi_labels_json(labels_path):
    """从 cc200_coordinates.json 加载：id -> 显示名（Region N (lobe_hemisphere)）。"""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    labels = {}
    for item in data:
        idx = item.get('id')
        if idx is None:
            continue
        name = (item.get('name') or '').strip() or f'Region_{idx + 1}'
        lobe = (item.get('lobe') or '').strip()
        hem = (item.get('hemisphere') or '').strip()
        if lobe and hem:
            labels[int(idx)] = f"{name} ({lobe}_{hem})"
        else:
            labels[int(idx)] = name
    return labels


def load_roi_labels_csv(labels_path):
    """加载 ROI 标签 CSV，返回 roi_index -> region_name_en 的字典。"""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    labels = {}
    with open(labels_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            idx = row.get('roi_index')
            name = (row.get('region_name_en') or '').strip()
            if idx != '' and idx is not None:
                try:
                    labels[int(idx)] = name or f'ROI_{idx}'
                except ValueError:
                    pass
    return labels


def load_roi_labels(labels_path):
    """根据扩展名从 JSON（cc200_coordinates）或 CSV 加载 ROI 标签。"""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    if labels_path.lower().endswith('.json'):
        return load_roi_labels_json(labels_path)
    return load_roi_labels_csv(labels_path)

# 中文字体（若无可回退到默认）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def load_data(result_dir):
    """加载 .npy 与可选 .json。"""
    conn_path = os.path.join(result_dir, 'connection_importance_matrix.npy')
    roi_path = os.path.join(result_dir, 'roi_importance_vector.npy')
    json_path = os.path.join(result_dir, 'interpretability_results.json')

    if not os.path.exists(conn_path) or not os.path.exists(roi_path):
        raise FileNotFoundError(
            f"Need {conn_path} and {roi_path}. Run interpretability_gradients.py first."
        )

    conn_matrix = np.load(conn_path)
    roi_vector = np.load(roi_path)

    top_connections = None
    top_rois = None
    if os.path.exists(json_path):
        import json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        top_connections = data.get('top_k_connections', [])[:20]
        top_rois = data.get('top_k_rois', [])[:30]

    return conn_matrix, roi_vector, top_connections, top_rois


def plot_connection_heatmap(conn_matrix, save_path, top_k_connections=None, figsize=(10, 8)):
    """绘制连接重要性矩阵热图。"""
    fig, ax = plt.subplots(figsize=figsize)

    # 对称矩阵，直接 imshow；上三角已包含全部信息，可 mask 下三角避免重复（可选）
    vmax = np.percentile(conn_matrix, 99) if np.any(conn_matrix > 0) else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(conn_matrix, cmap='YlOrRd', aspect='equal', vmin=0, vmax=vmax)

    ax.set_xlabel('ROI index')
    ax.set_ylabel('ROI index')
    ax.set_title('Connection importance (gradient, ASD class)')
    plt.colorbar(im, ax=ax, label='Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_connection_heatmap_uppertri(conn_matrix, save_path, figsize=(10, 8)):
    """仅绘制上三角（避免对角线及重复），更清晰。"""
    n = conn_matrix.shape[0]
    mask = np.triu(np.ones((n, n)), k=1)  # 上三角 k=1 不含对角线
    masked = np.where(mask > 0, conn_matrix, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    vmax = np.nanpercentile(masked, 99) if np.any(~np.isnan(masked)) else 1.0
    vmax = max(vmax, 1e-6)
    im = ax.imshow(masked, cmap='YlOrRd', aspect='equal', vmin=0, vmax=vmax)

    ax.set_xlabel('ROI index')
    ax.set_ylabel('ROI index')
    ax.set_title('Connection importance (upper triangle, ASD)')
    plt.colorbar(im, ax=ax, label='Importance')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_roi_bars(roi_vector, save_path, top_k=30, figsize=(10, 5), roi_labels=None):
    """绘制 Top-k ROI 重要性条形图。roi_labels: roi_index -> region_name_en，用于显示脑区学名。"""
    order = np.argsort(roi_vector)[::-1][:top_k]
    indices = order
    values = roi_vector[order]

    if roi_labels:
        tick_labels = [f"{roi_labels.get(int(i), i)} ({i})" for i in indices]
    else:
        tick_labels = [str(i) for i in indices]

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(indices)), values, color='steelblue', edgecolor='navy', alpha=0.8)
    ax.set_xticks(range(len(indices)))
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=7)
    ax.set_xlabel('ROI (region name, index)')
    ax.set_ylabel('Importance')
    ax.set_title(f'Top-{top_k} ROI importance (gradient, ASD class)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_top_connections_bars(top_connections, save_path, top_k=20, figsize=(10, 5), roi_labels=None):
    """若有 top_k_connections 列表，绘制 Top 连接条形图；roi_labels 可选，用于显示脑区学名。"""
    if not top_connections or len(top_connections) == 0:
        return
    top = top_connections[:top_k]
    if roi_labels:
        labels = [
            f"{roi_labels.get(c['roi_i'], c['roi_i'])}–{roi_labels.get(c['roi_j'], c['roi_j'])}"
            for c in top
        ]
    else:
        labels = [f"{c['roi_i']}-{c['roi_j']}" for c in top]
    values = [c['importance'] for c in top]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(range(len(labels)), values, color='coral', edgecolor='darkred', alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('Importance')
    ax.set_ylabel('Connection (region i – region j)')
    ax.set_title(f'Top-{len(top)} connections (ASD)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize interpretability results')
    parser.add_argument('--result_dir', type=str,
                        default='/root/autodl-tmp/TwoTST/results/interpretability',
                        help='Directory containing .npy and interpretability_results.json')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Where to save figures (default: same as result_dir)')
    parser.add_argument('--top_rois', type=int, default=30)
    parser.add_argument('--top_connections_bars', type=int, default=20)
    parser.add_argument('--labels', type=str, default=DEFAULT_LABELS_PATH,
                        help='ROI 标签文件：.json 使用 cc200_coordinates.json，.csv 使用 roi_index,region_name_en,lobe')
    args = parser.parse_args()

    output_dir = args.output_dir or args.result_dir
    os.makedirs(output_dir, exist_ok=True)

    roi_labels = load_roi_labels(args.labels)
    if roi_labels:
        print(f"Loaded {len(roi_labels)} ROI labels from {args.labels}")

    conn_matrix, roi_vector, top_connections, top_rois = load_data(args.result_dir)

    # 1. 连接矩阵热图（全矩阵）
    plot_connection_heatmap(
        conn_matrix,
        os.path.join(output_dir, 'connection_importance_heatmap.png'),
    )

    # 2. 连接矩阵热图（仅上三角）
    plot_connection_heatmap_uppertri(
        conn_matrix,
        os.path.join(output_dir, 'connection_importance_heatmap_uppertri.png'),
    )

    # 3. Top-k ROI 条形图（含脑区学名时更利于与医学研究对照）
    plot_roi_bars(
        roi_vector,
        os.path.join(output_dir, 'roi_importance_bars.png'),
        top_k=args.top_rois,
        roi_labels=roi_labels,
    )

    # 4. Top 连接条形图（若有 JSON；含脑区学名）
    plot_top_connections_bars(
        top_connections,
        os.path.join(output_dir, 'top_connections_bars.png'),
        top_k=args.top_connections_bars,
        roi_labels=roi_labels,
    )

    print(f"\nAll figures saved to: {output_dir}")


if __name__ == '__main__':
    main()
