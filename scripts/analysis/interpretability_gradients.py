#!/usr/bin/env python3
"""
可解释性：基于梯度的脑区/连接重要性
用于回答「模型依据哪些脑区、哪些连接区分 ASD vs TC」。
输出：ROI 重要性、连接 (ROI_i, ROI_j) 重要性，可对接脑网络可视化或神经科学解读。
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def pcc_index_to_roi_pair(k, n_rois=200):
    """将 PCC 上三角线性索引 k 映射为 (roi_i, roi_j)。"""
    triu_i, triu_j = np.triu_indices(n_rois, k=1)
    if k < 0 or k >= len(triu_i):
        return None, None
    return int(triu_i[k]), int(triu_j[k])


def _load_run_experiment_module():
    """动态加载 run_experiment 模块（与 run_best_config_5fold_loso 一致）。"""
    import importlib.util
    path = PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"
    spec = importlib.util.spec_from_file_location("run_experiment", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_experiment"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_best_model_from_experiment(checkpoint_path, config_path, device):
    """
    从 run_experiment 保存的 best_model.pt 和对应 config 加载
    projection + attention_pooling 等最佳配置模型。
    """
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    run_experiment = _load_run_experiment_module()
    load_pretrained_models = run_experiment.load_pretrained_models
    load_contrastive_checkpoint = run_experiment.load_contrastive_checkpoint
    create_fusion_model = run_experiment.create_fusion_model
    create_classifier = run_experiment.create_classifier
    ProjectionFinetuneModel = run_experiment.ProjectionFinetuneModel

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = ckpt['model_state_dict']
    config = ckpt.get('config', config)

    # 构建与 run_experiment 一致的模型结构；权重全部从 best_model.pt 加载
    ckpt_root = Path("/root/autodl-tmp/TwoTST")
    if not (ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt").exists():
        ckpt_root = PROJECT_ROOT
    if config.get('pretrain', {}).get('use_pretrained'):
        config['pretrain']['tst1']['checkpoint'] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
        config['pretrain']['tst2']['checkpoint'] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")
    cont_ckpt = ckpt_root / "checkpoints" / "contrastive_checkpoint.pt"
    config['contrastive']['load_checkpoint'] = str(cont_ckpt) if cont_ckpt.exists() else None

    tst1, tst2 = load_pretrained_models(config, device)
    if config['contrastive']['enabled'] and config['contrastive'].get('load_checkpoint') and os.path.exists(config['contrastive']['load_checkpoint']):
        tst1, tst2, proj_head1, proj_head2 = load_contrastive_checkpoint(tst1, tst2, config, device)
    else:
        from pretrain.contrastive import ProjectionHead
        tst1_dim = config['model']['tst1'].get('emb_dim', 512)
        tst2_dim = config['model']['tst2'].get('d_model', 256)
        ph = config['contrastive'].get('proj_hidden_dim', 256)
        po = config['contrastive'].get('proj_output_dim', 128)
        proj_head1 = ProjectionHead(tst1_dim, ph, po).to(device)
        proj_head2 = ProjectionHead(tst2_dim, ph, po).to(device)

    for p in proj_head1.parameters():
        p.requires_grad = False
    for p in proj_head2.parameters():
        p.requires_grad = False

    proj_output_dim = config['contrastive'].get('proj_output_dim', 128)
    fusion = create_fusion_model(config, proj_output_dim, proj_output_dim, device)
    classifier_input_dim = proj_output_dim
    classifier = create_classifier(classifier_input_dim, config, device)
    model = ProjectionFinetuneModel(
        tst1, tst2, proj_head1, proj_head2, fusion, classifier,
        use_tst1=True, use_tst2=True, use_projection=True
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, config


def compute_connection_importance(model, timeseries, pcc_vectors, device, target_class=1):
    """
    对 PCC 输入计算梯度重要性（对 target_class 的 logit）。
    timeseries: (N, T, n_rois), pcc_vectors: (N, pcc_dim)
    返回: (pcc_dim,) 每个连接的平均绝对梯度（越大越重要）。
    """
    ts = torch.tensor(timeseries, dtype=torch.float32, device=device, requires_grad=True)
    pcc = torch.tensor(pcc_vectors, dtype=torch.float32, device=device, requires_grad=True)

    logits = model(ts, pcc)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = logits[:, target_class].sum()
    model.zero_grad()
    target.backward()

    if pcc.grad is None:
        return np.zeros(pcc_vectors.shape[1], dtype=np.float32)
    grad = pcc.grad.detach().cpu().numpy()
    return np.mean(np.abs(grad), axis=0)


def compute_roi_importance(model, timeseries, pcc_vectors, device, target_class=1):
    """
    对时序输入 (T, n_rois) 计算梯度，沿时间聚合得到每个 ROI 的重要性。
    timeseries: (N, T, n_rois)
    返回: (n_rois,) 每个 ROI 的平均绝对梯度（沿时间与 batch 聚合）。
    """
    ts = torch.tensor(timeseries, dtype=torch.float32, device=device, requires_grad=True)
    pcc = torch.tensor(pcc_vectors, dtype=torch.float32, device=device)

    logits = model(ts, pcc)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    target = logits[:, target_class].sum()
    model.zero_grad()
    target.backward()

    if ts.grad is None:
        return np.zeros(timeseries.shape[2], dtype=np.float32)
    grad = ts.grad.detach().cpu().numpy()  # (N, T, n_rois)
    return np.mean(np.abs(grad), axis=(0, 1))


def run_interpretability(
    data_path,
    checkpoint_path,
    config_path,
    output_dir,
    n_rois=200,
    pcc_dim=19900,
    target_class=1,
    max_samples=500,
    batch_size=32,
    device="cuda",
    top_k_connections=100,
    top_k_rois=50,
):
    """
    主流程：加载数据与模型，计算连接/ROI 重要性，保存结果与 top-k 列表。
    """
    os.makedirs(output_dir, exist_ok=True)

    import pickle
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']

    n_total = timeseries.shape[0]
    n_use = min(max_samples, n_total)
    indices = np.random.RandomState(42).permutation(n_total)[:n_use]
    timeseries = timeseries[indices]
    pcc_vectors = pcc_vectors[indices]

    print(f"Loading model from {checkpoint_path} ...")
    model, config = load_best_model_from_experiment(checkpoint_path, config_path, device)

    # 逐 batch 累积梯度重要性（平均）
    conn_importance = np.zeros(pcc_dim, dtype=np.float64)
    roi_importance = np.zeros(n_rois, dtype=np.float64)
    n_batches = 0

    for start in tqdm(range(0, n_use, batch_size), desc="Gradient importance"):
        end = min(start + batch_size, n_use)
        ts_batch = timeseries[start:end]
        pcc_batch = pcc_vectors[start:end]
        conn_importance += compute_connection_importance(
            model, ts_batch, pcc_batch, device, target_class=target_class
        )
        roi_importance += compute_roi_importance(
            model, ts_batch, pcc_batch, device, target_class=target_class
        )
        n_batches += 1

    conn_importance /= n_batches
    roi_importance /= n_batches

    # Top-k 连接（线性索引 -> (roi_i, roi_j)）
    triu_i, triu_j = np.triu_indices(n_rois, k=1)
    top_conn_idx = np.argsort(conn_importance)[::-1][:top_k_connections]
    top_connections = [
        {
            "roi_i": int(triu_i[k]),
            "roi_j": int(triu_j[k]),
            "importance": float(conn_importance[k]),
            "linear_index": int(k),
        }
        for k in top_conn_idx
    ]

    top_roi_idx = np.argsort(roi_importance)[::-1][:top_k_rois]
    top_rois = [
        {"roi_index": int(i), "importance": float(roi_importance[i])}
        for i in top_roi_idx
    ]

    results = {
        "target_class": target_class,
        "target_class_name": "ASD" if target_class == 1 else "TC",
        "n_samples_used": n_use,
        "connection_importance_mean": conn_importance.tolist(),
        "roi_importance_mean": roi_importance.tolist(),
        "top_k_connections": top_connections,
        "top_k_rois": top_rois,
        "n_rois": n_rois,
        "pcc_dim": pcc_dim,
    }

    out_json = os.path.join(output_dir, "interpretability_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存连接矩阵（n_rois x n_rois）便于脑网络可视化
    conn_matrix = np.zeros((n_rois, n_rois))
    conn_matrix[triu_i, triu_j] = conn_importance
    conn_matrix = conn_matrix + conn_matrix.T
    np.save(os.path.join(output_dir, "connection_importance_matrix.npy"), conn_matrix)
    np.save(os.path.join(output_dir, "roi_importance_vector.npy"), roi_importance)

    print(f"\nSaved: {out_json}")
    print(f"      {output_dir}/connection_importance_matrix.npy (n_rois x n_rois)")
    print(f"      {output_dir}/roi_importance_vector.npy (n_rois,)")
    print(f"\nTop 10 connections (model-driven importance for {results['target_class_name']}):")
    for c in top_connections[:10]:
        print(f"  ROI {c['roi_i']} <-> ROI {c['roi_j']}: importance = {c['importance']:.6f}")
    print(f"\nTop 10 ROIs:")
    for r in top_rois[:10]:
        print(f"  ROI {r['roi_index']}: importance = {r['importance']:.6f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Interpretability: connection and ROI importance from gradients")
    parser.add_argument("--data_path", type=str,
                        default=str(PROJECT_ROOT / "data" / "processed" / "processed_data.pkl"),
                        help="Processed data pkl (timeseries, pcc_vectors, labels)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="best_model.pt from run_experiment (e.g. projection_fusion_attention_pooling)")
    parser.add_argument("--config", type=str, default=None,
                        help="Config yaml (default: group7_projection_fusion_attention_pooling.yaml)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: results/interpretability)")
    parser.add_argument("--n_rois", type=int, default=200)
    parser.add_argument("--pcc_dim", type=int, default=19900)
    parser.add_argument("--target_class", type=int, default=1, help="1=ASD, 0=TC")
    parser.add_argument("--max_samples", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k_connections", type=int, default=100)
    parser.add_argument("--top_k_rois", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.config is None:
        args.config = str(PROJECT_ROOT / "configs" / "experiments" / "group7_projection_fusion_attention_pooling.yaml")
    if args.output_dir is None:
        args.output_dir = str(PROJECT_ROOT / "results" / "interpretability")

    run_interpretability(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        n_rois=args.n_rois,
        pcc_dim=args.pcc_dim,
        target_class=args.target_class,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=args.device,
        top_k_connections=args.top_k_connections,
        top_k_rois=args.top_k_rois,
    )


if __name__ == "__main__":
    main()
