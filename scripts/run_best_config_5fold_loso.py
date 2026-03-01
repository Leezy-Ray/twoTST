#!/usr/bin/env python3
"""
最佳配置的 5-fold CV 与 LOSO 评估
配置：pretrain TST1/TST2 → 对比学习 projection → attention_pooling 融合，微调 unfrozen
用于回应审稿人：对主实验做统计验证（mean±std、95% CI）与跨站点泛化（LOSO）。
"""

import os
import sys
import argparse
import json
import pickle
import numpy as np
import torch
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 动态加载 run_experiment 模块（避免包结构依赖）
import importlib.util
_run_exp_path = PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"
_spec = importlib.util.spec_from_file_location("run_experiment", _run_exp_path)
_run_experiment = importlib.util.module_from_spec(_spec)
sys.modules["run_experiment"] = _run_experiment
_spec.loader.exec_module(_run_experiment)

load_config = _run_experiment.load_config
load_data = _run_experiment.load_data
load_pretrained_models = _run_experiment.load_pretrained_models
load_contrastive_checkpoint = _run_experiment.load_contrastive_checkpoint
create_fusion_model = _run_experiment.create_fusion_model
create_classifier = _run_experiment.create_classifier
ProjectionFinetuneModel = _run_experiment.ProjectionFinetuneModel
run_finetune = _run_experiment.run_finetune

from utils.splitters import get_subject_level_fold_splits, get_loso_fold_splits
from utils.metrics import bootstrap_confidence_interval, get_reproducibility_info


def get_splits(config, timeseries, pcc_vectors, labels, subject_indices, site_ids, eval_protocol, n_folds, seed):
    """获取 K-fold 或 LOSO 划分（受试者级）。"""
    if eval_protocol == "loso":
        if site_ids is None:
            raise ValueError("LOSO requires site_ids in data.")
        splits = get_loso_fold_splits(
            labels, subject_indices, site_ids,
            val_ratio=0.15, seed=seed
        )
        print(f"LOSO: {len(splits)} sites (folds)")
    else:
        splits = get_subject_level_fold_splits(
            labels, subject_indices, site_ids=site_ids,
            n_splits=n_folds, val_ratio=0.15, seed=seed
        )
        print(f"5-fold CV: {len(splits)} folds")
    return splits


def run_one_fold(
    fold_idx, split, timeseries, pcc_vectors, labels, subject_indices,
    config, device, save_dir, log_dir, seed, subject_agg_strategy
):
    """跑单折：构建 projection+attention_pooling 模型并微调，返回 test 指标。"""
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]
    test_idx = split["test_idx"]

    train_data = (
        timeseries[train_idx],
        pcc_vectors[train_idx],
        labels[train_idx],
    )
    val_data = (
        timeseries[val_idx],
        pcc_vectors[val_idx],
        labels[val_idx],
    )
    test_data = (
        timeseries[test_idx],
        pcc_vectors[test_idx],
        labels[test_idx],
    )

    # 路径：优先数据盘
    ckpt_root = Path("/root/autodl-tmp/TwoTST")
    if not (ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt").exists():
        ckpt_root = PROJECT_ROOT
    config["pretrain"]["tst1"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")
    cont_ckpt = ckpt_root / "checkpoints" / "contrastive_checkpoint.pt"
    if cont_ckpt.exists():
        config["contrastive"]["load_checkpoint"] = str(cont_ckpt)
    else:
        config["contrastive"]["load_checkpoint"] = None

    # 模型
    tst1, tst2 = load_pretrained_models(config, device)
    if not config["contrastive"]["enabled"]:
        raise ValueError("Best config requires contrastive (projection).")
    tst1, tst2, proj_head1, proj_head2 = load_contrastive_checkpoint(tst1, tst2, config, device)
    for p in proj_head1.parameters():
        p.requires_grad = False
    for p in proj_head2.parameters():
        p.requires_grad = False

    proj_output_dim = config["contrastive"].get("proj_output_dim", 128)
    fusion_config = config["fusion"]
    fusion = create_fusion_model(config, proj_output_dim, proj_output_dim, device)
    if fusion_config["type"] == "attention_pooling":
        classifier_input_dim = proj_output_dim
    else:
        classifier_input_dim = proj_output_dim
    classifier = create_classifier(classifier_input_dim, config, device)
    model = ProjectionFinetuneModel(
        tst1, tst2, proj_head1, proj_head2, fusion, classifier,
        use_tst1=True, use_tst2=True, use_projection=True
    )

    # 确保 TensorBoard 日志目录存在，避免 FileNotFoundError
    fold_log_dir = os.path.join(log_dir, f"fold_{fold_idx}")
    os.makedirs(fold_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=fold_log_dir)
    results, _ = run_finetune(
        model, train_data, val_data, test_data, config, device, writer,
        subject_indices=subject_indices,
        test_idx=test_idx,
        subject_agg_strategy=subject_agg_strategy,
    )
    writer.close()

    if save_dir:
        fold_ckpt = os.path.join(save_dir, f"fold_{fold_idx}_results.json")
        with open(fold_ckpt, "w") as f:
            json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Best config (projection+attention_pooling) 5-fold / LOSO")
    parser.add_argument("--config", type=str,
                        default=str(PROJECT_ROOT / "configs" / "experiments" / "group7_projection_fusion_attention_pooling.yaml"),
                        help="Path to best config yaml")
    parser.add_argument("--eval_protocol", type=str, default="kfold", choices=["kfold", "loso"])
    parser.add_argument("--n_folds", type=int, default=5, help="For kfold only")
    parser.add_argument("--subject_agg_strategy", type=str, default="majority_vote", choices=["prob_mean", "majority_vote"])
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Save per-fold results and summary here (default: results/best_config_5fold or _loso)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config["training"].get("seed", args.seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config["training"].get("device", "cuda") if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    timeseries, pcc_vectors, labels, subject_indices, site_ids = load_data(config)
    if subject_indices is None:
        subject_indices = np.arange(len(labels))

    splits = get_splits(
        config, timeseries, pcc_vectors, labels, subject_indices, site_ids,
        args.eval_protocol, args.n_folds, seed
    )

    suffix = "5fold" if args.eval_protocol == "kfold" else "loso"
    save_dir = args.save_dir or str(PROJECT_ROOT / "results" / f"best_config_{suffix}")
    log_dir = str(Path(save_dir) / "logs")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    all_results = []
    for fold_idx, split in enumerate(splits):
        test_site = split.get("test_site", f"fold{fold_idx}")
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{len(splits)} (test_site={test_site})")
        print("="*60)
        res = run_one_fold(
            fold_idx, split, timeseries, pcc_vectors, labels, subject_indices,
            config, device, save_dir, log_dir, seed, args.subject_agg_strategy
        )
        all_results.append(res)
        print(f"  Fold {fold_idx + 1} Test AUC: {res['auc']:.4f}")

    # 汇总 mean ± std, 95% CI
    metric_names = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
    summary = {"protocol": args.eval_protocol, "n_folds": len(splits), "seed": seed}
    for name in metric_names:
        vals = [r.get(name) for r in all_results if r.get(name) is not None]
        if vals:
            mean_v, std_v, lower, upper = bootstrap_confidence_interval(
                vals, n_bootstrap=1000, ci=0.95, seed=seed
            )
            summary[name] = {"mean": mean_v, "std": std_v, "ci95_lower": lower, "ci95_upper": upper}

    print("\n" + "="*60)
    print(f"Best config (projection + attention_pooling) — {args.eval_protocol.upper()}")
    print("="*60)
    for name in metric_names:
        if name in summary and isinstance(summary[name], dict):
            s = summary[name]
            print(f"  {name:12s}: {s['mean']:.4f} ± {s['std']:.4f}  [{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]")
    print("="*60)

    summary["all_folds"] = all_results
    summary["reproducibility"] = get_reproducibility_info()
    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=float)
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
