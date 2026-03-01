"""
微调训练脚本
加载预训练权重，进行下游分类任务
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
import pickle

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.dual_stream import DualStreamModel, create_dual_stream_model
from pretrain.contrastive import ContrastiveWrapper
from utils.data_loader import TwoTSTDataset, get_finetune_loaders
from utils.metrics import (
    aggregate_window_predictions_to_subject_level,
    bootstrap_confidence_interval,
    get_reproducibility_info,
)


def get_metrics(y_true, y_pred, y_prob=None):
    """计算评估指标"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
    
    # 混淆矩阵指标
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    return metrics


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for batch in train_loader:
        timeseries = batch['timeseries'].to(device)
        pcc_vector = batch['pcc_vector'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits = model(timeseries, pcc_vector)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(train_loader)
    acc = accuracy_score(all_labels, all_preds)
    
    return avg_loss, acc


def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(timeseries, pcc_vector)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    metrics = get_metrics(all_labels, all_preds, all_probs)
    
    return avg_loss, metrics


def train_with_contrastive(model, contrastive_module, train_loader, 
                           optimizer, device, epochs=10):
    """对比学习微调"""
    model.train()
    contrastive_module.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Contrastive {epoch}/{epochs}")
        
        for batch in pbar:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)
            
            optimizer.zero_grad()
            
            # 获取特征
            h_ts, h_fc = model.get_features(timeseries, pcc_vector)
            
            # 对比损失
            loss, _, _ = contrastive_module(h_ts, h_fc)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        print(f"Contrastive Epoch {epoch}: Loss = {total_loss / len(train_loader):.4f}")


def finetune_fold(
    train_loader,
    val_loader,
    test_loader,
    model_config,
    tst1_checkpoint=None,
    tst2_checkpoint=None,
    use_contrastive=False,
    contrastive_epochs=10,
    epochs=100,
    lr=5e-5,
    weight_decay=1e-4,
    device='cuda',
    save_dir=None,
    fold_idx=0,
    split_info=None,
    subject_agg_strategy='prob_mean',
):
    """
    单折微调训练
    """
    # 创建模型
    model = create_dual_stream_model(**model_config).to(device)
    
    # 加载预训练权重
    if tst1_checkpoint and os.path.exists(tst1_checkpoint):
        model.load_pretrained_tst1(tst1_checkpoint, strict=False)
        print(f"Loaded TST1 pretrained weights from {tst1_checkpoint}")
    
    if tst2_checkpoint and os.path.exists(tst2_checkpoint):
        model.load_pretrained_tst2(tst2_checkpoint, strict=False)
        print(f"Loaded TST2 pretrained weights from {tst2_checkpoint}")
    
    # 对比学习阶段（可选）
    if use_contrastive:
        print("\n--- Contrastive Learning Phase ---")
        contrastive_module = ContrastiveWrapper(
            dim_ts=model.dim_ts,
            dim_fc=model.dim_fc,
            temperature=0.07
        ).to(device)
        
        # 只优化编码器和投影头
        model.freeze_encoders()
        model.unfreeze_encoders()  # 解冻进行对比学习
        
        contrastive_params = list(model.parameters()) + list(contrastive_module.parameters())
        contrastive_optimizer = torch.optim.Adam(contrastive_params, lr=lr * 10)
        
        train_with_contrastive(
            model, contrastive_module, train_loader,
            contrastive_optimizer, device, contrastive_epochs
        )
    
    # 微调阶段
    print("\n--- Finetuning Phase ---")
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        # 验证
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        
        scheduler.step()
        
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Loss={val_loss:.4f}, Val Acc={val_metrics['accuracy']:.4f}")
        
        # 保存最佳模型
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_model_state = model.state_dict().copy()
    
    # 加载最佳模型
    model.load_state_dict(best_model_state)

    # 测试：收集窗口级预测
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)
            labels = batch['label'].to(device)
            logits = model(timeseries, pcc_vector)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # 受试者级汇总（滑窗时避免窗口级指标虚高）
    if split_info and split_info.get('subject_indices') is not None and split_info.get('test_idx') is not None:
        test_idx = split_info['test_idx']
        subj_indices = split_info['subject_indices']
        n_test_samples = len(test_idx)
        n_test_subjects = len(np.unique(subj_indices[test_idx]))
        if n_test_samples > n_test_subjects:  # 存在一个受试者多窗口
            yt, yp, yprob = aggregate_window_predictions_to_subject_level(
                all_labels, all_preds, all_probs, test_idx, subj_indices, strategy=subject_agg_strategy
            )
            test_metrics = get_metrics(yt, yp, yprob)
            print(f"\nFold {fold_idx} Results (subject-level):")
        else:
            test_metrics = get_metrics(all_labels, all_preds, all_probs)
            print(f"\nFold {fold_idx} Results (window-level):")
    else:
        test_metrics = get_metrics(all_labels, all_preds, all_probs)
        print(f"\nFold {fold_idx} Results:")
    print(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Precision: {test_metrics['precision']:.4f}")
    print(f"  Recall: {test_metrics['recall']:.4f}")
    print(f"  F1: {test_metrics['f1']:.4f}")
    print(f"  AUC: {test_metrics.get('auc', 0):.4f}")
    
    # 保存模型
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': test_metrics,
            'config': model_config
        }, os.path.join(save_dir, f'fold{fold_idx}_best.pt'))
    
    return test_metrics


def main(args):
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 确定折数：LOSO 时由站点数决定
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    n_folds = args.n_folds
    if getattr(args, 'eval_protocol', 'kfold') == 'loso':
        site_ids = data.get('site_ids')
        if site_ids is not None:
            n_folds = len(np.unique(site_ids))
            print(f"LOSO: {n_folds} sites, {n_folds} folds")
        else:
            raise ValueError("LOSO requires site_ids in data; use --eval_protocol kfold for data without sites")

    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']

    # 使用 get_finetune_loaders 实现受试者级 K 折或 LOSO
    print("Loading data...")
    train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_folds=args.n_folds if getattr(args, 'eval_protocol', 'kfold') != 'loso' else n_folds,
        fold_idx=0,
        val_ratio=0.15,
        seed=args.seed,
        use_subject_level_split=getattr(args, 'use_subject_level_split', True),
        eval_protocol=getattr(args, 'eval_protocol', 'kfold'),
    )

    model_config = {
        'n_rois': timeseries.shape[2],
        'time_points': timeseries.shape[1],
        'pcc_dim': pcc_vectors.shape[1],
        'tst1_emb_dim': args.tst1_emb_dim,
        'tst2_d_model': args.tst2_d_model,
        'fusion_type': args.fusion_type,
        'num_classes': 2,
        'dropout': args.dropout
    }

    all_metrics = []

    for fold_idx in range(n_folds):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}")
        print('='*60)

        train_loader, val_loader, test_loader, split_info = get_finetune_loaders(
            args.data_path,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            n_folds=args.n_folds if getattr(args, 'eval_protocol', 'kfold') != 'loso' else n_folds,
            fold_idx=fold_idx,
            val_ratio=0.15,
            seed=args.seed,
            use_subject_level_split=getattr(args, 'use_subject_level_split', True),
            eval_protocol=getattr(args, 'eval_protocol', 'kfold'),
        )

        fold_metrics = finetune_fold(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model_config=model_config,
            tst1_checkpoint=args.tst1_checkpoint,
            tst2_checkpoint=args.tst2_checkpoint,
            use_contrastive=args.use_contrastive,
            contrastive_epochs=args.contrastive_epochs,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device,
            save_dir=args.save_dir,
            fold_idx=fold_idx,
            split_info=split_info,
            subject_agg_strategy=getattr(args, 'subject_agg_strategy', 'prob_mean'),
        )

        all_metrics.append(fold_metrics)
    
    # 汇总结果（含 Bootstrap 95% CI）
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc']
    print(f"\n{'='*60}")
    print("Cross-Validation Results (mean ± std, 95% CI):")
    print('='*60)

    mean_metrics = {}
    std_metrics = {}
    ci_metrics = {}

    for metric in metric_names:
        values = [m.get(metric, 0) for m in all_metrics]
        mean_val, std_val, lower, upper = bootstrap_confidence_interval(
            values, n_bootstrap=1000, ci=0.95, seed=args.seed
        )
        mean_metrics[metric] = mean_val
        std_metrics[metric] = std_val
        ci_metrics[metric] = (lower, upper)
        display_name = 'AUC' if metric == 'auc' else metric.capitalize()
        print(f"{display_name:12s}: {mean_val:.4f} ± {std_val:.4f}  [{lower:.4f}, {upper:.4f}]")

    # 保存结果（含复现信息）
    if args.save_dir:
        repro_info = get_reproducibility_info()
        repro_info['seed'] = args.seed
        repro_info['n_folds'] = args.n_folds

        results = {
            'all_metrics': all_metrics,
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'ci_95_metrics': ci_metrics,
            'config': model_config,
            'args': vars(args),
            'reproducibility': repro_info,
        }
        with open(os.path.join(args.save_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(results, f, protocol=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwoTST Finetuning')
    
    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    
    # 预训练权重
    parser.add_argument('--tst1_checkpoint', type=str, default=None,
                        help='Path to TST1 pretrained checkpoint')
    parser.add_argument('--tst2_checkpoint', type=str, default=None,
                        help='Path to TST2 pretrained checkpoint')
    
    # 模型参数
    parser.add_argument('--tst1_emb_dim', type=int, default=512)
    parser.add_argument('--tst2_d_model', type=int, default=256)
    parser.add_argument('--fusion_type', type=str, default='cross_attention',
                        choices=['concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling'])
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 对比学习参数
    parser.add_argument('--use_contrastive', action='store_true',
                        help='Use contrastive learning before finetuning')
    parser.add_argument('--contrastive_epochs', type=int, default=10)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--n_folds', type=int, default=5)
    
    # 划分与评估
    parser.add_argument('--use_subject_level_split', action='store_true', default=True,
                        help='Use subject-level split to avoid data leakage (default: True)')
    parser.add_argument('--no_subject_level_split', action='store_false', dest='use_subject_level_split',
                        help='Disable subject-level split')
    parser.add_argument('--eval_protocol', type=str, default='kfold',
                        choices=['kfold', 'loso'],
                        help='kfold: 5-fold CV | loso: Leave-One-Site-Out for cross-site generalization')
    parser.add_argument('--subject_agg_strategy', type=str, default='prob_mean',
                        choices=['prob_mean', 'majority_vote'],
                        help='Subject-level aggregation for sliding window: prob_mean or majority_vote')

    # 其他参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints/finetune')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)
