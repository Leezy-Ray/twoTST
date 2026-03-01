"""
预训练入口脚本
顺序执行TST1和TST2的预训练
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pretrain.pretrain_ts import pretrain_transformer_ts, PretrainTSDataset
from pretrain.pretrain_fc import pretrain_transformer_fc, PretrainFCDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pickle


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(args):
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载配置（如果提供）
    if args.config:
        config = load_config(args.config)
    else:
        config = {}
    
    # 加载数据
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)

    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    subject_indices = data.get('subject_indices')
    site_ids = data.get('site_ids')

    print(f"Timeseries shape: {timeseries.shape}")
    print(f"PCC vectors shape: {pcc_vectors.shape}")

    # 受试者级划分：预训练仅在训练被试上进行，避免测试集信息泄漏
    if subject_indices is not None:
        from utils.splitters import get_subject_level_train_val_test_split
        train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(
            labels, subject_indices, site_ids=site_ids,
            train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=args.seed
        )
        print(f"Subject-level split (pre-training uses train only) - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices, test_size=0.3, random_state=args.seed, stratify=labels
        )
        val_test_labels = labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=2/3, random_state=args.seed, stratify=val_test_labels
        )
        print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # Phase 1: TST1 预训练
    if args.pretrain_tst1:
        print("\n" + "=" * 60)
        print("Phase 1: TST1 Pretraining (Time Series Transformer)")
        print("=" * 60)
        
        # 创建数据加载器
        train_ts_dataset = PretrainTSDataset(timeseries[train_idx])
        val_ts_dataset = PretrainTSDataset(timeseries[val_idx])
        
        train_ts_loader = DataLoader(
            train_ts_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_ts_loader = DataLoader(
            val_ts_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # TST1 模型配置
        tst1_config = config.get('tst1', {})
        tst1_config.update({
            'n_rois': timeseries.shape[2],
            'max_seq_len': timeseries.shape[1],
        })
        
        # 预训练
        from pretrain.pretrain_ts import pretrain_transformer_ts
        pretrain_transformer_ts(
            train_loader=train_ts_loader,
            val_loader=val_ts_loader,
            model_config=tst1_config,
            epochs=args.tst1_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mask_ratio=args.tst1_mask_ratio,
            device=device,
            save_dir=os.path.join(args.save_dir, 'tst1'),
            log_dir=os.path.join(args.log_dir, 'tst1')
        )
    
    # Phase 2: TST2 预训练
    if args.pretrain_tst2:
        print("\n" + "=" * 60)
        print("Phase 2: TST2 Pretraining (Functional Connectivity Transformer)")
        print("=" * 60)
        
        # 创建数据加载器
        train_fc_dataset = PretrainFCDataset(pcc_vectors[train_idx])
        val_fc_dataset = PretrainFCDataset(pcc_vectors[val_idx])
        
        train_fc_loader = DataLoader(
            train_fc_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True
        )
        val_fc_loader = DataLoader(
            val_fc_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True
        )
        
        # TST2 模型配置
        tst2_config = config.get('tst2', {})
        tst2_config.update({
            'pcc_dim': pcc_vectors.shape[1],
        })
        
        # 预训练
        from pretrain.pretrain_fc import pretrain_transformer_fc
        pretrain_transformer_fc(
            train_loader=train_fc_loader,
            val_loader=val_fc_loader,
            model_config=tst2_config,
            epochs=args.tst2_epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            mask_ratio=args.tst2_mask_ratio,
            device=device,
            save_dir=os.path.join(args.save_dir, 'tst2'),
            log_dir=os.path.join(args.log_dir, 'tst2')
        )
    
    print("\n" + "=" * 60)
    print("Pretraining completed!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwoTST Pretraining')
    
    # 数据参数
    parser.add_argument('--data_path', type=str,
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    
    # 预训练选项
    parser.add_argument('--pretrain_tst1', action='store_true', default=True,
                        help='Pretrain TST1')
    parser.add_argument('--pretrain_tst2', action='store_true', default=True,
                        help='Pretrain TST2')
    parser.add_argument('--no_tst1', action='store_false', dest='pretrain_tst1',
                        help='Skip TST1 pretraining')
    parser.add_argument('--no_tst2', action='store_false', dest='pretrain_tst2',
                        help='Skip TST2 pretraining')
    
    # TST1 参数
    parser.add_argument('--tst1_epochs', type=int, default=100)
    parser.add_argument('--tst1_mask_ratio', type=float, default=None,
                        help='TST1 mask ratio (None for random 0.25/0.5)')
    
    # TST2 参数
    parser.add_argument('--tst2_epochs', type=int, default=100)
    parser.add_argument('--tst2_mask_ratio', type=float, default=0.15)
    
    # 通用训练参数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints')
    parser.add_argument('--log_dir', type=str,
                        default='/root/workplace/exp/TwoTST/logs',
                        help='TensorBoard log directory')
    
    args = parser.parse_args()
    main(args)
