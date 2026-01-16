"""
TST1 预训练脚本
使用ROI-level掩码策略预训练时序Transformer
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer_ts import TransformerTS, create_transformer_ts
from pretrain.mask_utils import mask_roi_level


class PretrainTSDataset(torch.utils.data.Dataset):
    """TST1预训练数据集"""
    
    def __init__(self, timeseries, normalize=True):
        """
        Args:
            timeseries: ndarray, shape (n_samples, n_timepoints, n_rois)
            normalize: 是否进行标准化
        """
        self.timeseries = timeseries.astype(np.float32)
        self.normalize = normalize
        
        if normalize:
            self.mean = np.mean(self.timeseries)
            self.std = np.std(self.timeseries) + 1e-8
    
    def __len__(self):
        return len(self.timeseries)
    
    def __getitem__(self, idx):
        ts = self.timeseries[idx].copy()
        
        if self.normalize:
            ts = (ts - self.mean) / self.std
        
        return torch.tensor(ts, dtype=torch.float32)


def train_epoch(model, train_loader, optimizer, device, mask_ratio=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        batch = batch.to(device)
        
        # 应用ROI-level掩码
        masked_batch, mask, target, roi_mask = mask_roi_level(batch, mask_ratio)
        
        # 前向传播
        optimizer.zero_grad()
        output = model(masked_batch, mode='pretrain')
        
        # 计算掩码位置的MSE损失
        loss = nn.functional.mse_loss(output[mask], target[mask])
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / num_batches


def validate(model, val_loader, device, mask_ratio=None):
    """验证"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            
            # 应用ROI-level掩码
            masked_batch, mask, target, roi_mask = mask_roi_level(batch, mask_ratio)
            
            # 前向传播
            output = model(masked_batch, mode='pretrain')
            
            # 计算掩码位置的MSE损失
            loss = nn.functional.mse_loss(output[mask], target[mask])
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def pretrain_transformer_ts(
    train_loader,
    val_loader,
    model_config=None,
    epochs=100,
    lr=1e-4,
    weight_decay=1e-4,
    mask_ratio=None,
    device='cuda',
    save_dir='checkpoints',
    log_dir=None
):
    """
    预训练TST1模型
    
    Args:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        model_config: 模型配置
        epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        mask_ratio: 掩码比例（None表示随机选择0.25或0.5）
        device: 设备
        save_dir: 保存目录
        log_dir: TensorBoard日志目录
    
    Returns:
        model: 训练好的模型
    """
    # 创建模型
    model = create_transformer_ts(model_config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 优化器和调度器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建TensorBoard日志目录
    if log_dir is None:
        log_dir = os.path.join(save_dir, '../logs/tst1')
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 训练循环
    best_val_loss = float('inf')
    best_epoch = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 40)
        
        # 训练
        train_loss = train_epoch(
            model, train_loader, optimizer, device, mask_ratio
        )
        
        # 验证
        val_loss = validate(model, val_loader, device, mask_ratio)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 更新学习率
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"LR: {current_lr:.6f}")
        
        # 只保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            checkpoint = {
                'epoch': epoch,
                'best_epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'config': model_config,
                'train_losses': train_losses.copy(),
                'val_losses': val_losses.copy()
            }
            torch.save(checkpoint, os.path.join(save_dir, 'tst1_best.pt'))
            print(f"Saved best model at epoch {epoch} (val_loss: {val_loss:.4f})")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(history, os.path.join(save_dir, 'tst1_history.pt'))
    
    # 关闭TensorBoard writer
    writer.close()
    
    print(f"\nTraining completed. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")
    print(f"TensorBoard logs saved to: {log_dir}")
    print(f"Run 'tensorboard --logdir={log_dir}' to view training curves")
    return model


def main(args):
    import pickle
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    with open(args.data_path, 'rb') as f:
        data = pickle.load(f)
    
    timeseries = data['timeseries']
    labels = data['labels']
    
    print(f"Timeseries shape: {timeseries.shape}")
    
    # 7:1:2划分数据集
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(labels))
    
    # 第一次划分：训练集 vs (验证集+测试集)
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.3, random_state=args.seed, stratify=labels
    )
    
    # 第二次划分：验证集 vs 测试集
    val_test_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=2/3, random_state=args.seed, stratify=val_test_labels
    )
    
    train_ts = timeseries[train_idx]
    val_ts = timeseries[val_idx]
    test_ts = timeseries[test_idx]
    
    print(f"Data split - Train: {len(train_ts)}, Val: {len(val_ts)}, Test: {len(test_ts)}")
    
    # 创建数据集和数据加载器
    train_dataset = PretrainTSDataset(train_ts)
    val_dataset = PretrainTSDataset(val_ts)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    
    # 模型配置
    model_config = {
        'n_rois': timeseries.shape[2],
        'emb_dim': args.emb_dim,
        'n_heads': args.n_heads,
        'n_layers': args.n_layers,
        'dim_feedforward': args.dim_feedforward,
        'dropout': args.dropout,
        'max_seq_len': timeseries.shape[1],
        'use_cls_token': True
    }
    
    # 预训练
    model = pretrain_transformer_ts(
        train_loader=train_loader,
        val_loader=val_loader,
        model_config=model_config,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        mask_ratio=args.mask_ratio,
        device=device,
        save_dir=args.save_dir,
        log_dir=args.log_dir
    )
    
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TST1 Pretraining')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, 
                        default='/root/workplace/exp/TwoTST/data/processed/processed_data.pkl')
    
    # 模型参数
    parser.add_argument('--emb_dim', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--mask_ratio', type=float, default=None,
                        help='Mask ratio (None for random 0.25/0.5)')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/checkpoints/tst1')
    parser.add_argument('--log_dir', type=str, 
                        default='/root/workplace/exp/TwoTST/logs/tst1',
                        help='TensorBoard log directory')
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)
