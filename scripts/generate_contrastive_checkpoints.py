"""
生成对比学习checkpoint脚本
只运行对比学习部分，保存checkpoint供后续投影头微调实验使用
"""

import os
import sys
import yaml
import torch
import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def generate_checkpoint(data_path, tst1_ckpt, tst2_ckpt, save_path, is_sw=False):
    """生成对比学习checkpoint"""
    from models.transformer_ts import create_transformer_ts
    from models.transformer_fc import create_transformer_fc
    from pretrain.contrastive import InfoNCELoss, ProjectionHead
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print(f"Loading data from: {data_path}")
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    print(f"Data shape: timeseries={timeseries.shape}, pcc={pcc_vectors.shape}")
    
    # 划分数据 (7:1:2)
    indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
    val_test_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, random_state=42, stratify=val_test_labels)
    
    train_ts = timeseries[train_idx]
    train_pcc = pcc_vectors[train_idx]
    val_ts = timeseries[val_idx]
    val_pcc = pcc_vectors[val_idx]
    
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # 加载预训练模型配置
    tst1_checkpoint = torch.load(tst1_ckpt, map_location=device, weights_only=False)
    tst2_checkpoint = torch.load(tst2_ckpt, map_location=device, weights_only=False)
    
    tst1_config = tst1_checkpoint.get('config', {})
    tst2_config = tst2_checkpoint.get('config', {})
    
    # 创建模型
    n_rois = timeseries.shape[2]
    max_seq_len = tst1_config.get('max_seq_len', timeseries.shape[1])
    pcc_dim = pcc_vectors.shape[1]
    
    tst1_model_config = {
        'n_rois': n_rois,
        'max_seq_len': max_seq_len,
        'emb_dim': tst1_config.get('emb_dim', 512),
        'n_heads': tst1_config.get('n_heads', 8),
        'n_layers': tst1_config.get('n_layers', 6),
        'dim_feedforward': tst1_config.get('dim_feedforward', 2048),
        'dropout': tst1_config.get('dropout', 0.1)
    }
    tst1 = create_transformer_ts(config=tst1_model_config).to(device)
    
    tst2_model_config = {
        'pcc_dim': pcc_dim,
        'd_model': tst2_config.get('d_model', 256),
        'n_heads': tst2_config.get('n_heads', 8),
        'n_layers': tst2_config.get('n_layers', 2),
        'dim_feedforward': tst2_config.get('dim_feedforward', 512),
        'dropout': tst2_config.get('dropout', 0.1)
    }
    tst2 = create_transformer_fc(config=tst2_model_config).to(device)
    
    # 加载预训练权重
    tst1.load_state_dict(tst1_checkpoint['model_state_dict'])
    tst2.load_state_dict(tst2_checkpoint['model_state_dict'])
    print("Loaded pretrained weights")
    
    # 创建投影头
    tst1_dim = tst1_config.get('emb_dim', 512)
    tst2_dim = tst2_config.get('d_model', 256)
    proj_hidden_dim = 256
    proj_output_dim = 128
    
    proj_head1 = ProjectionHead(tst1_dim, proj_hidden_dim, proj_output_dim).to(device)
    proj_head2 = ProjectionHead(tst2_dim, proj_hidden_dim, proj_output_dim).to(device)
    
    # 对比学习设置 - 冻结TST1，训练TST2和投影头
    for param in tst1.parameters():
        param.requires_grad = False
    print("TST1 frozen for contrastive learning")
    
    # 优化器
    params = list(proj_head1.parameters()) + list(proj_head2.parameters()) + list(tst2.parameters())
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    
    # 损失函数
    criterion = InfoNCELoss(temperature=0.07)
    
    # 数据加载器
    train_dataset = TensorDataset(torch.FloatTensor(train_ts), torch.FloatTensor(train_pcc))
    val_dataset = TensorDataset(torch.FloatTensor(val_ts), torch.FloatTensor(val_pcc))
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    # 对比学习训练
    epochs = 50
    best_val_loss = float('inf')
    best_state = None
    
    print(f"\n--- Starting Contrastive Learning ({epochs} epochs) ---")
    for epoch in range(1, epochs + 1):
        # Training
        tst1.eval()
        tst2.train()
        proj_head1.train()
        proj_head2.train()
        
        train_loss = 0
        for ts_batch, pcc_batch in train_loader:
            ts_batch = ts_batch.to(device)
            pcc_batch = pcc_batch.to(device)
            
            with torch.no_grad():
                h1 = tst1.get_features(ts_batch)
            h2 = tst2.get_features(pcc_batch)
            
            z1 = proj_head1(h1)
            z2 = proj_head2(h2)
            
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        tst1.eval()
        tst2.eval()
        proj_head1.eval()
        proj_head2.eval()
        
        val_loss = 0
        with torch.no_grad():
            for ts_batch, pcc_batch in val_loader:
                ts_batch = ts_batch.to(device)
                pcc_batch = pcc_batch.to(device)
                
                h1 = tst1.get_features(ts_batch)
                h2 = tst2.get_features(pcc_batch)
                z1 = proj_head1(h1)
                z2 = proj_head2(h2)
                
                loss = criterion(z1, z2)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                'tst1_state_dict': tst1.state_dict(),
                'tst2_state_dict': tst2.state_dict(),
                'proj_head1_state_dict': proj_head1.state_dict(),
                'proj_head2_state_dict': proj_head2.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'config': {
                    'tst1_dim': tst1_dim,
                    'tst2_dim': tst2_dim,
                    'proj_hidden_dim': proj_hidden_dim,
                    'proj_output_dim': proj_output_dim,
                    'freeze_tst1': True,
                    'freeze_tst2': False
                }
            }
    
    # 保存checkpoint
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(best_state, save_path)
    print(f"\nSaved contrastive checkpoint to: {save_path}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return save_path


def main():
    DATA_DISK = "/root/autodl-tmp/TwoTST"
    
    # 生成非滑动窗口的checkpoint
    print("\n" + "="*60)
    print("Generating Non-Sliding Window Contrastive Checkpoint")
    print("="*60)
    
    normal_ckpt = generate_checkpoint(
        data_path="/root/workplace/exp/TwoTST/data/processed/processed_data.pkl",
        tst1_ckpt="/root/workplace/exp/TwoTST/checkpoints/tst1/tst1_best.pt",
        tst2_ckpt="/root/workplace/exp/TwoTST/checkpoints/tst2/tst2_best.pt",
        save_path=f"{DATA_DISK}/checkpoints/contrastive_checkpoint.pt",
        is_sw=False
    )
    
    # 生成滑动窗口的checkpoint
    print("\n" + "="*60)
    print("Generating Sliding Window Contrastive Checkpoint")
    print("="*60)
    
    sw_ckpt = generate_checkpoint(
        data_path="/root/workplace/exp/TwoTST/data/processed_sw/processed_data.pkl",
        tst1_ckpt="/root/workplace/exp/TwoTST/checkpoints_sw/tst1/tst1_best.pt",
        tst2_ckpt="/root/workplace/exp/TwoTST/checkpoints_sw/tst2/tst2_best.pt",
        save_path=f"{DATA_DISK}/checkpoints_sw/contrastive_checkpoint.pt",
        is_sw=True
    )
    
    print("\n" + "="*60)
    print("All checkpoints generated!")
    print(f"  Normal: {normal_ckpt}")
    print(f"  Sliding Window: {sw_ckpt}")
    print("="*60)


if __name__ == '__main__':
    main()
