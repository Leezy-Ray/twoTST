"""
实验运行脚本
根据配置文件运行完整的实验流程
"""

import os
import sys
import argparse
import yaml
import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
import pickle
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_data(config):
    """加载数据。支持 truncate_timeseries_to 截断时间序列以适配 SW 预训练模型。"""
    data_path = config['data']['data_path']
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    subject_indices = data.get('subject_indices')
    site_ids = data.get('site_ids')

    truncate_to = config['data'].get('truncate_timeseries_to')
    if truncate_to is not None and timeseries.shape[1] > truncate_to:
        timeseries = timeseries[:, :truncate_to, :].copy()
        print(f"Truncated timeseries to {truncate_to} timepoints for pretrain compatibility")

    print(f"Loaded data: timeseries {timeseries.shape}, pcc {pcc_vectors.shape}")
    return timeseries, pcc_vectors, labels, subject_indices, site_ids


def split_data(labels, config, seed=42, subject_indices=None, site_ids=None):
    """划分数据集。若 subject_indices 存在则按受试者级划分，避免滑窗泄漏。"""
    split_ratio = config['data']['split_ratio']
    train_ratio = split_ratio['train']
    val_ratio = split_ratio['val'] / (split_ratio['val'] + split_ratio['test'] + 1e-9)
    test_ratio = split_ratio['test'] / (split_ratio['val'] + split_ratio['test'] + 1e-9)

    if subject_indices is not None:
        from utils.splitters import get_subject_level_train_val_test_split
        sr = config['data']['split_ratio']
        train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(
            labels, subject_indices, site_ids=site_ids,
            train_ratio=sr['train'],
            val_ratio=sr['val'],
            test_ratio=sr['test'],
            seed=seed,
        )
        print(f"Subject-level split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        indices = np.arange(len(labels))
        train_size = split_ratio['train']
        train_idx, temp_idx = train_test_split(
            indices, test_size=1 - train_size, random_state=seed, stratify=labels
        )
        val_ratio_split = split_ratio['val'] / (split_ratio['val'] + split_ratio['test'])
        val_test_labels = labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=1 - val_ratio_split, random_state=seed, stratify=val_test_labels
        )
        print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    return train_idx, val_idx, test_idx


def load_pretrained_models(config, device):
    """加载预训练模型。当 use_pretrained=false 时，TST1/TST2 随机初始化，用于端到端训练对比实验。"""
    from models.transformer_ts import create_transformer_ts
    from models.transformer_fc import create_transformer_fc
    
    model_config = config.get('model', {})
    tst1_config = model_config.get('tst1', {}).copy()
    tst2_config = model_config.get('tst2', {}).copy()
    
    if not config['pretrain']['use_pretrained']:
        # 无预训练：从 data 配置补充 n_rois, max_seq_len, pcc_dim
        data_config = config.get('data', {})
        tst1_config.setdefault('n_rois', data_config.get('n_rois', 200))
        tst1_config.setdefault('max_seq_len', data_config.get('time_points', data_config.get('max_seq_len', 100)))
        tst2_config.setdefault('pcc_dim', data_config.get('pcc_dim', 19900))
        print("No pretrain: TST1 and TST2 will be randomly initialized (end-to-end training from scratch)")
    
    # 加载预训练权重并从checkpoint中获取配置
    if config['pretrain']['use_pretrained']:
        tst1_ckpt = config['pretrain']['tst1'].get('checkpoint')
        tst2_ckpt = config['pretrain']['tst2'].get('checkpoint')
        
        # 从TST1 checkpoint获取配置
        if tst1_ckpt and os.path.exists(tst1_ckpt):
            checkpoint = torch.load(tst1_ckpt, map_location=device, weights_only=False)
            if 'config' in checkpoint:
                # 使用checkpoint中保存的配置
                ckpt_config = checkpoint['config']
                tst1_config.update(ckpt_config)
                print(f"Using TST1 config from checkpoint: {ckpt_config}")
        
        # 从TST2 checkpoint获取配置
        if tst2_ckpt and os.path.exists(tst2_ckpt):
            checkpoint2 = torch.load(tst2_ckpt, map_location=device, weights_only=False)
            if 'config' in checkpoint2:
                ckpt_config2 = checkpoint2['config']
                tst2_config.update(ckpt_config2)
                print(f"Using TST2 config from checkpoint: {ckpt_config2}")
    
    # 创建模型
    tst1 = create_transformer_ts(tst1_config).to(device)
    tst2 = create_transformer_fc(tst2_config).to(device)
    
    # 加载预训练权重
    if config['pretrain']['use_pretrained']:
        tst1_ckpt = config['pretrain']['tst1'].get('checkpoint')
        tst2_ckpt = config['pretrain']['tst2'].get('checkpoint')
        
        if tst1_ckpt and os.path.exists(tst1_ckpt):
            checkpoint = torch.load(tst1_ckpt, map_location=device, weights_only=False)
            tst1.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded TST1 weights from {tst1_ckpt}")
        
        if tst2_ckpt and os.path.exists(tst2_ckpt):
            checkpoint2 = torch.load(tst2_ckpt, map_location=device, weights_only=False)
            tst2.load_state_dict(checkpoint2['model_state_dict'])
            print(f"Loaded TST2 weights from {tst2_ckpt}")
    
    return tst1, tst2


def load_contrastive_checkpoint(tst1, tst2, config, device):
    """加载已有的对比学习checkpoint"""
    from pretrain.contrastive import ProjectionHead
    
    cont_config = config['contrastive']
    checkpoint_path = cont_config.get('load_checkpoint')
    
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise ValueError(f"Contrastive checkpoint not found: {checkpoint_path}")
    
    print(f"Loading contrastive checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 获取维度信息
    tst1_dim = config['model']['tst1'].get('emb_dim', 512)
    tst2_dim = config['model']['tst2'].get('d_model', 256)
    proj_hidden_dim = cont_config.get('proj_hidden_dim', 256)
    proj_output_dim = cont_config.get('proj_output_dim', 128)
    
    # 创建投影头
    proj_head1 = ProjectionHead(tst1_dim, proj_hidden_dim, proj_output_dim).to(device)
    proj_head2 = ProjectionHead(tst2_dim, proj_hidden_dim, proj_output_dim).to(device)
    
    # 加载权重
    tst1.load_state_dict(checkpoint['tst1_state_dict'])
    tst2.load_state_dict(checkpoint['tst2_state_dict'])
    proj_head1.load_state_dict(checkpoint['proj_head1_state_dict'])
    proj_head2.load_state_dict(checkpoint['proj_head2_state_dict'])
    
    print(f"Loaded contrastive checkpoint: best_val_loss={checkpoint.get('best_val_loss', 'N/A'):.4f}")
    
    return tst1, tst2, proj_head1, proj_head2


def run_contrastive_learning(tst1, tst2, train_data, val_data, config, device, writer):
    """运行对比学习"""
    from pretrain.contrastive import InfoNCELoss, NTXentLoss, ProjectionHead
    
    cont_config = config['contrastive']
    
    # 创建投影头
    tst1_dim = config['model']['tst1'].get('emb_dim', 512)
    tst2_dim = config['model']['tst2'].get('d_model', 256)
    
    proj_head1 = ProjectionHead(
        tst1_dim, cont_config['proj_hidden_dim'], cont_config['proj_output_dim']
    ).to(device)
    proj_head2 = ProjectionHead(
        tst2_dim, cont_config['proj_hidden_dim'], cont_config['proj_output_dim']
    ).to(device)
    
    # 冻结策略
    if cont_config['freeze_tst1']:
        for param in tst1.parameters():
            param.requires_grad = False
        print("TST1 frozen for contrastive learning")
    
    if cont_config['freeze_tst2']:
        for param in tst2.parameters():
            param.requires_grad = False
        print("TST2 frozen for contrastive learning")
    
    # 优化器
    params = list(proj_head1.parameters()) + list(proj_head2.parameters())
    if not cont_config['freeze_tst1']:
        params += list(tst1.parameters())
    if not cont_config['freeze_tst2']:
        params += list(tst2.parameters())
    
    optimizer = torch.optim.AdamW(
        params, lr=float(cont_config['lr']), weight_decay=float(cont_config['weight_decay'])
    )
    
    # 损失函数
    loss_type = cont_config.get('loss_type', 'infonce')
    if loss_type == 'infonce':
        criterion = InfoNCELoss(temperature=cont_config['temperature'])
    elif loss_type == 'ntxent':
        criterion = NTXentLoss(temperature=cont_config['temperature'])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # 训练
    best_val_loss = float('inf')
    best_state = None
    train_ts, train_pcc, _ = train_data
    val_ts, val_pcc, _ = val_data
    
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_ts), torch.FloatTensor(train_pcc)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=cont_config['batch_size'], shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_ts), torch.FloatTensor(val_pcc)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cont_config['batch_size'], shuffle=False
    )
    
    def _alignment(z1, z2):
        """正样本对平均余弦相似度 (表征对齐度, 越高越好)"""
        z1 = torch.nn.functional.normalize(z1, p=2, dim=1)
        z2 = torch.nn.functional.normalize(z2, p=2, dim=1)
        return (z1 * z2).sum(dim=1).mean().item()

    history = []
    for epoch in range(1, cont_config['epochs'] + 1):
        # Training
        tst1.train() if not cont_config['freeze_tst1'] else tst1.eval()
        tst2.train() if not cont_config['freeze_tst2'] else tst2.eval()
        proj_head1.train()
        proj_head2.train()
        
        train_loss = 0
        train_align_sum = 0
        train_align_n = 0
        for ts_batch, pcc_batch in train_loader:
            ts_batch = ts_batch.to(device)
            pcc_batch = pcc_batch.to(device)
            
            # 获取特征
            with torch.set_grad_enabled(not cont_config['freeze_tst1']):
                h1 = tst1.get_features(ts_batch)
            with torch.set_grad_enabled(not cont_config['freeze_tst2']):
                h2 = tst2.get_features(pcc_batch)
            
            # 投影
            z1 = proj_head1(h1)
            z2 = proj_head2(h2)
            
            # 对比损失
            loss = criterion(z1, z2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                train_align_sum += _alignment(z1, z2) * z1.size(0)
                train_align_n += z1.size(0)
        
        train_loss /= len(train_loader)
        train_align = train_align_sum / max(train_align_n, 1)
        
        # Validation
        tst1.eval()
        tst2.eval()
        proj_head1.eval()
        proj_head2.eval()
        
        val_loss = 0
        val_align_sum = 0
        val_align_n = 0
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
                val_align_sum += _alignment(z1, z2) * z1.size(0)
                val_align_n += z1.size(0)
        
        val_loss /= len(val_loader)
        val_align = val_align_sum / max(val_align_n, 1)
        
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_alignment': train_align,
            'val_alignment': val_align,
        })
        
        # Log
        writer.add_scalar('Contrastive/train_loss', train_loss, epoch)
        writer.add_scalar('Contrastive/val_loss', val_loss, epoch)
        writer.add_scalar('Contrastive/train_alignment', train_align, epoch)
        writer.add_scalar('Contrastive/val_alignment', val_align, epoch)
        
        if epoch % 10 == 0:
            print(f"Contrastive Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_align={val_align:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                'tst1_state_dict': tst1.state_dict(),
                'tst2_state_dict': tst2.state_dict(),
                'proj_head1_state_dict': proj_head1.state_dict(),
                'proj_head2_state_dict': proj_head2.state_dict(),
                'best_val_loss': best_val_loss,
                'epoch': epoch,
                'config': cont_config
            }
    
    # 保存对比学习checkpoint 和 训练曲线历史
    save_dir = config['finetune']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    contrastive_ckpt_path = os.path.join(save_dir, 'contrastive_checkpoint.pt')
    torch.save(best_state, contrastive_ckpt_path)
    print(f"Saved contrastive checkpoint to: {contrastive_ckpt_path}")
    import json
    history_path = os.path.join(save_dir, 'contrastive_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump({'history': history, 'config': {k: v for k, v in cont_config.items() if k != 'load_checkpoint'}}, f, indent=2)
    print(f"Saved contrastive history to: {history_path}")
    
    # 加载最佳状态
    if best_state is not None:
        tst1.load_state_dict(best_state['tst1_state_dict'])
        tst2.load_state_dict(best_state['tst2_state_dict'])
        proj_head1.load_state_dict(best_state['proj_head1_state_dict'])
        proj_head2.load_state_dict(best_state['proj_head2_state_dict'])
    
    # 返回TST和投影头
    return tst1, tst2, proj_head1, proj_head2


def create_fusion_model(config, tst1_dim, tst2_dim, device):
    """创建融合模型"""
    from models.fusion import (
        ConcatFusion, GatedFusion, CrossAttentionFusion,
        BilinearFusion, AttentionPoolingFusion, GatedMultiFusion
    )
    
    fusion_config = config['fusion']
    fusion_type = fusion_config['type']
    
    if fusion_type == 'concat':
        fusion = ConcatFusion(tst1_dim, tst2_dim, fusion_config.get('concat', {}).get('hidden_dim', 512))
    elif fusion_type == 'gated':
        fusion = GatedFusion(tst1_dim, tst2_dim, fusion_config.get('gated', {}).get('hidden_dim', 512))
    elif fusion_type == 'cross_attention':
        ca_config = fusion_config.get('cross_attention', {})
        fusion = CrossAttentionFusion(
            tst1_dim, tst2_dim,
            n_heads=ca_config.get('n_heads', 8),
            dropout=ca_config.get('dropout', 0.1)
        )
    elif fusion_type == 'bilinear':
        fusion = BilinearFusion(tst1_dim, tst2_dim, fusion_config.get('bilinear', {}).get('output_dim', 256))
    elif fusion_type == 'attention_pooling':
        fusion = AttentionPoolingFusion(tst1_dim, tst2_dim, fusion_config.get('attention_pooling', {}).get('hidden_dim', 256))
    elif fusion_type == 'gated_multi':
        gm_config = fusion_config.get('gated_multi', {})
        fusion = GatedMultiFusion(
            tst1_dim, tst2_dim,
            output_dim=gm_config.get('output_dim', 256),
            hidden_dim=gm_config.get('hidden_dim', 128),
            dropout=gm_config.get('dropout', 0.1),
            init_favor_idx=gm_config.get('init_favor_idx', 4)
        )
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion.to(device)


class FinetuneModel(torch.nn.Module):
    """微调模型"""
    def __init__(self, tst1, tst2, fusion, classifier, use_tst1=True, use_tst2=True):
        super().__init__()
        self.tst1 = tst1
        self.tst2 = tst2
        self.fusion = fusion
        self.classifier = classifier
        self.use_tst1 = use_tst1
        self.use_tst2 = use_tst2
    
    def forward(self, timeseries, pcc):
        if self.use_tst1 and self.use_tst2:
            h1 = self.tst1.get_features(timeseries)
            h2 = self.tst2.get_features(pcc)
            fused = self.fusion(h1, h2)
        elif self.use_tst1:
            fused = self.tst1.get_features(timeseries)
        else:
            fused = self.tst2.get_features(pcc)
        
        return self.classifier(fused)


class ProjectionFinetuneModel(torch.nn.Module):
    """使用投影头的微调模型"""
    def __init__(self, tst1, tst2, proj_head1, proj_head2, fusion, classifier, 
                 use_tst1=True, use_tst2=True, use_projection=True):
        """
        Args:
            tst1: TST1模型
            tst2: TST2模型
            proj_head1: TST1的投影头（固定参数）
            proj_head2: TST2的投影头（固定参数）
            fusion: 融合模块（可选，如果只用一个流则为None）
            classifier: 分类器
            use_tst1: 是否使用TST1
            use_tst2: 是否使用TST2
            use_projection: 是否使用投影头
        """
        super().__init__()
        self.tst1 = tst1
        self.tst2 = tst2
        self.proj_head1 = proj_head1
        self.proj_head2 = proj_head2
        self.fusion = fusion
        self.classifier = classifier
        self.use_tst1 = use_tst1
        self.use_tst2 = use_tst2
        self.use_projection = use_projection
    
    def forward(self, timeseries, pcc):
        if self.use_projection:
            # 使用投影头
            if self.use_tst1 and self.use_tst2:
                # 双流：TST1 -> projection1, TST2 -> projection2 -> 融合
                h1 = self.tst1.get_features(timeseries)
                h2 = self.tst2.get_features(pcc)
                z1 = self.proj_head1(h1)  # (batch, proj_output_dim)
                z2 = self.proj_head2(h2)  # (batch, proj_output_dim)
                if self.fusion is not None:
                    fused = self.fusion(z1, z2)
                else:
                    # 如果没有融合模块，直接拼接
                    fused = torch.cat([z1, z2], dim=1)
            elif self.use_tst1:
                # 单流TST1：TST1 -> projection1
                h1 = self.tst1.get_features(timeseries)
                fused = self.proj_head1(h1)
            else:
                # 单流TST2：TST2 -> projection2
                h2 = self.tst2.get_features(pcc)
                fused = self.proj_head2(h2)
        else:
            # 不使用投影头（原始方式）
            if self.use_tst1 and self.use_tst2:
                h1 = self.tst1.get_features(timeseries)
                h2 = self.tst2.get_features(pcc)
                fused = self.fusion(h1, h2)
            elif self.use_tst1:
                fused = self.tst1.get_features(timeseries)
            else:
                fused = self.tst2.get_features(pcc)
        
        return self.classifier(fused)


def create_classifier(input_dim, config, device):
    """创建分类器"""
    hidden_dims = config['finetune']['classifier']['hidden_dims']
    dropout = config['finetune']['classifier']['dropout']
    
    layers = []
    prev_dim = input_dim
    for dim in hidden_dims:
        layers.extend([
            torch.nn.Linear(prev_dim, dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout)
        ])
        prev_dim = dim
    layers.append(torch.nn.Linear(prev_dim, 2))  # 二分类
    
    return torch.nn.Sequential(*layers).to(device)


def run_finetune(model, train_data, val_data, test_data, config, device, writer,
                 subject_indices=None, test_idx=None, subject_agg_strategy=None):
    """运行微调。若 subject_indices 与 subject_agg_strategy 均提供，则按受试者级汇总测试指标。"""
    finetune_config = config['finetune']
    
    train_ts, train_pcc, train_labels = train_data
    val_ts, val_pcc, val_labels = val_data
    test_ts, test_pcc, test_labels = test_data
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_ts),
        torch.FloatTensor(train_pcc),
        torch.LongTensor(train_labels)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=finetune_config['batch_size'], shuffle=True
    )
    
    val_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(val_ts),
        torch.FloatTensor(val_pcc),
        torch.LongTensor(val_labels)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=finetune_config['batch_size'], shuffle=False
    )
    
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_ts),
        torch.FloatTensor(test_pcc),
        torch.LongTensor(test_labels)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=finetune_config['batch_size'], shuffle=False
    )
    
    # 冻结策略
    if finetune_config['freeze_tst1'] and model.use_tst1:
        for param in model.tst1.parameters():
            param.requires_grad = False
        print("TST1 frozen for finetuning")
    
    if finetune_config['freeze_tst2'] and model.use_tst2:
        for param in model.tst2.parameters():
            param.requires_grad = False
        print("TST2 frozen for finetuning")
    
    # 优化器
    lr = float(finetune_config['lr'])
    weight_decay = float(finetune_config['weight_decay'])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )
    
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=finetune_config['epochs']
    )
    
    best_val_auc = 0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0
    train_losses = []
    val_aucs = []
    val_accs = []
    
    for epoch in range(1, finetune_config['epochs'] + 1):
        # Training
        model.train()
        train_loss = 0
        for ts, pcc, labels in train_loader:
            ts, pcc, labels = ts.to(device), pcc.to(device), labels.to(device)
            
            outputs = model(ts, pcc)
            loss = criterion(outputs, labels)

            # 熵正则：鼓励门控多策略融合集中选择，降低熵
            entropy_weight = finetune_config.get('gated_multi_entropy_weight', 0.0)
            if entropy_weight > 0 and hasattr(model, 'fusion') and hasattr(model.fusion, 'last_gate_weights'):
                gw = model.fusion.last_gate_weights
                if gw is not None:
                    entropy = -torch.sum(gw * (gw + 1e-8).log(), dim=-1).mean()
                    loss = loss + entropy_weight * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        model.eval()
        val_preds, val_probs, val_true = [], [], []
        with torch.no_grad():
            for ts, pcc, labels in val_loader:
                ts, pcc = ts.to(device), pcc.to(device)
                outputs = model(ts, pcc)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = outputs.argmax(dim=1)
                
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(labels.numpy())
        
        val_auc = roc_auc_score(val_true, val_probs)
        val_acc = accuracy_score(val_true, val_preds)
        
        # 记录历史
        train_losses.append(train_loss)
        val_aucs.append(val_auc)
        val_accs.append(val_acc)
        
        # Log
        writer.add_scalar('Finetune/train_loss', train_loss, epoch)
        writer.add_scalar('Finetune/val_auc', val_auc, epoch)
        writer.add_scalar('Finetune/val_acc', val_acc, epoch)
        
        if epoch % 10 == 0:
            print(f"Finetune Epoch {epoch}: train_loss={train_loss:.4f}, val_auc={val_auc:.4f}, val_acc={val_acc:.4f}")
        
        # Best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= finetune_config['patience']:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    model.eval()
    
    test_preds, test_probs, test_true = [], [], []
    with torch.no_grad():
        for ts, pcc, labels in test_loader:
            ts, pcc = ts.to(device), pcc.to(device)
            outputs = model(ts, pcc)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = outputs.argmax(dim=1)
            
            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_true.extend(labels.numpy())
    
    # 受试者级汇总（滑窗场景）：若提供 subject_indices 与 subject_agg_strategy
    if subject_indices is not None and test_idx is not None and subject_agg_strategy is not None:
        from utils.metrics import aggregate_window_predictions_to_subject_level
        test_true = np.array(test_true)
        test_preds = np.array(test_preds)
        test_probs = np.array(test_probs)
        test_true, test_preds, test_probs = aggregate_window_predictions_to_subject_level(
            test_true, test_preds, test_probs, test_idx, subject_indices, strategy=subject_agg_strategy
        )
        print(f"Subject-level aggregation ({subject_agg_strategy}): {len(test_true)} subjects")
    
    # Calculate metrics
    test_auc = roc_auc_score(test_true, test_probs)
    test_acc = accuracy_score(test_true, test_preds)
    test_f1 = f1_score(test_true, test_preds)
    
    tn, fp, fn, tp = confusion_matrix(test_true, test_preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    results = {
        'auc': test_auc,
        'accuracy': test_acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': test_f1,
        'best_val_auc': best_val_auc,
        'best_epoch': best_epoch,
        'total_epochs': len(train_losses),
        'train_losses': train_losses,
        'val_aucs': val_aucs,
        'val_accs': val_accs
    }
    
    return results, model


def main(args):
    # 加载配置
    config = load_config(args.config)
    exp_name = config['experiment']['name']
    
    print(f"\n{'='*60}")
    print(f"Running experiment: {exp_name}")
    print(f"Description: {config['experiment']['description']}")
    print(f"{'='*60}\n")
    
    # 设置随机种子
    seed = config['training'].get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 设置设备
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建日志目录
    log_dir = config['logging']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 创建保存目录
    save_dir = config['finetune']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    timeseries, pcc_vectors, labels, subject_indices, site_ids = load_data(config)

    # 划分数据（受试者级划分以避免滑窗泄漏）
    train_idx, val_idx, test_idx = split_data(
        labels, config, seed, subject_indices=subject_indices, site_ids=site_ids
    )
    
    train_data = (timeseries[train_idx], pcc_vectors[train_idx], labels[train_idx])
    val_data = (timeseries[val_idx], pcc_vectors[val_idx], labels[val_idx])
    test_data = (timeseries[test_idx], pcc_vectors[test_idx], labels[test_idx])
    
    # 加载预训练模型
    tst1, tst2 = load_pretrained_models(config, device)
    
    # 对比学习（可选）
    proj_head1, proj_head2 = None, None
    if config['contrastive']['enabled']:
        cont_config = config['contrastive']
        load_checkpoint = cont_config.get('load_checkpoint')
        
        if load_checkpoint and os.path.exists(load_checkpoint):
            # 加载已有的对比学习checkpoint
            print("\n--- Loading Existing Contrastive Checkpoint ---")
            tst1, tst2, proj_head1, proj_head2 = load_contrastive_checkpoint(
                tst1, tst2, config, device
            )
        else:
            # 运行对比学习
            print("\n--- Running Contrastive Learning ---")
            tst1, tst2, proj_head1, proj_head2 = run_contrastive_learning(
                tst1, tst2, train_data, val_data, config, device, writer
            )
        
        if getattr(args, 'contrastive_only', False):
            writer.close()
            print("Contrastive-only mode: saved history, exiting.")
            return None
        
        # 投影头：默认冻结，可通过 finetune.freeze_projection=false 解冻
        freeze_proj = config.get('finetune', {}).get('freeze_projection', True)
        if proj_head1 is not None:
            for param in proj_head1.parameters():
                param.requires_grad = not freeze_proj
        if proj_head2 is not None:
            for param in proj_head2.parameters():
                param.requires_grad = not freeze_proj
        print("Projection heads frozen for finetuning" if freeze_proj else "Projection heads trainable for finetuning")
    
    # 检查是否使用投影头微调
    finetune_config = config['finetune']
    use_projection = finetune_config.get('use_projection', False)
    
    # 创建融合模型
    fusion_config = config['fusion']
    use_tst1 = fusion_config.get('use_tst1', True)
    use_tst2 = fusion_config.get('use_tst2', True)
    
    tst1_dim = config['model']['tst1'].get('emb_dim', 512)
    tst2_dim = config['model']['tst2'].get('d_model', 256)
    
    if use_projection:
        # 使用投影头微调模式
        if proj_head1 is None or proj_head2 is None:
            raise ValueError("Projection heads are required for projection finetuning, but contrastive learning is not enabled!")
        
        proj_output_dim = config['contrastive'].get('proj_output_dim', 128)
        
        if use_tst1 and use_tst2:
            # 双流：使用投影后的特征进行融合
            fusion = create_fusion_model(config, proj_output_dim, proj_output_dim, device)
            # 获取融合输出维度（投影后维度相同，都是proj_output_dim）
            if fusion_config['type'] == 'concat':
                # ConcatFusion输出维度由hidden_dim参数指定
                classifier_input_dim = fusion_config.get('concat', {}).get('hidden_dim', proj_output_dim)
            elif fusion_config['type'] == 'gated':
                # GatedFusion输出维度是max(dim_ts, dim_fc) = proj_output_dim
                classifier_input_dim = proj_output_dim
            elif fusion_config['type'] == 'cross_attention':
                classifier_input_dim = proj_output_dim
            elif fusion_config['type'] == 'bilinear':
                classifier_input_dim = fusion_config.get('bilinear', {}).get('output_dim', 256)
            elif fusion_config['type'] == 'attention_pooling':
                classifier_input_dim = proj_output_dim
            elif fusion_config['type'] == 'gated_multi':
                classifier_input_dim = fusion_config.get('gated_multi', {}).get('output_dim', 256)
            else:
                classifier_input_dim = proj_output_dim
        else:
            # 单流：只使用一个投影头
            fusion = None
            classifier_input_dim = proj_output_dim
        
        # 创建分类器
        classifier = create_classifier(classifier_input_dim, config, device)
        
        # 创建使用投影头的微调模型
        model = ProjectionFinetuneModel(
            tst1, tst2, proj_head1, proj_head2, fusion, classifier,
            use_tst1, use_tst2, use_projection=True
        )
        print("Using projection heads for finetuning")
    else:
        # 原始微调模式（不使用投影头）
        if use_tst1 and use_tst2:
            fusion = create_fusion_model(config, tst1_dim, tst2_dim, device)
            # 获取融合输出维度
            if fusion_config['type'] == 'concat':
                classifier_input_dim = fusion_config.get('concat', {}).get('hidden_dim', 512)
            elif fusion_config['type'] == 'gated':
                classifier_input_dim = fusion_config.get('gated', {}).get('hidden_dim', 512)
            elif fusion_config['type'] == 'cross_attention':
                classifier_input_dim = tst1_dim
            elif fusion_config['type'] == 'bilinear':
                classifier_input_dim = fusion_config.get('bilinear', {}).get('output_dim', 256)
            elif fusion_config['type'] == 'attention_pooling':
                # AttentionPooling的输出维度是max(dim_ts, dim_fc)
                classifier_input_dim = max(tst1_dim, tst2_dim)
            elif fusion_config['type'] == 'gated_multi':
                classifier_input_dim = fusion_config.get('gated_multi', {}).get('output_dim', 256)
            else:
                classifier_input_dim = tst1_dim
        else:
            fusion = None
            classifier_input_dim = tst1_dim if use_tst1 else tst2_dim
        
        # 创建分类器
        classifier = create_classifier(classifier_input_dim, config, device)
        
        # 创建原始微调模型
        model = FinetuneModel(tst1, tst2, fusion, classifier, use_tst1, use_tst2)
    
    # 微调
    subject_agg = getattr(args, 'subject_agg_strategy', None)
    print("\n--- Running Finetuning ---")
    results, model = run_finetune(
        model, train_data, val_data, test_data, config, device, writer,
        subject_indices=subject_indices, test_idx=test_idx, subject_agg_strategy=subject_agg
    )
    
    # 打印结果
    print(f"\n{'='*60}")
    print(f"Experiment: {exp_name}")
    print(f"{'='*60}")
    print(f"Test AUC:         {results['auc']:.4f}")
    print(f"Test Accuracy:    {results['accuracy']:.4f}")
    print(f"Test Sensitivity: {results['sensitivity']:.4f}")
    print(f"Test Specificity: {results['specificity']:.4f}")
    print(f"Test F1:          {results['f1']:.4f}")
    print(f"Best Val AUC:     {results['best_val_auc']:.4f}")
    print(f"{'='*60}\n")
    
    # 保存结果（含复现信息）
    from utils.metrics import get_reproducibility_info
    results['experiment'] = exp_name
    results['config'] = config
    results['reproducibility'] = {**get_reproducibility_info(), 'seed': seed}
    results_path = os.path.join(save_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")
    
    # 保存模型
    model_path = os.path.join(save_dir, 'best_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'config': config
    }, model_path)
    print(f"Model saved to {model_path}")
    
    writer.close()
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run TwoTST experiment')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to experiment config file')
    parser.add_argument('--subject_agg_strategy', type=str, default=None,
                        choices=['prob_mean', 'majority_vote'],
                        help='Subject-level aggregation for sliding window: prob_mean or majority_vote')
    parser.add_argument('--contrastive_only', action='store_true',
                        help='Run contrastive learning only, save history and exit (for curve plotting)')
    args = parser.parse_args()
    
    main(args)
