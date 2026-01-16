"""
数据加载器模块
提供用于TwoTST框架的数据集类和数据加载函数
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split


class TwoTSTDataset(Dataset):
    """
    TwoTST数据集类
    返回时间序列、PCC向量和标签
    """
    
    def __init__(self, timeseries, pcc_vectors, labels, 
                 normalize_ts=True, normalize_pcc=True):
        """
        Args:
            timeseries: ndarray, shape (n_samples, n_timepoints, n_rois)
            pcc_vectors: ndarray, shape (n_samples, pcc_dim)
            labels: ndarray, shape (n_samples,)
            normalize_ts: 是否对时间序列进行标准化
            normalize_pcc: 是否对PCC向量进行标准化
        """
        self.timeseries = timeseries.astype(np.float32)
        self.pcc_vectors = pcc_vectors.astype(np.float32)
        self.labels = labels.astype(np.int64)
        
        self.normalize_ts = normalize_ts
        self.normalize_pcc = normalize_pcc
        
        # 计算全局统计量用于标准化
        if normalize_ts:
            self.ts_mean = np.mean(self.timeseries)
            self.ts_std = np.std(self.timeseries) + 1e-8
        
        if normalize_pcc:
            self.pcc_mean = np.mean(self.pcc_vectors)
            self.pcc_std = np.std(self.pcc_vectors) + 1e-8
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        ts = self.timeseries[idx].copy()
        pcc = self.pcc_vectors[idx].copy()
        label = self.labels[idx]
        
        # 标准化
        if self.normalize_ts:
            ts = (ts - self.ts_mean) / self.ts_std
        
        if self.normalize_pcc:
            pcc = (pcc - self.pcc_mean) / self.pcc_std
        
        return {
            'timeseries': torch.tensor(ts, dtype=torch.float32),
            'pcc_vector': torch.tensor(pcc, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


class PretrainTSDataset(Dataset):
    """
    TST1预训练数据集
    只返回时间序列数据
    """
    
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


class PretrainFCDataset(Dataset):
    """
    TST2预训练数据集
    只返回PCC向量数据
    """
    
    def __init__(self, pcc_vectors, normalize=True):
        """
        Args:
            pcc_vectors: ndarray, shape (n_samples, pcc_dim)
            normalize: 是否进行标准化
        """
        self.pcc_vectors = pcc_vectors.astype(np.float32)
        self.normalize = normalize
        
        if normalize:
            self.mean = np.mean(self.pcc_vectors)
            self.std = np.std(self.pcc_vectors) + 1e-8
    
    def __len__(self):
        return len(self.pcc_vectors)
    
    def __getitem__(self, idx):
        pcc = self.pcc_vectors[idx].copy()
        
        if self.normalize:
            pcc = (pcc - self.mean) / self.std
        
        return torch.tensor(pcc, dtype=torch.float32)


def load_processed_data(data_path):
    """
    加载预处理后的数据
    
    Args:
        data_path: processed_data.pkl 文件路径
    
    Returns:
        data: dict containing timeseries, pcc_vectors, labels, etc.
    """
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    return data


def get_pretrain_loaders(data_path, batch_size=32, num_workers=4, 
                         train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    """
    获取预训练数据加载器（7:1:2划分）
    
    Args:
        data_path: 预处理数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.2）
        seed: 随机种子
    
    Returns:
        ts_train_loader: TST1训练数据加载器
        ts_val_loader: TST1验证数据加载器
        ts_test_loader: TST1测试数据加载器
        fc_train_loader: TST2训练数据加载器
        fc_val_loader: TST2验证数据加载器
        fc_test_loader: TST2测试数据加载器
    """
    data = load_processed_data(data_path)
    
    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    
    # 7:1:2划分：先分出训练集，再从剩余中分出验证集和测试集
    indices = np.arange(len(labels))
    
    # 第一次划分：训练集 vs (验证集+测试集)
    train_idx, temp_idx = train_test_split(
        indices, test_size=(val_ratio + test_ratio), random_state=seed, stratify=labels
    )
    
    # 第二次划分：验证集 vs 测试集（从temp_idx中）
    val_test_labels = labels[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=test_ratio/(val_ratio+test_ratio), 
        random_state=seed, stratify=val_test_labels
    )
    
    print(f"Data split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    
    # TST1 数据加载器
    ts_train_dataset = PretrainTSDataset(timeseries[train_idx])
    ts_val_dataset = PretrainTSDataset(timeseries[val_idx])
    ts_test_dataset = PretrainTSDataset(timeseries[test_idx])
    
    ts_train_loader = DataLoader(
        ts_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    ts_val_loader = DataLoader(
        ts_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    ts_test_loader = DataLoader(
        ts_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    # TST2 数据加载器
    fc_train_dataset = PretrainFCDataset(pcc_vectors[train_idx])
    fc_val_dataset = PretrainFCDataset(pcc_vectors[val_idx])
    fc_test_dataset = PretrainFCDataset(pcc_vectors[test_idx])
    
    fc_train_loader = DataLoader(
        fc_train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    fc_val_loader = DataLoader(
        fc_val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    fc_test_loader = DataLoader(
        fc_test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return (ts_train_loader, ts_val_loader, ts_test_loader, 
            fc_train_loader, fc_val_loader, fc_test_loader)


def get_finetune_loaders(data_path, batch_size=32, num_workers=4,
                         n_folds=5, fold_idx=0, seed=42):
    """
    获取微调数据加载器（K折交叉验证）
    
    Args:
        data_path: 预处理数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        n_folds: 交叉验证折数
        fold_idx: 当前折索引
        seed: 随机种子
    
    Returns:
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        test_loader: 测试数据加载器
    """
    data = load_processed_data(data_path)
    
    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    
    # K折交叉验证
    kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    splits = list(kfold.split(timeseries, labels))
    
    train_val_idx, test_idx = splits[fold_idx]
    
    # 从训练集中划分验证集
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.15, random_state=seed,
        stratify=labels[train_val_idx]
    )
    
    # 创建数据集
    train_dataset = TwoTSTDataset(
        timeseries[train_idx], pcc_vectors[train_idx], labels[train_idx]
    )
    val_dataset = TwoTSTDataset(
        timeseries[val_idx], pcc_vectors[val_idx], labels[val_idx]
    )
    test_dataset = TwoTSTDataset(
        timeseries[test_idx], pcc_vectors[test_idx], labels[test_idx]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def collate_fn_pretrain_ts(batch):
    """TST1预训练的collate函数"""
    return torch.stack(batch, dim=0)


def collate_fn_pretrain_fc(batch):
    """TST2预训练的collate函数"""
    return torch.stack(batch, dim=0)


def collate_fn_finetune(batch):
    """微调的collate函数"""
    timeseries = torch.stack([item['timeseries'] for item in batch], dim=0)
    pcc_vectors = torch.stack([item['pcc_vector'] for item in batch], dim=0)
    labels = torch.stack([item['label'] for item in batch], dim=0)
    
    return {
        'timeseries': timeseries,
        'pcc_vector': pcc_vectors,
        'label': labels
    }
