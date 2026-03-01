"""
数据加载器模块
提供用于TwoTST框架的数据集类和数据加载函数。
支持受试者级划分与站点分层K折，避免滑窗导致的信息泄漏。
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import StratifiedKFold, train_test_split

from .splitters import (
    get_subject_level_fold_splits,
    get_subject_level_train_val_test_split,
    get_loso_fold_splits,
)


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
    获取预训练数据加载器（7:1:2划分）。
    若数据包含 subject_indices，则按受试者级划分，避免滑窗信息泄漏。

    Args:
        data_path: 预处理数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.1）
        test_ratio: 测试集比例（默认0.2）
        seed: 随机种子

    Returns:
        ts_train_loader, ts_val_loader, ts_test_loader, fc_train_loader, fc_val_loader, fc_test_loader
    """
    data = load_processed_data(data_path)

    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    subject_indices = data.get('subject_indices')
    site_ids = data.get('site_ids')

    if subject_indices is not None:
        train_idx, val_idx, test_idx = get_subject_level_train_val_test_split(
            labels, subject_indices, site_ids=site_ids,
            train_ratio=train_ratio, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed
        )
        print(f"Subject-level split - Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    else:
        indices = np.arange(len(labels))
        train_idx, temp_idx = train_test_split(
            indices, test_size=(val_ratio + test_ratio), random_state=seed, stratify=labels
        )
        val_test_labels = labels[temp_idx]
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=test_ratio / (val_ratio + test_ratio),
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
                         n_folds=5, fold_idx=0, val_ratio=0.15, seed=42,
                         use_subject_level_split=True,
                         eval_protocol='kfold'):
    """
    获取微调数据加载器（K折或 LOSO 交叉验证）。
    若数据包含 subject_indices 且 use_subject_level_split=True，
    则按受试者级划分，避免滑窗信息泄漏。

    Args:
        data_path: 预处理数据路径
        batch_size: 批次大小
        num_workers: 数据加载线程数
        n_folds: K折时有效；LOSO 时由站点数决定
        fold_idx: 当前折索引
        val_ratio: 验证集比例
        seed: 随机种子
        use_subject_level_split: 是否使用受试者级划分
        eval_protocol: 'kfold' | 'loso'；loso 即 Leave-One-Site-Out，评估跨站点泛化

    Returns:
        train_loader, val_loader, test_loader, split_info
    """
    data = load_processed_data(data_path)

    timeseries = data['timeseries']
    pcc_vectors = data['pcc_vectors']
    labels = data['labels']
    subject_indices = data.get('subject_indices')
    site_ids = data.get('site_ids')

    split_info = {}

    if eval_protocol == 'loso' and site_ids is not None and subject_indices is not None:
        splits = get_loso_fold_splits(
            labels, subject_indices, site_ids, val_ratio=val_ratio, seed=seed
        )
        if fold_idx >= len(splits):
            raise ValueError(f"fold_idx {fold_idx} >= n_sites {len(splits)}")
        s = splits[fold_idx]
        train_idx, val_idx, test_idx = s['train_idx'], s['val_idx'], s['test_idx']
        split_info = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'test_subjects': s['test_subjects'],
            'test_site': s['test_site'],
            'subject_indices': subject_indices,
        }
        print(f"LOSO fold {fold_idx} (test_site={s['test_site']}): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    elif use_subject_level_split and subject_indices is not None:
        splits = get_subject_level_fold_splits(
            labels, subject_indices, site_ids=site_ids,
            n_splits=n_folds, val_ratio=val_ratio, seed=seed
        )
        if fold_idx >= len(splits):
            raise ValueError(f"fold_idx {fold_idx} >= n_splits {len(splits)}")
        s = splits[fold_idx]
        train_idx, val_idx, test_idx = s['train_idx'], s['val_idx'], s['test_idx']
        split_info = {
            'train_idx': train_idx,
            'val_idx': val_idx,
            'test_idx': test_idx,
            'test_subjects': s['test_subjects'],
            'subject_indices': subject_indices,
        }
        print(f"Subject-level fold {fold_idx}: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        kfold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(kfold.split(timeseries, labels))
        train_val_idx, test_idx = splits[fold_idx]
        train_idx, val_idx = train_test_split(
            train_val_idx, test_size=val_ratio, random_state=seed,
            stratify=labels[train_val_idx]
        )
        split_info = {'test_idx': test_idx, 'subject_indices': subject_indices}

    train_dataset = TwoTSTDataset(
        timeseries[train_idx], pcc_vectors[train_idx], labels[train_idx]
    )
    val_dataset = TwoTSTDataset(
        timeseries[val_idx], pcc_vectors[val_idx], labels[val_idx]
    )
    test_dataset = TwoTSTDataset(
        timeseries[test_idx], pcc_vectors[test_idx], labels[test_idx]
    )

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

    return train_loader, val_loader, test_loader, split_info


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
