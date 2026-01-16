"""
掩码策略模块
实现 ROI-level 掩码和 PCC 元素级掩码
"""

import numpy as np
import torch
import random


def mask_roi_level(timeseries, mask_ratio=None):
    """
    ROI-level 掩码策略
    随机掩码一定比例的 ROI，将整列时间序列置零
    
    Args:
        timeseries: torch.Tensor, shape (batch, T, n_rois) 或 (T, n_rois)
        mask_ratio: 掩码比例，如果为 None 则随机选择 0.25 或 0.5
    
    Returns:
        masked_ts: 掩码后的时间序列
        mask: 掩码位置，True 表示被掩码
        target: 被掩码位置的原始值
    """
    if mask_ratio is None:
        # 随机选择掩码比例
        mask_ratio = 0.25 if random.random() < 0.5 else 0.5
    
    is_batch = timeseries.dim() == 3
    
    if not is_batch:
        timeseries = timeseries.unsqueeze(0)
    
    batch_size, T, n_rois = timeseries.shape
    device = timeseries.device
    
    # 计算要掩码的 ROI 数量
    num_mask = int(n_rois * mask_ratio)
    
    # 为每个样本生成不同的掩码
    masked_ts = timeseries.clone()
    mask = torch.zeros(batch_size, n_rois, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # 随机选择要掩码的 ROI 索引
        mask_indices = torch.randperm(n_rois, device=device)[:num_mask]
        mask[i, mask_indices] = True
        # 将选中的 ROI 整列置零
        masked_ts[i, :, mask_indices] = 0
    
    # 扩展 mask 到时间维度 (batch, T, n_rois)
    mask_expanded = mask.unsqueeze(1).expand(-1, T, -1)
    
    # 获取被掩码位置的原始值
    target = timeseries.clone()
    
    if not is_batch:
        masked_ts = masked_ts.squeeze(0)
        mask_expanded = mask_expanded.squeeze(0)
        target = target.squeeze(0)
        mask = mask.squeeze(0)
    
    return masked_ts, mask_expanded, target, mask


def mask_pcc_level(pcc_vector, mask_ratio=0.15):
    """
    PCC 元素级掩码策略
    随机掩码一定比例的 PCC 值
    
    Args:
        pcc_vector: torch.Tensor, shape (batch, pcc_dim) 或 (pcc_dim,)
        mask_ratio: 掩码比例，默认 0.15
    
    Returns:
        masked_pcc: 掩码后的 PCC 向量
        mask: 掩码位置，True 表示被掩码
        target: 被掩码位置的原始值
    """
    is_batch = pcc_vector.dim() == 2
    
    if not is_batch:
        pcc_vector = pcc_vector.unsqueeze(0)
    
    batch_size, pcc_dim = pcc_vector.shape
    device = pcc_vector.device
    
    # 计算要掩码的元素数量
    num_mask = int(pcc_dim * mask_ratio)
    
    # 为每个样本生成不同的掩码
    masked_pcc = pcc_vector.clone()
    mask = torch.zeros(batch_size, pcc_dim, dtype=torch.bool, device=device)
    
    for i in range(batch_size):
        # 随机选择要掩码的元素索引
        mask_indices = torch.randperm(pcc_dim, device=device)[:num_mask]
        mask[i, mask_indices] = True
        # 将选中的元素置零
        masked_pcc[i, mask_indices] = 0
    
    # 获取被掩码位置的原始值
    target = pcc_vector.clone()
    
    if not is_batch:
        masked_pcc = masked_pcc.squeeze(0)
        mask = mask.squeeze(0)
        target = target.squeeze(0)
    
    return masked_pcc, mask, target


class ROIMaskTransform:
    """
    ROI-level 掩码变换类
    用于数据加载时的在线掩码
    """
    
    def __init__(self, mask_ratio=None):
        """
        Args:
            mask_ratio: 掩码比例，如果为 None 则随机选择 0.25 或 0.5
        """
        self.mask_ratio = mask_ratio
    
    def __call__(self, timeseries):
        """
        Args:
            timeseries: torch.Tensor, shape (T, n_rois)
        
        Returns:
            masked_ts: 掩码后的时间序列
            target: 原始时间序列
            mask: 掩码位置 (T, n_rois)
            roi_mask: ROI级别的掩码 (n_rois,)
        """
        masked_ts, mask, target, roi_mask = mask_roi_level(
            timeseries, self.mask_ratio
        )
        return masked_ts, target, mask, roi_mask


class PCCMaskTransform:
    """
    PCC 元素级掩码变换类
    用于数据加载时的在线掩码
    """
    
    def __init__(self, mask_ratio=0.15):
        """
        Args:
            mask_ratio: 掩码比例，默认 0.15
        """
        self.mask_ratio = mask_ratio
    
    def __call__(self, pcc_vector):
        """
        Args:
            pcc_vector: torch.Tensor, shape (pcc_dim,)
        
        Returns:
            masked_pcc: 掩码后的 PCC 向量
            target: 原始 PCC 向量
            mask: 掩码位置
        """
        masked_pcc, mask, target = mask_pcc_level(pcc_vector, self.mask_ratio)
        return masked_pcc, target, mask


def create_attention_mask_from_roi_mask(roi_mask, seq_len):
    """
    从 ROI 掩码创建注意力掩码
    被掩码的 ROI 对应的时间点不应该被其他位置看到
    
    Args:
        roi_mask: torch.Tensor, shape (batch, n_rois) 或 (n_rois,)
        seq_len: 序列长度（时间点数量）
    
    Returns:
        attn_mask: 注意力掩码，shape (batch, seq_len, seq_len) 或 (seq_len, seq_len)
    """
    is_batch = roi_mask.dim() == 2
    
    if not is_batch:
        roi_mask = roi_mask.unsqueeze(0)
    
    batch_size, n_rois = roi_mask.shape
    device = roi_mask.device
    
    # 对于 Transformer，我们通常不需要特殊的注意力掩码
    # 因为掩码的目的是让模型学习重建，而不是阻止信息流
    # 这里返回全零掩码（表示全部可见）
    attn_mask = torch.zeros(batch_size, seq_len, seq_len, device=device)
    
    if not is_batch:
        attn_mask = attn_mask.squeeze(0)
    
    return attn_mask


def batch_mask_roi_level(batch_timeseries, mask_ratio=None):
    """
    批量 ROI-level 掩码
    
    Args:
        batch_timeseries: torch.Tensor, shape (batch, T, n_rois)
        mask_ratio: 掩码比例
    
    Returns:
        masked_ts: 掩码后的时间序列
        mask: 掩码位置 (batch, T, n_rois)
        target: 原始时间序列
        roi_mask: ROI级别的掩码 (batch, n_rois)
    """
    return mask_roi_level(batch_timeseries, mask_ratio)


def batch_mask_pcc_level(batch_pcc_vectors, mask_ratio=0.15):
    """
    批量 PCC 元素级掩码
    
    Args:
        batch_pcc_vectors: torch.Tensor, shape (batch, pcc_dim)
        mask_ratio: 掩码比例
    
    Returns:
        masked_pcc: 掩码后的 PCC 向量
        mask: 掩码位置
        target: 原始 PCC 向量
    """
    return mask_pcc_level(batch_pcc_vectors, mask_ratio)


# 用于测试的辅助函数
def test_mask_functions():
    """测试掩码函数"""
    print("Testing ROI-level mask...")
    ts = torch.randn(4, 100, 200)  # batch=4, T=100, n_rois=200
    masked_ts, mask, target, roi_mask = mask_roi_level(ts, mask_ratio=0.25)
    print(f"  Input shape: {ts.shape}")
    print(f"  Masked shape: {masked_ts.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  ROI mask shape: {roi_mask.shape}")
    print(f"  Masked ROIs per sample: {roi_mask.sum(dim=1)}")
    
    print("\nTesting PCC-level mask...")
    pcc = torch.randn(4, 19900)  # batch=4, pcc_dim=19900
    masked_pcc, mask, target = mask_pcc_level(pcc, mask_ratio=0.15)
    print(f"  Input shape: {pcc.shape}")
    print(f"  Masked shape: {masked_pcc.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  Masked elements per sample: {mask.sum(dim=1)}")
    
    print("\nAll tests passed!")


if __name__ == '__main__':
    test_mask_functions()
