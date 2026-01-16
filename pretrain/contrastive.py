"""
对比学习模块
实现InfoNCE损失，对齐TST1和TST2的特征表示
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InfoNCELoss(nn.Module):
    """
    InfoNCE对比损失
    同一样本的TST1和TST2特征为正样本对
    不同样本的为负样本对
    """
    
    def __init__(self, temperature=0.07):
        """
        Args:
            temperature: 温度参数，控制分布的尖锐程度
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            loss: InfoNCE损失
        """
        batch_size = h_ts.shape[0]
        
        # L2归一化
        h_ts = F.normalize(h_ts, p=2, dim=1)
        h_fc = F.normalize(h_fc, p=2, dim=1)
        
        # 计算相似度矩阵
        # sim[i, j] = h_ts[i] · h_fc[j]
        sim_matrix = torch.matmul(h_ts, h_fc.T) / self.temperature  # (batch, batch)
        
        # 正样本对在对角线上
        labels = torch.arange(batch_size, device=h_ts.device)
        
        # 双向对比损失
        # TST1 -> TST2
        loss_ts2fc = F.cross_entropy(sim_matrix, labels)
        # TST2 -> TST1
        loss_fc2ts = F.cross_entropy(sim_matrix.T, labels)
        
        # 平均损失
        loss = (loss_ts2fc + loss_fc2ts) / 2
        
        return loss


class NTXentLoss(nn.Module):
    """
    NT-Xent损失 (Normalized Temperature-scaled Cross Entropy)
    SimCLR风格的对比损失
    """
    
    def __init__(self, temperature=0.5):
        """
        Args:
            temperature: 温度参数
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim)
            h_fc: TST2特征 (batch, dim)
        
        Returns:
            loss: NT-Xent损失
        """
        batch_size = h_ts.shape[0]
        
        # L2归一化
        h_ts = F.normalize(h_ts, p=2, dim=1)
        h_fc = F.normalize(h_fc, p=2, dim=1)
        
        # 拼接特征
        features = torch.cat([h_ts, h_fc], dim=0)  # (2*batch, dim)
        
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        
        # 创建标签：正样本对是(i, i+batch)和(i+batch, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(batch_size)
        ], dim=0).to(h_ts.device)
        
        # 移除对角线（自身相似度）
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=h_ts.device)
        sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
        
        # 计算损失
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class ProjectionHead(nn.Module):
    """
    投影头
    将特征投影到对比学习空间
    """
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=128):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
        """
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ContrastiveWrapper(nn.Module):
    """
    对比学习包装类
    包含投影头和对比损失
    """
    
    def __init__(
        self,
        dim_ts,
        dim_fc,
        proj_hidden_dim=256,
        proj_output_dim=128,
        temperature=0.07,
        loss_type='infonce'
    ):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            proj_hidden_dim: 投影头隐藏维度
            proj_output_dim: 投影头输出维度
            temperature: 温度参数
            loss_type: 损失类型 ('infonce' 或 'ntxent')
        """
        super().__init__()
        
        # 投影头
        self.proj_ts = ProjectionHead(dim_ts, proj_hidden_dim, proj_output_dim)
        self.proj_fc = ProjectionHead(dim_fc, proj_hidden_dim, proj_output_dim)
        
        # 对比损失
        if loss_type == 'infonce':
            self.criterion = InfoNCELoss(temperature)
        elif loss_type == 'ntxent':
            self.criterion = NTXentLoss(temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        self.loss_type = loss_type
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            loss: 对比损失
            z_ts: TST1投影特征
            z_fc: TST2投影特征
        """
        # 投影
        z_ts = self.proj_ts(h_ts)
        z_fc = self.proj_fc(h_fc)
        
        # 计算损失
        loss = self.criterion(z_ts, z_fc)
        
        return loss, z_ts, z_fc


def train_contrastive(
    model,
    dataloader,
    contrastive_module,
    optimizer,
    device,
    epochs=50
):
    """
    对比学习训练
    
    Args:
        model: 双流模型
        dataloader: 数据加载器
        contrastive_module: 对比学习模块
        optimizer: 优化器
        device: 设备
        epochs: 训练轮数
    
    Returns:
        losses: 每个epoch的损失列表
    """
    from tqdm import tqdm
    
    model.train()
    contrastive_module.train()
    losses = []
    
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}/{epochs}")
        for batch in pbar:
            timeseries = batch['timeseries'].to(device)
            pcc_vector = batch['pcc_vector'].to(device)
            
            # 获取特征
            h_ts, h_fc = model.get_features(timeseries, pcc_vector)
            
            # 计算对比损失
            optimizer.zero_grad()
            loss, z_ts, z_fc = contrastive_module(h_ts, h_fc)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"Epoch {epoch}: Average Loss = {avg_loss:.4f}")
    
    return losses


if __name__ == '__main__':
    # 测试对比学习模块
    print("Testing contrastive learning modules...")
    
    batch_size = 32
    dim_ts = 512
    dim_fc = 256
    
    h_ts = torch.randn(batch_size, dim_ts)
    h_fc = torch.randn(batch_size, dim_fc)
    
    # 测试InfoNCE
    print("\nTesting InfoNCELoss:")
    criterion = InfoNCELoss(temperature=0.07)
    loss = criterion(h_ts, h_fc)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试NT-Xent
    print("\nTesting NTXentLoss:")
    criterion = NTXentLoss(temperature=0.5)
    loss = criterion(h_ts, h_fc)
    print(f"  Loss: {loss.item():.4f}")
    
    # 测试ContrastiveWrapper
    print("\nTesting ContrastiveWrapper:")
    wrapper = ContrastiveWrapper(dim_ts, dim_fc)
    loss, z_ts, z_fc = wrapper(h_ts, h_fc)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Projected TST1 shape: {z_ts.shape}")
    print(f"  Projected TST2 shape: {z_fc.shape}")
    
    print("\nAll tests passed!")
