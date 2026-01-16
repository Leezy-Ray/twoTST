"""
TST2: 连接Transformer
处理PCC上三角向量，使用元素级掩码策略进行预训练
"""

import math
import torch
import torch.nn as nn


class TransformerFC(nn.Module):
    """
    连接Transformer (TST2)
    
    输入: (batch, pcc_dim) - PCC上三角向量
    输出:
        - pretrain模式: 重建的PCC向量 (batch, pcc_dim)
        - finetune模式: 特征向量 (batch, d_model)
    """
    
    def __init__(
        self,
        pcc_dim=19900,
        d_model=256,
        n_heads=8,
        n_layers=2,
        dim_feedforward=512,
        dropout=0.1
    ):
        super().__init__()
        
        self.pcc_dim = pcc_dim
        self.d_model = d_model
        
        # 输入嵌入层：将PCC向量映射到嵌入空间
        self.input_embedding = nn.Linear(pcc_dim, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        
        # 预训练解码器：重建PCC向量
        self.pretrain_decoder = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, pcc_dim)
        )
        
        # 激活函数和Dropout
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        
        # 层归一化
        self.norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, mode='pretrain'):
        """
        Args:
            x: Tensor, shape (batch, pcc_dim)
            mode: 'pretrain' 或 'finetune'
        
        Returns:
            pretrain模式: 重建的PCC向量 (batch, pcc_dim)
            finetune模式: 特征向量 (batch, d_model)
        """
        # 输入嵌入
        x = self.input_embedding(x) / math.sqrt(self.d_model)
        
        # 添加假的序列维度 (batch, 1, d_model)
        x = x.unsqueeze(1)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        
        # 去掉序列维度 (batch, d_model)
        x = x.squeeze(1)
        
        # 激活和归一化
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        
        if mode == 'finetune':
            # 返回特征向量
            return x  # (batch, d_model)
        else:
            # pretrain模式：重建PCC向量
            output = self.pretrain_decoder(x)  # (batch, pcc_dim)
            return output
    
    def get_features(self, x):
        """
        获取特征表示（用于对比学习）
        
        Args:
            x: Tensor, shape (batch, pcc_dim)
        
        Returns:
            features: Tensor, shape (batch, d_model)
        """
        return self.forward(x, mode='finetune')
    
    def load_pretrained(self, checkpoint_path, strict=True):
        """
        加载预训练权重
        
        Args:
            checkpoint_path: 预训练权重路径
            strict: 是否严格匹配
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # 过滤掉decoder的权重（微调时不需要）
        if not strict:
            state_dict = {
                k: v for k, v in state_dict.items() 
                if 'pretrain_decoder' not in k
            }
        
        self.load_state_dict(state_dict, strict=strict)
        print(f"Loaded pretrained weights from {checkpoint_path}")


class TransformerFCForPretrain(nn.Module):
    """
    用于预训练的TST2包装类
    包含掩码逻辑和损失计算
    """
    
    def __init__(self, transformer_fc):
        super().__init__()
        self.transformer = transformer_fc
    
    def forward(self, x, masked_x, mask):
        """
        Args:
            x: 原始PCC向量 (batch, pcc_dim)
            masked_x: 掩码后的PCC向量 (batch, pcc_dim)
            mask: 掩码位置 (batch, pcc_dim)
        
        Returns:
            loss: 重建损失
            pred: 预测的PCC向量
        """
        # 前向传播
        pred = self.transformer(masked_x, mode='pretrain')
        
        # 计算掩码位置的MSE损失
        loss = nn.functional.mse_loss(
            pred[mask], x[mask], reduction='mean'
        )
        
        return loss, pred


class MaskedMSELoss(nn.Module):
    """
    掩码MSE损失
    只计算被掩码位置的重建损失
    """
    
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)
    
    def forward(self, y_pred, y_true, mask):
        """
        Args:
            y_pred: 预测值
            y_true: 真实值
            mask: 布尔掩码，True表示被掩码的位置
        
        Returns:
            loss: 掩码位置的MSE损失
        """
        masked_pred = torch.masked_select(y_pred, mask)
        masked_true = torch.masked_select(y_true, mask)
        
        return self.mse_loss(masked_pred, masked_true)


def create_transformer_fc(config=None):
    """
    创建TST2模型的工厂函数
    
    Args:
        config: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: TransformerFC实例
    """
    default_config = {
        'pcc_dim': 19900,
        'd_model': 256,
        'n_heads': 8,
        'n_layers': 2,
        'dim_feedforward': 512,
        'dropout': 0.1
    }
    
    if config is not None:
        default_config.update(config)
    
    return TransformerFC(**default_config)


if __name__ == '__main__':
    # 测试模型
    print("Testing TransformerFC...")
    
    model = create_transformer_fc()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试pretrain模式
    x = torch.randn(4, 19900)  # batch=4, pcc_dim=19900
    output = model(x, mode='pretrain')
    print(f"Pretrain output shape: {output.shape}")  # 应该是 (4, 19900)
    
    # 测试finetune模式
    features = model(x, mode='finetune')
    print(f"Finetune output shape: {features.shape}")  # 应该是 (4, 256)
    
    # 测试MaskedMSELoss
    print("\nTesting MaskedMSELoss...")
    criterion = MaskedMSELoss()
    pred = torch.randn(4, 19900)
    target = torch.randn(4, 19900)
    mask = torch.rand(4, 19900) > 0.85  # 15%掩码
    loss = criterion(pred, target, mask)
    print(f"Loss: {loss.item():.4f}")
    
    print("\nAll tests passed!")
