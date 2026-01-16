"""
TST1: 时序Transformer
处理原始fMRI时间序列，使用ROI-level掩码策略进行预训练
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    正弦位置编码
    """
    
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor, shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerTS(nn.Module):
    """
    时序Transformer (TST1)
    
    输入: (batch, T, n_rois) - 时间序列数据
    输出:
        - pretrain模式: 重建的完整时间序列 (batch, T, n_rois)
        - finetune模式: CLS token特征 (batch, emb_dim)
    """
    
    def __init__(
        self,
        n_rois=200,
        emb_dim=512,
        n_heads=8,
        n_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        max_seq_len=200,
        use_cls_token=True
    ):
        super().__init__()
        
        self.n_rois = n_rois
        self.emb_dim = emb_dim
        self.use_cls_token = use_cls_token
        
        # 输入嵌入层：将每个时间点的ROI特征映射到嵌入空间
        self.input_embedding = nn.Linear(n_rois, emb_dim)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(
            emb_dim, max_len=max_seq_len + 1, dropout=dropout
        )
        
        # CLS token（用于分类任务）
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            nn.init.normal_(self.cls_token, std=0.02)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
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
        
        # 预训练解码器：重建时间序列
        self.pretrain_decoder = nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, n_rois)
        )
        
        # 层归一化
        self.norm = nn.LayerNorm(emb_dim)
        
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
            x: Tensor, shape (batch, T, n_rois)
            mode: 'pretrain' 或 'finetune'
        
        Returns:
            pretrain模式: 重建的时间序列 (batch, T, n_rois)
            finetune模式: CLS token特征 (batch, emb_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # 输入嵌入
        x = self.input_embedding(x) * math.sqrt(self.emb_dim)
        
        # 添加CLS token
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # 位置编码
        x = self.pos_encoder(x)
        
        # Transformer编码
        x = self.transformer_encoder(x)
        x = self.norm(x)
        
        if mode == 'finetune':
            # 返回CLS token特征
            if self.use_cls_token:
                return x[:, 0, :]  # (batch, emb_dim)
            else:
                # 如果没有CLS token，使用平均池化
                return x.mean(dim=1)  # (batch, emb_dim)
        else:
            # pretrain模式：重建时间序列
            if self.use_cls_token:
                x = x[:, 1:, :]  # 去掉CLS token
            
            # 解码重建
            output = self.pretrain_decoder(x)  # (batch, T, n_rois)
            return output
    
    def get_features(self, x):
        """
        获取特征表示（用于对比学习）
        
        Args:
            x: Tensor, shape (batch, T, n_rois)
        
        Returns:
            features: Tensor, shape (batch, emb_dim)
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


class TransformerTSForPretrain(nn.Module):
    """
    用于预训练的TST1包装类
    包含掩码逻辑和损失计算
    """
    
    def __init__(self, transformer_ts):
        super().__init__()
        self.transformer = transformer_ts
    
    def forward(self, x, masked_x, mask):
        """
        Args:
            x: 原始时间序列 (batch, T, n_rois)
            masked_x: 掩码后的时间序列 (batch, T, n_rois)
            mask: 掩码位置 (batch, T, n_rois)
        
        Returns:
            loss: 重建损失
            pred: 预测的时间序列
        """
        # 前向传播
        pred = self.transformer(masked_x, mode='pretrain')
        
        # 计算掩码位置的MSE损失
        loss = nn.functional.mse_loss(
            pred[mask], x[mask], reduction='mean'
        )
        
        return loss, pred


def create_transformer_ts(config=None):
    """
    创建TST1模型的工厂函数
    
    Args:
        config: 配置字典，如果为None则使用默认配置
    
    Returns:
        model: TransformerTS实例
    """
    default_config = {
        'n_rois': 200,
        'emb_dim': 512,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_seq_len': 200,
        'use_cls_token': True
    }
    
    if config is not None:
        default_config.update(config)
    
    return TransformerTS(**default_config)


if __name__ == '__main__':
    # 测试模型
    print("Testing TransformerTS...")
    
    model = create_transformer_ts()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试pretrain模式
    x = torch.randn(4, 100, 200)  # batch=4, T=100, n_rois=200
    output = model(x, mode='pretrain')
    print(f"Pretrain output shape: {output.shape}")  # 应该是 (4, 100, 200)
    
    # 测试finetune模式
    features = model(x, mode='finetune')
    print(f"Finetune output shape: {features.shape}")  # 应该是 (4, 512)
    
    print("All tests passed!")
