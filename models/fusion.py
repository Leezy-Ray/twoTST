"""
融合模块
实现多种特征融合策略：拼接、门控、交叉注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ConcatFusion(nn.Module):
    """
    简单拼接融合
    将两个特征向量直接拼接
    """
    
    def __init__(self, dim_ts, dim_fc, output_dim=None):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            output_dim: 输出维度（如果为None则直接拼接）
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim or (dim_ts + dim_fc)
        
        if output_dim is not None:
            self.proj = nn.Linear(dim_ts + dim_fc, output_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            fused: 融合后的特征 (batch, output_dim)
        """
        concat = torch.cat([h_ts, h_fc], dim=-1)
        return self.proj(concat)


class GatedFusion(nn.Module):
    """
    门控融合
    使用可学习的门控机制加权融合两个特征
    gate * h_ts + (1 - gate) * h_fc
    """
    
    def __init__(self, dim_ts, dim_fc, hidden_dim=None):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        
        # 将两个特征投影到相同维度
        self.output_dim = max(dim_ts, dim_fc)
        
        self.proj_ts = nn.Linear(dim_ts, self.output_dim)
        self.proj_fc = nn.Linear(dim_fc, self.output_dim)
        
        # 门控网络
        hidden_dim = hidden_dim or self.output_dim
        self.gate_net = nn.Sequential(
            nn.Linear(dim_ts + dim_fc, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            fused: 融合后的特征 (batch, output_dim)
        """
        # 投影到相同维度
        h_ts_proj = self.proj_ts(h_ts)
        h_fc_proj = self.proj_fc(h_fc)
        
        # 计算门控权重
        concat = torch.cat([h_ts, h_fc], dim=-1)
        gate = self.gate_net(concat)
        
        # 门控融合
        fused = gate * h_ts_proj + (1 - gate) * h_fc_proj
        return fused


class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合
    使用交叉注意力机制融合两个特征
    """
    
    def __init__(self, dim_ts, dim_fc, n_heads=8, dropout=0.1):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            n_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        
        # 统一到较大的维度
        self.d_model = max(dim_ts, dim_fc)
        
        # 投影层
        self.proj_ts = nn.Linear(dim_ts, self.d_model)
        self.proj_fc = nn.Linear(dim_fc, self.d_model)
        
        # 交叉注意力：TS -> FC
        self.cross_attn_ts2fc = nn.MultiheadAttention(
            self.d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 交叉注意力：FC -> TS
        self.cross_attn_fc2ts = nn.MultiheadAttention(
            self.d_model, n_heads, dropout=dropout, batch_first=True
        )
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(self.d_model)
        self.norm2 = nn.LayerNorm(self.d_model)
        self.norm3 = nn.LayerNorm(self.d_model)
        
        self.output_dim = self.d_model
    
    def forward(self, h_ts, h_fc, return_attention=False):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
            return_attention: 是否返回注意力权重
        
        Returns:
            fused: 融合后的特征 (batch, d_model)
            attention_weights (optional): 注意力权重dict
        """
        # 投影到相同维度并添加序列维度
        h_ts = self.proj_ts(h_ts).unsqueeze(1)  # (batch, 1, d_model)
        h_fc = self.proj_fc(h_fc).unsqueeze(1)  # (batch, 1, d_model)
        
        # 交叉注意力
        # TS作为Query，FC作为Key/Value
        attn_ts, attn_weights_ts2fc = self.cross_attn_ts2fc(h_ts, h_fc, h_fc)
        h_ts = self.norm1(h_ts + attn_ts)
        
        # FC作为Query，TS作为Key/Value
        attn_fc, attn_weights_fc2ts = self.cross_attn_fc2ts(h_fc, h_ts, h_ts)
        h_fc = self.norm2(h_fc + attn_fc)
        
        # 拼接并通过前馈网络
        concat = torch.cat([h_ts, h_fc], dim=-1)  # (batch, 1, d_model*2)
        fused = self.ffn(concat)  # (batch, 1, d_model)
        fused = self.norm3(fused)
        
        fused_out = fused.squeeze(1)  # (batch, d_model)
        
        if return_attention:
            attention_weights = {
                'ts2fc': attn_weights_ts2fc,  # (batch, n_heads, 1, 1)
                'fc2ts': attn_weights_fc2ts   # (batch, n_heads, 1, 1)
            }
            return fused_out, attention_weights
        else:
            return fused_out


class BilinearFusion(nn.Module):
    """
    双线性融合
    使用双线性变换融合两个特征
    """
    
    def __init__(self, dim_ts, dim_fc, output_dim=256):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            output_dim: 输出维度
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim
        
        # 双线性层
        self.bilinear = nn.Bilinear(dim_ts, dim_fc, output_dim)
        
        # 残差连接的投影
        self.proj_ts = nn.Linear(dim_ts, output_dim)
        self.proj_fc = nn.Linear(dim_fc, output_dim)
        
        # 层归一化
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            fused: 融合后的特征 (batch, output_dim)
        """
        # 双线性变换
        bilinear_out = self.bilinear(h_ts, h_fc)
        
        # 残差
        residual = self.proj_ts(h_ts) + self.proj_fc(h_fc)
        
        # 融合
        fused = self.norm(bilinear_out + residual)
        return fused


class AttentionPoolingFusion(nn.Module):
    """
    注意力池化融合
    使用注意力机制对两个特征进行加权融合
    """
    
    def __init__(self, dim_ts, dim_fc, hidden_dim=None):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            hidden_dim: 隐藏层维度
        """
        super().__init__()
        
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = max(dim_ts, dim_fc)
        
        # 投影到相同维度
        self.proj_ts = nn.Linear(dim_ts, self.output_dim)
        self.proj_fc = nn.Linear(dim_fc, self.output_dim)
        
        # 注意力权重计算
        hidden_dim = hidden_dim or self.output_dim
        self.attention = nn.Sequential(
            nn.Linear(self.output_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        
        Returns:
            fused: 融合后的特征 (batch, output_dim)
        """
        # 投影到相同维度
        h_ts = self.proj_ts(h_ts)
        h_fc = self.proj_fc(h_fc)
        
        # 堆叠成序列 (batch, 2, output_dim)
        features = torch.stack([h_ts, h_fc], dim=1)
        
        # 计算注意力权重
        attn_scores = self.attention(features)  # (batch, 2, 1)
        attn_weights = F.softmax(attn_scores, dim=1)
        
        # 加权融合
        fused = (features * attn_weights).sum(dim=1)  # (batch, output_dim)
        return fused


class GatedMultiFusion(nn.Module):
    """
    门控多策略融合
    并行运行5种融合策略(concat, gated, cross_attention, bilinear, attention_pooling)，
    学习门控权重矩阵，灵活选择一种或多种策略进行融合。
    改进：加深门控、初始化偏向 attention_pooling、存储 gate_weights 供熵正则。
    """

    def __init__(self, dim_ts, dim_fc, output_dim=256, hidden_dim=128, dropout=0.1,
                 init_favor_idx=4):
        """
        Args:
            dim_ts: TST1特征维度
            dim_fc: TST2特征维度
            output_dim: 最终融合输出维度
            hidden_dim: 各融合策略内部隐藏维度
            dropout: Dropout比例
            init_favor_idx: 初始偏向的策略索引 (4=attention_pooling，单策略最优)
        """
        super().__init__()
        self.dim_ts = dim_ts
        self.dim_fc = dim_fc
        self.output_dim = output_dim
        self.n_strategies = 5
        self.init_favor_idx = init_favor_idx
        self.last_gate_weights = None  # 供熵正则使用

        # 5种融合策略
        self.fusions = nn.ModuleList([
            ConcatFusion(dim_ts, dim_fc, output_dim=hidden_dim),
            GatedFusion(dim_ts, dim_fc, hidden_dim=hidden_dim),
            CrossAttentionFusion(dim_ts, dim_fc, n_heads=8, dropout=dropout),
            BilinearFusion(dim_ts, dim_fc, output_dim=hidden_dim),
            AttentionPoolingFusion(dim_ts, dim_fc, hidden_dim=hidden_dim),
        ])

        # 各融合输出维度可能不同，统一投影到 output_dim
        fusion_dims = [
            hidden_dim,  # concat
            max(dim_ts, dim_fc),  # gated
            max(dim_ts, dim_fc),  # cross_attention
            hidden_dim,  # bilinear
            max(dim_ts, dim_fc),  # attention_pooling
        ]
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, output_dim),
                nn.LayerNorm(output_dim),
                nn.GELU(),
            )
            for d in fusion_dims
        ])

        # 加深的门控网络
        self.gate_net = nn.Sequential(
            nn.Linear(dim_ts + dim_fc, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.n_strategies),
        )
        self._init_gate_favor_strategy()

    def _init_gate_favor_strategy(self):
        """初始化门控：使 init_favor_idx 策略（如 attention_pooling）初始权重更高"""
        last_linear = self.gate_net[-1]
        with torch.no_grad():
            last_linear.weight.data *= 0.1
            bias = torch.zeros(self.n_strategies)
            bias[self.init_favor_idx] = 2.0
            last_linear.bias.data.copy_(bias)

    def forward(self, h_ts, h_fc):
        """
        Args:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)

        Returns:
            fused: 融合后的特征 (batch, output_dim)
        """
        # 并行计算5种融合输出
        fusion_outputs = []
        for i, fusion in enumerate(self.fusions):
            out = fusion(h_ts, h_fc)
            out = self.projections[i](out)
            fusion_outputs.append(out)

        # stack: (batch, 5, output_dim)
        stacked = torch.stack(fusion_outputs, dim=1)

        # 门控权重: (batch, 5)
        gate_input = torch.cat([h_ts, h_fc], dim=-1)
        gate_logits = self.gate_net(gate_input)
        gate_weights = F.softmax(gate_logits, dim=-1)
        self.last_gate_weights = gate_weights  # 供熵正则使用

        # 加权求和: (batch, 5, output_dim) * (batch, 5, 1) -> (batch, output_dim)
        fused = (stacked * gate_weights.unsqueeze(-1)).sum(dim=1)

        return fused


def create_fusion_module(fusion_type, dim_ts, dim_fc, **kwargs):
    """
    创建融合模块的工厂函数
    
    Args:
        fusion_type: 融合类型 ('concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling')
        dim_ts: TST1特征维度
        dim_fc: TST2特征维度
        **kwargs: 其他参数
    
    Returns:
        fusion_module: 融合模块实例
    """
    fusion_classes = {
        'concat': ConcatFusion,
        'gated': GatedFusion,
        'cross_attention': CrossAttentionFusion,
        'bilinear': BilinearFusion,
        'attention_pooling': AttentionPoolingFusion,
        'gated_multi': GatedMultiFusion,
    }
    
    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}. "
                        f"Available: {list(fusion_classes.keys())}")
    
    return fusion_classes[fusion_type](dim_ts, dim_fc, **kwargs)


if __name__ == '__main__':
    # 测试各种融合模块
    batch_size = 4
    dim_ts = 512
    dim_fc = 256
    
    h_ts = torch.randn(batch_size, dim_ts)
    h_fc = torch.randn(batch_size, dim_fc)
    
    print("Testing fusion modules...")
    
    for fusion_type in ['concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling']:
        print(f"\n{fusion_type}:")
        fusion = create_fusion_module(fusion_type, dim_ts, dim_fc)
        output = fusion(h_ts, h_fc)
        print(f"  Output shape: {output.shape}")
        print(f"  Output dim: {fusion.output_dim}")
    
    print("\nAll tests passed!")
