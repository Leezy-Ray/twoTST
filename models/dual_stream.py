"""
双流模型
整合TST1、TST2和融合模块的完整模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer_ts import TransformerTS, create_transformer_ts
from .transformer_fc import TransformerFC, create_transformer_fc
from .fusion import create_fusion_module


class DualStreamModel(nn.Module):
    """
    双流自监督预训练模型
    
    包含:
    - TST1: 时序Transformer，处理fMRI时间序列
    - TST2: 连接Transformer，处理PCC向量
    - 融合模块: 融合两个Transformer的特征
    - 分类头: MLP分类器
    """
    
    def __init__(
        self,
        tst1_config=None,
        tst2_config=None,
        fusion_type='cross_attention',
        fusion_config=None,
        num_classes=2,
        dropout=0.1
    ):
        """
        Args:
            tst1_config: TST1配置
            tst2_config: TST2配置
            fusion_type: 融合类型
            fusion_config: 融合模块配置
            num_classes: 分类类别数
            dropout: Dropout比例
        """
        super().__init__()
        
        # 创建TST1
        self.transformer_ts = create_transformer_ts(tst1_config)
        
        # 创建TST2
        self.transformer_fc = create_transformer_fc(tst2_config)
        
        # 获取特征维度
        self.dim_ts = self.transformer_ts.emb_dim
        self.dim_fc = self.transformer_fc.d_model
        
        # 创建融合模块
        fusion_config = fusion_config or {}
        self.fusion = create_fusion_module(
            fusion_type, self.dim_ts, self.dim_fc, **fusion_config
        )
        self.fusion_type = fusion_type
        
        # 分类头
        fusion_dim = self.fusion.output_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, timeseries, pcc_vector, return_features=False):
        """
        Args:
            timeseries: 时间序列 (batch, T, n_rois)
            pcc_vector: PCC向量 (batch, pcc_dim)
            return_features: 是否返回中间特征
        
        Returns:
            logits: 分类logits (batch, num_classes)
            features (optional): 融合特征 (batch, fusion_dim)
        """
        # 获取TST1特征
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        
        # 获取TST2特征
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')
        
        # 融合
        fused = self.fusion(h_ts, h_fc)
        
        # 分类
        logits = self.classifier(fused)
        
        if return_features:
            return logits, fused, h_ts, h_fc
        return logits
    
    def get_features(self, timeseries, pcc_vector):
        """
        获取两个Transformer的特征（用于对比学习）
        
        Args:
            timeseries: 时间序列 (batch, T, n_rois)
            pcc_vector: PCC向量 (batch, pcc_dim)
        
        Returns:
            h_ts: TST1特征 (batch, dim_ts)
            h_fc: TST2特征 (batch, dim_fc)
        """
        h_ts = self.transformer_ts(timeseries, mode='finetune')
        h_fc = self.transformer_fc(pcc_vector, mode='finetune')
        return h_ts, h_fc
    
    def load_pretrained_tst1(self, checkpoint_path, strict=False):
        """加载TST1预训练权重"""
        self.transformer_ts.load_pretrained(checkpoint_path, strict=strict)
    
    def load_pretrained_tst2(self, checkpoint_path, strict=False):
        """加载TST2预训练权重"""
        self.transformer_fc.load_pretrained(checkpoint_path, strict=strict)
    
    def freeze_encoders(self):
        """冻结两个Transformer编码器"""
        for param in self.transformer_ts.parameters():
            param.requires_grad = False
        for param in self.transformer_fc.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """解冻两个Transformer编码器"""
        for param in self.transformer_ts.parameters():
            param.requires_grad = True
        for param in self.transformer_fc.parameters():
            param.requires_grad = True


class DualStreamModelSingleBranch(nn.Module):
    """
    单分支模型（用于消融实验）
    只使用TST1或TST2
    """
    
    def __init__(
        self,
        branch='ts',
        tst_config=None,
        num_classes=2,
        dropout=0.1
    ):
        """
        Args:
            branch: 使用哪个分支 ('ts' 或 'fc')
            tst_config: Transformer配置
            num_classes: 分类类别数
            dropout: Dropout比例
        """
        super().__init__()
        
        self.branch = branch
        
        if branch == 'ts':
            self.transformer = create_transformer_ts(tst_config)
            feature_dim = self.transformer.emb_dim
        elif branch == 'fc':
            self.transformer = create_transformer_fc(tst_config)
            feature_dim = self.transformer.d_model
        else:
            raise ValueError(f"Unknown branch: {branch}")
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes)
        )
        
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Args:
            x: 输入数据
               - ts分支: (batch, T, n_rois)
               - fc分支: (batch, pcc_dim)
        
        Returns:
            logits: 分类logits (batch, num_classes)
        """
        features = self.transformer(x, mode='finetune')
        logits = self.classifier(features)
        return logits


def create_dual_stream_model(
    n_rois=200,
    time_points=100,
    pcc_dim=19900,
    tst1_emb_dim=512,
    tst2_d_model=256,
    fusion_type='cross_attention',
    num_classes=2,
    dropout=0.1
):
    """
    创建双流模型的便捷函数
    
    Args:
        n_rois: ROI数量
        time_points: 时间点数
        pcc_dim: PCC向量维度
        tst1_emb_dim: TST1嵌入维度
        tst2_d_model: TST2模型维度
        fusion_type: 融合类型
        num_classes: 分类类别数
        dropout: Dropout比例
    
    Returns:
        model: DualStreamModel实例
    """
    tst1_config = {
        'n_rois': n_rois,
        'emb_dim': tst1_emb_dim,
        'n_heads': 8,
        'n_layers': 6,
        'dim_feedforward': 2048,
        'dropout': dropout,
        'max_seq_len': time_points,
        'use_cls_token': True
    }
    
    tst2_config = {
        'pcc_dim': pcc_dim,
        'd_model': tst2_d_model,
        'n_heads': 8,
        'n_layers': 2,
        'dim_feedforward': 512,
        'dropout': dropout
    }
    
    return DualStreamModel(
        tst1_config=tst1_config,
        tst2_config=tst2_config,
        fusion_type=fusion_type,
        num_classes=num_classes,
        dropout=dropout
    )


if __name__ == '__main__':
    # 测试双流模型
    print("Testing DualStreamModel...")
    
    model = create_dual_stream_model()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 测试前向传播
    batch_size = 4
    timeseries = torch.randn(batch_size, 100, 200)
    pcc_vector = torch.randn(batch_size, 19900)
    
    logits = model(timeseries, pcc_vector)
    print(f"Logits shape: {logits.shape}")
    
    # 测试返回特征
    logits, fused, h_ts, h_fc = model(timeseries, pcc_vector, return_features=True)
    print(f"Fused features shape: {fused.shape}")
    print(f"TST1 features shape: {h_ts.shape}")
    print(f"TST2 features shape: {h_fc.shape}")
    
    # 测试不同融合策略
    print("\nTesting different fusion types:")
    for fusion_type in ['concat', 'gated', 'cross_attention', 'bilinear', 'attention_pooling']:
        model = create_dual_stream_model(fusion_type=fusion_type)
        logits = model(timeseries, pcc_vector)
        print(f"  {fusion_type}: logits shape = {logits.shape}")
    
    # 测试单分支模型
    print("\nTesting single branch models:")
    
    # TST1单分支
    ts_model = DualStreamModelSingleBranch(branch='ts')
    ts_logits = ts_model(timeseries)
    print(f"  TS branch: logits shape = {ts_logits.shape}")
    
    # TST2单分支
    fc_model = DualStreamModelSingleBranch(branch='fc')
    fc_logits = fc_model(pcc_vector)
    print(f"  FC branch: logits shape = {fc_logits.shape}")
    
    print("\nAll tests passed!")
