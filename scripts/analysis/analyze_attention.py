"""
注意力权重分析脚本
提取交叉注意力权重，分析CC200脑区的异常连接模式
"""

import os
import sys
import pickle
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from models.dual_stream import DualStreamModel
from utils.data_loader import TwoTSTDataset


def pcc_vector_to_matrix(pcc_vector, n_rois=200):
    """
    将PCC向量（上三角）转换为200x200的连接矩阵
    
    Args:
        pcc_vector: ndarray, shape (pcc_dim,) 或 (batch, pcc_dim)
        n_rois: ROI数量（默认200）
    
    Returns:
        pcc_matrix: ndarray, shape (n_rois, n_rois) 或 (batch, n_rois, n_rois)
    """
    if pcc_vector.ndim == 1:
        # 单个向量
        matrix = np.zeros((n_rois, n_rois))
        upper_indices = np.triu_indices(n_rois, k=1)
        matrix[upper_indices] = pcc_vector
        # 对称填充
        matrix = matrix + matrix.T
        # 对角线设为1（自身连接）
        np.fill_diagonal(matrix, 1.0)
        return matrix
    else:
        # 批量向量
        batch_size = pcc_vector.shape[0]
        matrices = np.zeros((batch_size, n_rois, n_rois))
        upper_indices = np.triu_indices(n_rois, k=1)
        for i in range(batch_size):
            matrices[i][upper_indices] = pcc_vector[i]
            matrices[i] = matrices[i] + matrices[i].T
            np.fill_diagonal(matrices[i], 1.0)
        return matrices


def extract_attention_weights(model, windowed_data, device='cuda', batch_size=32):
    """
    从模型中提取注意力权重
    
    Args:
        model: 加载的模型
        windowed_data: 窗口化数据dict
        device: 设备
        batch_size: 批次大小
    
    Returns:
        attention_results: dict包含所有注意力权重
    """
    print(f"\nExtracting attention weights with batch_size={batch_size}...")
    
    timeseries = windowed_data['timeseries']
    pcc_vectors = windowed_data['pcc_vectors']
    labels = windowed_data['labels']
    subject_ids = windowed_data['subject_ids']
    window_indices = windowed_data['window_indices']
    
    # 创建数据集
    dataset = TwoTSTDataset(timeseries, pcc_vectors, labels, 
                           normalize_ts=True, normalize_pcc=True)
    
    # 数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_attention_weights = []
    all_logits = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Extracting attention'):
            ts = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            
            # 使用模型的forward方法获取注意力权重
            result = model(ts, pcc, return_attention=True)
            if isinstance(result, tuple):
                logits = result[0]
                # 注意力权重应该是最后一个元素（如果是dict类型）
                attention_weights = None
                for item in reversed(result[1:]):
                    if isinstance(item, dict):
                        attention_weights = item
                        break
            else:
                logits = result
                attention_weights = None
            else:
                # 获取TST1和TST2特征
                h_ts = model.transformer_ts(ts, mode='finetune')
                h_fc = model.transformer_fc(pcc, mode='finetune')
                
                # 通过融合层获取注意力权重
                if hasattr(model.fusion, 'forward'):
                    fused, attention_weights = model.fusion(
                        h_ts, h_fc, return_attention=True
                    )
                else:
                    fused = model.fusion(h_ts, h_fc)
                    attention_weights = None
                
                # 分类
                logits = model.classifier(fused)
            probs = F.softmax(logits, dim=1)
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            
            if attention_weights is not None:
                # 将注意力权重转换为numpy
                attn_dict = {
                    'ts2fc': attention_weights['ts2fc'].cpu().numpy(),
                    'fc2ts': attention_weights['fc2ts'].cpu().numpy()
                }
                all_attention_weights.append(attn_dict)
            else:
                all_attention_weights.append(None)
    
    # 组织结果
    attention_results = {
        'subject_ids': subject_ids,
        'window_indices': window_indices,
        'labels': labels.tolist(),
        'asd_probs': [float(p) for p in all_probs],
        'attention_weights': all_attention_weights,
        'logits': [l.tolist() for l in all_logits]
    }
    
    print(f"\nExtracted attention weights for {len(all_attention_weights)} windows")
    
    return attention_results


def analyze_abnormal_connections(attention_results, pcc_vectors, n_rois=200):
    """
    分析异常连接模式
    
    Args:
        attention_results: 注意力权重结果
        pcc_vectors: PCC向量数组，shape (n_windows, pcc_dim)
        n_rois: ROI数量
    
    Returns:
        analysis: dict包含异常连接分析结果
    """
    print(f"\nAnalyzing abnormal connections...")
    
    labels = np.array(attention_results['labels'])
    attention_weights = attention_results['attention_weights']
    
    # 分离ASD和TC组
    asd_mask = labels == 1
    tc_mask = labels == 0
    
    asd_indices = np.where(asd_mask)[0]
    tc_indices = np.where(tc_mask)[0]
    
    print(f"  ASD windows: {len(asd_indices)}")
    print(f"  TC windows: {len(tc_indices)}")
    
    # 提取注意力权重
    # 注意：注意力权重形状是 (batch, n_heads, 1, 1)，这里是单个token之间的注意力
    # 对于PCC向量的分析，我们需要分析TST2内部的自注意力或者交叉注意力的模式
    
    # 但是，CrossAttentionFusion的注意力是TS和FC之间的交互
    # 我们需要的是TST2（处理PCC向量）内部的自注意力来理解哪些连接更重要
    
    # 由于模型结构限制，我们分析：
    # 1. 注意力权重的统计特征
    # 2. PCC向量本身的模式差异
    
    # 分析PCC向量差异
    asd_pcc = pcc_vectors[asd_indices]
    tc_pcc = pcc_vectors[tc_indices]
    
    # 计算平均PCC向量
    asd_mean_pcc = np.mean(asd_pcc, axis=0)
    tc_mean_pcc = np.mean(tc_pcc, axis=0)
    
    # 计算差异
    pcc_diff = asd_mean_pcc - tc_mean_pcc
    
    # 转换为连接矩阵
    asd_mean_matrix = pcc_vector_to_matrix(asd_mean_pcc, n_rois)
    tc_mean_matrix = pcc_vector_to_matrix(tc_mean_pcc, n_rois)
    diff_matrix = pcc_vector_to_matrix(pcc_diff, n_rois)
    
    # 找出差异最大的连接（绝对值）
    diff_abs = np.abs(diff_matrix)
    # 上三角索引
    upper_indices = np.triu_indices(n_rois, k=1)
    
    # 找出top-k差异最大的连接
    top_k = 50
    diff_upper = diff_abs[upper_indices]
    top_k_indices = np.argsort(diff_upper)[-top_k:]
    
    # 转换为(i, j)坐标
    top_k_connections = [(upper_indices[0][idx], upper_indices[1][idx]) 
                         for idx in top_k_indices]
    top_k_values = [diff_upper[idx] for idx in top_k_indices]
    
    # 分析注意力权重统计
    attention_stats = {
        'asd': {'ts2fc_mean': None, 'fc2ts_mean': None},
        'tc': {'ts2fc_mean': None, 'fc2ts_mean': None}
    }
    
    if attention_weights[0] is not None:
        # 收集ASD和TC的注意力权重
        asd_attn_ts2fc = []
        asd_attn_fc2ts = []
        tc_attn_ts2fc = []
        tc_attn_fc2ts = []
        
        for i, attn in enumerate(attention_weights):
            if attn is not None:
                if labels[i] == 1:  # ASD
                    asd_attn_ts2fc.append(attn['ts2fc'])
                    asd_attn_fc2ts.append(attn['fc2ts'])
                else:  # TC
                    tc_attn_ts2fc.append(attn['ts2fc'])
                    tc_attn_fc2ts.append(attn['fc2ts'])
        
        if len(asd_attn_ts2fc) > 0:
            asd_attn_ts2fc = np.concatenate(asd_attn_ts2fc, axis=0)
            asd_attn_fc2ts = np.concatenate(asd_attn_fc2ts, axis=0)
            attention_stats['asd']['ts2fc_mean'] = float(np.mean(asd_attn_ts2fc))
            attention_stats['asd']['fc2ts_mean'] = float(np.mean(asd_attn_fc2ts))
        
        if len(tc_attn_ts2fc) > 0:
            tc_attn_ts2fc = np.concatenate(tc_attn_ts2fc, axis=0)
            tc_attn_fc2ts = np.concatenate(tc_attn_fc2ts, axis=0)
            attention_stats['tc']['ts2fc_mean'] = float(np.mean(tc_attn_ts2fc))
            attention_stats['tc']['fc2ts_mean'] = float(np.mean(tc_attn_fc2ts))
    
    # 组织分析结果
    analysis = {
        'pcc_analysis': {
            'asd_mean_pcc': asd_mean_pcc.tolist(),
            'tc_mean_pcc': tc_mean_pcc.tolist(),
            'pcc_diff': pcc_diff.tolist(),
            'top_k_connections': [
                {'roi_i': int(i), 'roi_j': int(j), 'diff': float(val)}
                for (i, j), val in zip(top_k_connections, top_k_values)
            ],
            'asd_mean_matrix': asd_mean_matrix.tolist(),
            'tc_mean_matrix': tc_mean_matrix.tolist(),
            'diff_matrix': diff_matrix.tolist()
        },
        'attention_stats': attention_stats,
        'summary': {
            'n_asd_windows': int(len(asd_indices)),
            'n_tc_windows': int(len(tc_indices)),
            'n_rois': n_rois,
            'top_k_connections_count': top_k
        }
    }
    
    print(f"\nAnalysis summary:")
    print(f"  Top {top_k} connections with largest differences:")
    for i, (conn, val) in enumerate(zip(top_k_connections[-10:], top_k_values[-10:])):
        print(f"    {i+1}. ROI {conn[0]} <-> ROI {conn[1]}: diff={val:.6f}")
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description='Analyze attention weights and abnormal connections')
    parser.add_argument('--windowed_data', type=str,
                        default=None,
                        help='Path to windowed data pickle file (if not provided, will load from extracted data)')
    parser.add_argument('--extracted_data', type=str,
                        default='/root/workplace/exp/TwoTST/data/extracted/extracted_subjects.pkl',
                        help='Path to extracted_subjects.pkl')
    parser.add_argument('--checkpoint', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints_sw/finetune/sw_baseline_cross_attention/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--results_json', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints_sw/finetune/sw_baseline_cross_attention/results.json',
                        help='Path to results.json')
    parser.add_argument('--window_size', type=int, default=32,
                        help='Sliding window size')
    parser.add_argument('--stride', type=int, default=16,
                        help='Sliding window stride')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    if args.windowed_data and os.path.exists(args.windowed_data):
        print(f"\nLoading windowed data from {args.windowed_data}...")
        with open(args.windowed_data, 'rb') as f:
            windowed_data = pickle.load(f)
    else:
        # 从提取的数据创建窗口化数据
        print(f"\nLoading extracted data from {args.extracted_data}...")
        with open(args.extracted_data, 'rb') as f:
            extracted_data = pickle.load(f)
        
        # 需要导入predict_with_sliding_window的apply_sliding_window函数
        # 为了简化，我们在这里直接应用滑动窗口
        from scripts.analysis.predict_with_sliding_window import apply_sliding_window
        
        windowed_data = apply_sliding_window(
            extracted_data['timeseries'],
            extracted_data['pcc_vectors'],
            extracted_data['labels'],
            extracted_data['subject_ids'],
            window_size=args.window_size,
            stride=args.stride
        )
    
    # 加载模型
    from scripts.analysis.predict_with_sliding_window import load_model
    model, config = load_model(args.checkpoint, args.results_json, device)
    
    # 提取注意力权重
    attention_results = extract_attention_weights(
        model, windowed_data, device, args.batch_size
    )
    
    # 分析异常连接
    analysis = analyze_abnormal_connections(
        attention_results, 
        windowed_data['pcc_vectors'],
        n_rois=200
    )
    
    # 保存结果
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'data', 'extracted', 'attention_analysis')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存注意力权重
    attention_json = os.path.join(args.output_dir, 'attention_weights.json')
    with open(attention_json, 'w', encoding='utf-8') as f:
        json.dump(attention_results, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved attention weights to: {attention_json}")
    
    # 保存分析结果
    analysis_json = os.path.join(args.output_dir, 'abnormal_connections_analysis.json')
    with open(analysis_json, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"Saved analysis to: {analysis_json}")
    
    print(f"\n{'='*60}")
    print(f"Analysis complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()