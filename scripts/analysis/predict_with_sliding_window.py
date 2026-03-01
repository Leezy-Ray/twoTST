"""
滑动窗口预测脚本
加载训练好的模型，对提取的10个样本进行滑动窗口预测，输出各窗口的ASD概率
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

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from models.dual_stream import DualStreamModel
from utils.data_loader import TwoTSTDataset


def apply_sliding_window(timeseries, pcc_vectors, labels, subject_ids, 
                        window_size=32, stride=16):
    """对提取的样本应用滑动窗口"""
    n_subjects = len(timeseries)
    n_timepoints = timeseries.shape[1]
    
    n_windows_per_subject = (n_timepoints - window_size) // stride + 1
    
    print(f"Applying sliding window: window_size={window_size}, stride={stride}")
    print(f"Windows per subject: {n_windows_per_subject}")
    
    all_timeseries = []
    all_pcc_vectors = []
    all_labels = []
    all_subject_ids = []
    window_indices = []
    
    for subj_idx in range(n_subjects):
        subject_data = timeseries[subj_idx]
        
        for win_idx in range(n_windows_per_subject):
            start = win_idx * stride
            end = start + window_size
            window = subject_data[start:end]
            
            # 从窗口重新计算PCC
            window_t = window.T
            pcc_matrix = np.corrcoef(window_t)
            pcc_matrix = np.nan_to_num(pcc_matrix, nan=0.0)
            upper_indices = np.triu_indices(pcc_matrix.shape[0], k=1)
            window_pcc = pcc_matrix[upper_indices]
            
            all_timeseries.append(window)
            all_pcc_vectors.append(window_pcc)
            all_labels.append(labels[subj_idx])
            all_subject_ids.append(f"{subject_ids[subj_idx]}_w{win_idx}")
            window_indices.append((subj_idx, win_idx))
    
    windowed_data = {
        'timeseries': np.array(all_timeseries, dtype=np.float32),
        'pcc_vectors': np.array(all_pcc_vectors, dtype=np.float32),
        'labels': np.array(all_labels, dtype=np.int64),
        'subject_ids': all_subject_ids,
        'window_indices': window_indices,
        'window_info': {'window_size': window_size, 'stride': stride}
    }
    
    print(f"Total windows: {len(all_labels)}")
    return windowed_data


def load_model(checkpoint_path, results_json_path, device='cuda'):
    """从checkpoint和results.json加载模型"""
    print(f"\nLoading model from {checkpoint_path}...")
    
    with open(results_json_path, 'r') as f:
        results = json.load(f)
    
    config = results.get('config', {})
    model_config = config.get('model', {})
    fusion_config = config.get('fusion', {})
    
    tst1_config = model_config.get('tst1', {})
    tst2_config = model_config.get('tst2', {})
    fusion_type = fusion_config.get('type', 'cross_attention')
    fusion_kwargs = fusion_config.get(fusion_type, {})
    
    model = DualStreamModel(
        tst1_config=tst1_config,
        tst2_config=tst2_config,
        fusion_type=fusion_type,
        fusion_config=fusion_kwargs,
        num_classes=2,
        dropout=config.get('finetune', {}).get('classifier', {}).get('dropout', 0.3)
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    print("Model loaded successfully!")
    return model, config


def predict(model, windowed_data, device='cuda', batch_size=32, subject_agg_strategy='prob_mean'):
    """对窗口化数据进行预测"""
    print(f"\nPredicting with batch_size={batch_size}...")
    
    timeseries = windowed_data['timeseries']
    pcc_vectors = windowed_data['pcc_vectors']
    labels = windowed_data['labels']
    subject_ids = windowed_data['subject_ids']
    window_indices = windowed_data['window_indices']
    
    dataset = TwoTSTDataset(timeseries, pcc_vectors, labels, 
                           normalize_ts=True, normalize_pcc=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    
    all_probs = []
    all_preds = []
    all_logits = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Predicting'):
            ts = batch['timeseries'].to(device)
            pcc = batch['pcc_vector'].to(device)
            
            logits = model(ts, pcc)
            probs = F.softmax(logits, dim=1)
            
            all_logits.extend(logits.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
    
    predictions = {
        'subject_ids': subject_ids,
        'window_indices': window_indices,
        'labels': labels.tolist(),
        'asd_probs': [float(p) for p in all_probs],
        'predictions': [int(p) for p in all_preds],
        'logits': [l.tolist() for l in all_logits]
    }
    
    results_by_subject = {}
    for i, (subj_id, (subj_idx, win_idx)) in enumerate(zip(subject_ids, window_indices)):
        original_subj_id = subj_id.split('_w')[0]
        
        if original_subj_id not in results_by_subject:
            results_by_subject[original_subj_id] = {
                'subject_id': original_subj_id,
                'label': int(labels[i]),
                'windows': []
            }
        
        results_by_subject[original_subj_id]['windows'].append({
            'window_index': int(win_idx),
            'asd_prob': float(all_probs[i]),
            'prediction': int(all_preds[i])
        })
    
    # 受试者级汇总：支持 prob_mean（概率均值）或 majority_vote
    for subj_id in results_by_subject:
        probs = [w['asd_prob'] for w in results_by_subject[subj_id]['windows']]
        preds = [w['prediction'] for w in results_by_subject[subj_id]['windows']]
        results_by_subject[subj_id]['mean_asd_prob'] = float(np.mean(probs))
        results_by_subject[subj_id]['std_asd_prob'] = float(np.std(probs))
        results_by_subject[subj_id]['n_windows'] = len(probs)
        if subject_agg_strategy == 'majority_vote':
            results_by_subject[subj_id]['subject_prediction'] = int(np.round(np.mean(preds)))
        else:
            results_by_subject[subj_id]['subject_prediction'] = 1 if results_by_subject[subj_id]['mean_asd_prob'] >= 0.5 else 0
    
    predictions['results_by_subject'] = results_by_subject
    
    print(f"\nPrediction summary:")
    for subj_id, result in results_by_subject.items():
        print(f"  {subj_id} (label={result['label']}): "
              f"mean_prob={result['mean_asd_prob']:.4f}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(description='Predict with sliding window')
    parser.add_argument('--extracted_data', type=str,
                        default='/root/workplace/exp/TwoTST/data/extracted/extracted_subjects.pkl')
    parser.add_argument('--checkpoint', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints_sw/finetune/sw_baseline_cross_attention/best_model.pt')
    parser.add_argument('--results_json', type=str,
                        default='/root/workplace/exp/TwoTST/checkpoints_sw/finetune/sw_baseline_cross_attention/results.json')
    parser.add_argument('--window_size', type=int, default=32)
    parser.add_argument('--stride', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subject_agg_strategy', type=str, default='prob_mean',
                        choices=['prob_mean', 'majority_vote'],
                        help='Subject-level aggregation: prob_mean or majority_vote')

    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    with open(args.extracted_data, 'rb') as f:
        extracted_data = pickle.load(f)
    
    timeseries = extracted_data['timeseries']
    pcc_vectors = extracted_data['pcc_vectors']
    labels = extracted_data['labels']
    subject_ids = extracted_data['subject_ids']
    
    windowed_data = apply_sliding_window(
        timeseries, pcc_vectors, labels, subject_ids,
        window_size=args.window_size, stride=args.stride
    )
    
    model, config = load_model(args.checkpoint, args.results_json, device)
    predictions = predict(model, windowed_data, device, args.batch_size,
                         subject_agg_strategy=args.subject_agg_strategy)
    
    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'data', 'extracted', 'predictions')
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_json = os.path.join(args.output_dir, 'predictions_sliding_window.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
    
    print(f"\nPredictions saved to: {output_json}")


if __name__ == '__main__':
    main()