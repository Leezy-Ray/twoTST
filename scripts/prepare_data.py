"""
数据准备脚本
功能:
1. 加载 fmri.npy
2. 数据清洗（去除ROI全零样本）
3. 可选滑动窗口切分
4. 计算PCC上三角向量
5. 保存处理后的数据
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import pickle


def load_fmri_data(data_path):
    """
    加载fMRI数据
    
    Args:
        data_path: fmri.npy文件路径
    
    Returns:
        timeseries: ndarray, shape (n_subjects, n_rois, n_timepoints)
        labels: ndarray, shape (n_subjects,)
    """
    fmri_data = np.load(data_path, allow_pickle=True)
    
    # 如果是0维数组，提取实际对象
    if fmri_data.ndim == 0:
        fmri_data = fmri_data.item()
    
    timeseries = fmri_data['timeseries']  # (n_subjects, n_rois, n_timepoints)
    labels = fmri_data['label']
    
    return timeseries, np.array(labels).astype(int)


def clean_data(timeseries, labels):
    """
    数据清洗：去除ROI全零的样本
    
    Args:
        timeseries: ndarray, shape (n_subjects, n_rois, n_timepoints) = (1009, 200, 100)
        labels: ndarray, shape (n_subjects,)
    
    Returns:
        cleaned_timeseries: ndarray, shape (n_clean, n_rois, n_timepoints)
        cleaned_labels: ndarray
    """
    to_remove = []
    n_subjects = len(timeseries)
    
    for i in range(n_subjects):
        subject_data = timeseries[i]  # (n_rois, n_timepoints) = (200, 100)
        
        # 检查是否有全零的ROI列
        roi_all_zeros = np.all(subject_data == 0, axis=1)  # 按时间维度检查
        if np.any(roi_all_zeros):
            to_remove.append(i)
            print(f'Subject {i} removed: ROI(s) with all zeros')
            continue
        
        # 检查是否有全零的时间点行
        time_all_zeros = np.all(subject_data == 0, axis=0)  # 按ROI维度检查
        if np.any(time_all_zeros):
            to_remove.append(i)
            print(f'Subject {i} removed: timepoint(s) with all zeros')
    
    # 删除异常样本
    if to_remove:
        keep_idx = [i for i in range(n_subjects) if i not in to_remove]
        cleaned_timeseries = np.array([timeseries[i] for i in keep_idx])
        cleaned_labels = np.array([labels[i] for i in keep_idx])
        print(f'Removed {len(to_remove)} subjects, {len(cleaned_labels)} remaining')
    else:
        cleaned_timeseries = np.array(timeseries)
        cleaned_labels = np.array(labels)
        print(f'No subjects removed, {len(cleaned_labels)} subjects')
    
    print(f'Cleaned data shape: {cleaned_timeseries.shape}')
    return cleaned_timeseries, cleaned_labels


def apply_sliding_window(timeseries, window_size, stride):
    """
    应用滑动窗口切分
    
    Args:
        timeseries: ndarray, shape (n_subjects, n_rois, n_timepoints)
        window_size: 窗口大小
        stride: 滑动步长
    
    Returns:
        windowed_data: list of ndarray, each (n_windows, n_rois, window_size)
        window_indices: list of (subject_idx, window_idx) tuples
    """
    windowed_data = []
    window_indices = []
    
    for subj_idx, subject_data in enumerate(tqdm(timeseries, desc='Applying sliding window')):
        n_rois, n_timepoints = subject_data.shape
        n_windows = (n_timepoints - window_size) // stride + 1
        
        subject_windows = []
        for win_idx in range(n_windows):
            start = win_idx * stride
            end = start + window_size
            window = subject_data[:, start:end]  # (n_rois, window_size)
            subject_windows.append(window)
            window_indices.append((subj_idx, win_idx))
        
        windowed_data.append(np.array(subject_windows))
    
    return windowed_data, window_indices


def compute_pcc(timeseries):
    """
    计算Pearson相关系数矩阵并提取上三角向量
    
    Args:
        timeseries: ndarray, shape (n_rois, n_timepoints) = (200, 100)
                   每行是一个ROI的时间序列
    
    Returns:
        pcc_upper: ndarray, shape (n_rois * (n_rois - 1) // 2,) = (19900,)
    """
    # 输入应该是 (n_rois, n_timepoints) = (200, 100)
    # np.corrcoef 按行计算相关系数，所以每行应该是一个变量的观测值
    # 即每行是一个ROI的时间序列
    
    # 计算相关系数矩阵: corrcoef 按行计算，输出 (n_rois, n_rois)
    pcc_matrix = np.corrcoef(timeseries)  # (200, 200)
    
    # 处理NaN值（当某个ROI方差为0时会出现）
    pcc_matrix = np.nan_to_num(pcc_matrix, nan=0.0)
    
    # 提取上三角（不包含对角线）
    upper_indices = np.triu_indices(pcc_matrix.shape[0], k=1)
    pcc_upper = pcc_matrix[upper_indices]  # 200*199/2 = 19900
    
    return pcc_upper


def process_data(timeseries, labels, use_sliding_window=False, 
                 window_size=50, stride=25):
    """
    处理数据：计算PCC并组织数据
    
    Args:
        timeseries: ndarray, shape (n_subjects, n_rois, n_timepoints) = (N, 200, 100)
        labels: ndarray, shape (n_subjects,)
        use_sliding_window: 是否使用滑动窗口
        window_size: 窗口大小
        stride: 滑动步长
    
    Returns:
        processed_data: dict containing all processed data
    """
    n_subjects = timeseries.shape[0]
    n_rois = timeseries.shape[1]  # 200
    n_timepoints = timeseries.shape[2]  # 100
    pcc_dim = n_rois * (n_rois - 1) // 2  # 19900
    
    print(f"Processing {n_subjects} subjects, {n_rois} ROIs, {n_timepoints} timepoints")
    print(f"PCC dimension: {pcc_dim}")
    
    if use_sliding_window:
        # 使用滑动窗口 - 预先计算总样本数
        n_windows_per_subject = (n_timepoints - window_size) // stride + 1
        total_samples = n_subjects * n_windows_per_subject
        
        # 预分配数组
        all_timeseries = np.zeros((total_samples, window_size, n_rois), dtype=np.float32)
        all_pcc = np.zeros((total_samples, pcc_dim), dtype=np.float32)
        all_labels = np.zeros(total_samples, dtype=np.int32)
        subject_ids = []
        
        idx = 0
        for subj_idx in tqdm(range(n_subjects), desc='Processing subjects'):
            subject_data = timeseries[subj_idx]  # (n_rois, n_timepoints)
            label = labels[subj_idx]
            
            for win_idx in range(n_windows_per_subject):
                start = win_idx * stride
                end = start + window_size
                window = subject_data[:, start:end]  # (n_rois, window_size)
                
                all_timeseries[idx] = window.T  # (window_size, n_rois)
                all_pcc[idx] = compute_pcc(window)
                all_labels[idx] = label
                subject_ids.append(f's{subj_idx}_w{win_idx}')
                idx += 1
        
        processed_data = {
            'timeseries': all_timeseries[:idx],
            'pcc_vectors': all_pcc[:idx],
            'labels': all_labels[:idx],
            'subject_ids': subject_ids,
            'window_info': {
                'window_size': window_size,
                'stride': stride,
                'use_sliding_window': True
            }
        }
    else:
        # 不使用滑动窗口 - 预分配数组
        all_timeseries = np.zeros((n_subjects, n_timepoints, n_rois), dtype=np.float32)
        all_pcc = np.zeros((n_subjects, pcc_dim), dtype=np.float32)
        subject_ids = []
        
        for subj_idx in tqdm(range(n_subjects), desc='Processing subjects'):
            subject_data = timeseries[subj_idx]  # (n_rois, n_timepoints)
            all_timeseries[subj_idx] = subject_data.T  # (n_timepoints, n_rois)
            all_pcc[subj_idx] = compute_pcc(subject_data)
            subject_ids.append(f's{subj_idx}')
        
        processed_data = {
            'timeseries': all_timeseries,
            'pcc_vectors': all_pcc,
            'labels': labels.copy(),
            'subject_ids': subject_ids,
            'window_info': {
                'use_sliding_window': False
            }
        }
    
    return processed_data


def main(args):
    print('=' * 60)
    print('TwoTST Data Preparation')
    print('=' * 60)
    
    # 加载数据
    print('\n[1/4] Loading data...')
    timeseries, labels = load_fmri_data(args.data_path)
    print(f'Loaded data: {timeseries.shape}')
    print(f'Labels distribution: {np.bincount(labels)}')
    
    # 数据清洗
    print('\n[2/4] Cleaning data...')
    timeseries, labels = clean_data(timeseries, labels)
    
    # 处理数据
    print('\n[3/4] Processing data...')
    processed_data = process_data(
        timeseries, labels,
        use_sliding_window=args.use_sliding_window,
        window_size=args.window_size,
        stride=args.stride
    )
    
    # 保存处理后的数据
    print('\n[4/4] Saving processed data...')
    os.makedirs(args.output_dir, exist_ok=True)
    
    output_path = os.path.join(args.output_dir, 'processed_data.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(processed_data, f, protocol=4)
    
    # 打印数据统计
    print('\n' + '=' * 60)
    print('Data Statistics:')
    print('=' * 60)
    print(f"Timeseries shape: {processed_data['timeseries'].shape}")
    print(f"PCC vectors shape: {processed_data['pcc_vectors'].shape}")
    print(f"Labels shape: {processed_data['labels'].shape}")
    print(f"Labels distribution: {np.bincount(processed_data['labels'])}")
    print(f"Saved to: {output_path}")
    print('=' * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TwoTST Data Preparation')
    parser.add_argument('--data_path', type=str, 
                        default='/root/workplace/exp/TwoTST/data/fmri.npy',
                        help='Path to fmri.npy')
    parser.add_argument('--output_dir', type=str,
                        default='/root/workplace/exp/TwoTST/data/processed',
                        help='Output directory for processed data')
    parser.add_argument('--use_sliding_window', action='store_true',
                        help='Use sliding window for data augmentation')
    parser.add_argument('--window_size', type=int, default=50,
                        help='Sliding window size')
    parser.add_argument('--stride', type=int, default=25,
                        help='Sliding window stride')
    
    args = parser.parse_args()
    main(args)
