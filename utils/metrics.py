"""
评估指标模块
提供分类任务的各种评估指标、Bootstrap 置信区间与复现信息
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)


def get_reproducibility_info():
    """收集复现所需的环境与配置信息"""
    info = {
        'numpy': np.__version__,
    }
    try:
        import torch
        info['pytorch'] = torch.__version__
        info['cuda_available'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['cuda_version'] = torch.version.cuda or 'N/A'
            info['cudnn_version'] = torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'
            info['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else 'N/A'
    except ImportError:
        info['pytorch'] = 'not installed'
    try:
        import sklearn
        info['sklearn'] = sklearn.__version__
    except ImportError:
        info['sklearn'] = 'not installed'
    return info


def bootstrap_confidence_interval(values, n_bootstrap=1000, ci=0.95, seed=None):
    """
    计算指标的 Bootstrap 置信区间

    Args:
        values: 每折/每次运行的指标值列表
        n_bootstrap: Bootstrap 采样次数
        ci: 置信水平 (0-1)
        seed: 随机种子

    Returns:
        (mean, std, lower, upper)
    """
    values = np.asarray(values)
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        boot_means.append(np.mean(values[idx]))
    boot_means = np.array(boot_means)
    alpha = 1 - ci
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return float(np.mean(values)), float(np.std(values)), float(lower), float(upper)


def compute_metrics(y_true, y_pred, y_prob=None):
    """
    计算分类评估指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        y_prob: 预测概率（用于计算AUC）
    
    Returns:
        metrics: 包含各种指标的字典
    """
    metrics = {}
    
    # 基本指标
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # AUC指标（需要概率）
    if y_prob is not None:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
            metrics['ap'] = average_precision_score(y_true, y_prob)
        except ValueError:
            metrics['auc'] = 0.0
            metrics['ap'] = 0.0
    
    # 混淆矩阵相关指标
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0  # TPR
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0  # TNR
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0  # 假阳性率
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0  # 假阴性率
        
        # 添加混淆矩阵
        metrics['confusion_matrix'] = cm
    
    return metrics


def print_metrics(metrics, prefix=''):
    """
    打印评估指标
    
    Args:
        metrics: 指标字典
        prefix: 前缀字符串
    """
    print(f"{prefix}Accuracy:    {metrics.get('accuracy', 0):.4f}")
    print(f"{prefix}Precision:   {metrics.get('precision', 0):.4f}")
    print(f"{prefix}Recall:      {metrics.get('recall', 0):.4f}")
    print(f"{prefix}F1 Score:    {metrics.get('f1', 0):.4f}")
    
    if 'auc' in metrics:
        print(f"{prefix}AUC:         {metrics['auc']:.4f}")
    if 'ap' in metrics:
        print(f"{prefix}AP:          {metrics['ap']:.4f}")
    if 'sensitivity' in metrics:
        print(f"{prefix}Sensitivity: {metrics['sensitivity']:.4f}")
    if 'specificity' in metrics:
        print(f"{prefix}Specificity: {metrics['specificity']:.4f}")


def aggregate_window_predictions_to_subject_level(
    y_true_window, y_pred_window, y_prob_window,
    test_sample_indices, subject_indices,
    strategy='prob_mean',
):
    """
    将窗口级预测汇总为受试者级预测（用于滑窗增强场景的临床相关评估）。

    Args:
        y_true_window: 窗口级真实标签 (n_windows,)
        y_pred_window: 窗口级预测 (n_windows,)
        y_prob_window: 窗口级正类概率 (n_windows,)
        test_sample_indices: 测试集样本索引，与 y_* 一一对应
        subject_indices: 全量数据的受试者索引 (n_samples,)
        strategy: 'prob_mean' 概率均值 | 'majority_vote' 多数投票

    Returns:
        y_true_subj: 受试者级真实标签
        y_pred_subj: 受试者级预测
        y_prob_subj: 受试者级概率（prob_mean 时为均值，majority_vote 时为窗口预测均值）
    """
    test_subjects = subject_indices[test_sample_indices]
    unique_subjects = np.unique(test_subjects)

    y_true_subj = []
    y_pred_subj = []
    y_prob_subj = []

    for s in unique_subjects:
        mask = test_subjects == s
        probs = np.array(y_prob_window)[mask]
        preds = np.array(y_pred_window)[mask]
        labels = np.array(y_true_window)[mask]
        true_label = int(labels[0])  # 同一受试者标签相同

        if strategy == 'prob_mean':
            prob_subj = float(np.mean(probs))
            pred_subj = 1 if prob_subj >= 0.5 else 0
        else:  # majority_vote
            pred_subj = int(np.round(np.mean(preds)))  # 等价于多数投票
            prob_subj = float(np.mean(probs))

        y_true_subj.append(true_label)
        y_pred_subj.append(pred_subj)
        y_prob_subj.append(prob_subj)

    return (
        np.array(y_true_subj),
        np.array(y_pred_subj),
        np.array(y_prob_subj),
    )


def aggregate_cv_metrics(all_metrics):
    """
    聚合交叉验证结果
    
    Args:
        all_metrics: 每折的指标列表
    
    Returns:
        mean_metrics: 平均指标
        std_metrics: 标准差
    """
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc', 
                    'sensitivity', 'specificity']
    
    mean_metrics = {}
    std_metrics = {}
    
    for name in metric_names:
        values = [m.get(name, 0) for m in all_metrics]
        mean_metrics[name] = np.mean(values)
        std_metrics[name] = np.std(values)
    
    return mean_metrics, std_metrics


def print_cv_results(mean_metrics, std_metrics):
    """
    打印交叉验证结果
    
    Args:
        mean_metrics: 平均指标
        std_metrics: 标准差
    """
    print("\nCross-Validation Results:")
    print("-" * 40)
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'auc',
                    'sensitivity', 'specificity']
    
    for name in metric_names:
        if name in mean_metrics:
            mean_val = mean_metrics[name]
            std_val = std_metrics[name]
            print(f"{name.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")


class MetricTracker:
    """
    指标追踪器
    用于训练过程中追踪和记录指标
    """
    
    def __init__(self, metric_names=None):
        """
        Args:
            metric_names: 要追踪的指标名称列表
        """
        self.metric_names = metric_names or ['loss', 'accuracy']
        self.history = {name: [] for name in self.metric_names}
        self.best = {name: float('inf') if 'loss' in name else 0 
                     for name in self.metric_names}
    
    def update(self, metrics):
        """
        更新指标
        
        Args:
            metrics: 指标字典
        """
        for name, value in metrics.items():
            if name in self.history:
                self.history[name].append(value)
                
                # 更新最佳值
                if 'loss' in name:
                    if value < self.best[name]:
                        self.best[name] = value
                else:
                    if value > self.best[name]:
                        self.best[name] = value
    
    def get_history(self, name):
        """获取指标历史"""
        return self.history.get(name, [])
    
    def get_best(self, name):
        """获取最佳值"""
        return self.best.get(name, None)
    
    def is_best(self, name, value):
        """检查是否是最佳值"""
        if 'loss' in name:
            return value < self.best.get(name, float('inf'))
        else:
            return value > self.best.get(name, 0)


if __name__ == '__main__':
    # 测试
    y_true = np.array([0, 0, 1, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3, 0.4, 0.7, 0.1])
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print_metrics(metrics)
    
    print("\nConfusion Matrix:")
    print(metrics['confusion_matrix'])
