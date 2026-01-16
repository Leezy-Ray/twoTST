"""
评估指标模块
提供分类任务的各种评估指标
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
