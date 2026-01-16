"""
滑动窗口增强完整实验流程
1. 使用滑动窗口准备数据
2. 预训练TST1和TST2
3. 运行所有实验组
4. 收集结果并生成报告
"""

import os
import sys
import subprocess
import json
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


def run_command(cmd, desc=""):
    """运行命令并显示输出"""
    print(f"\n{'='*60}")
    print(f"[{desc}]")
    print(f"Command: {cmd}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False)
    return result.returncode


def prepare_sliding_window_data(window_size=32, stride=16):
    """准备滑动窗口数据"""
    print("\n" + "="*80)
    print("STEP 1: 准备滑动窗口增强数据")
    print(f"Window Size: {window_size}, Stride: {stride}")
    print("="*80)
    
    # 创建输出目录
    output_dir = os.path.join(PROJECT_ROOT, 'data', 'processed_sw')
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = f"""cd {PROJECT_ROOT} && python scripts/prepare_data.py \
        --data_path {PROJECT_ROOT}/data/fmri.npy \
        --output_dir {output_dir} \
        --use_sliding_window \
        --window_size {window_size} \
        --stride {stride}"""
    
    return run_command(cmd, "数据准备")


def create_sw_experiment_configs(window_size=32, stride=16):
    """创建滑动窗口实验配置"""
    print("\n" + "="*80)
    print("STEP 2: 创建滑动窗口实验配置文件")
    print("="*80)
    
    sw_config_dir = os.path.join(PROJECT_ROOT, 'configs', 'experiments_sw')
    os.makedirs(sw_config_dir, exist_ok=True)
    
    # 基础配置模板
    base_config = {
        'experiment': {
            'name': '',
            'description': '',
        },
        'data': {
            'data_path': f'{PROJECT_ROOT}/data/processed_sw/processed_data.pkl',
            'split_ratio': {'train': 0.7, 'val': 0.1, 'test': 0.2},
            'sliding_window': {
                'enabled': True,
                'window_size': window_size,
                'stride': stride
            }
        },
        'model': {
            'tst1': {
                'emb_dim': 512,
                'n_heads': 8,
                'n_layers': 6,
                'dim_feedforward': 2048,
                'dropout': 0.1,
                'n_rois': 200,
                'max_seq_len': window_size  # 使用窗口大小
            },
            'tst2': {
                'd_model': 256,
                'n_heads': 8,
                'n_layers': 2,
                'dim_feedforward': 512,
                'dropout': 0.1,
                'pcc_dim': 19900
            }
        },
        'pretrain': {
            'use_pretrained': True,
            'tst1': {
                'checkpoint': f'{PROJECT_ROOT}/checkpoints_sw/tst1/tst1_best.pt'
            },
            'tst2': {
                'checkpoint': f'{PROJECT_ROOT}/checkpoints_sw/tst2/tst2_best.pt'
            }
        },
        'contrastive': {
            'enabled': False,
            'freeze_tst1': False,
            'freeze_tst2': False,
            'epochs': 50,
            'batch_size': 64,
            'lr': 1e-4,
            'weight_decay': 1e-4,
            'temperature': 0.07,
            'proj_hidden_dim': 256,
            'proj_output_dim': 128,
            'loss_type': 'infonce'
        },
        'fusion': {
            'type': 'concat',
            'use_tst1': True,
            'use_tst2': True,
            'concat': {'hidden_dim': 512},
            'gated': {'hidden_dim': 512},
            'cross_attention': {'n_heads': 8, 'dropout': 0.1},
            'bilinear': {'output_dim': 256},
            'attention_pooling': {'hidden_dim': 256}
        },
        'finetune': {
            'epochs': 100,
            'batch_size': 64,
            'lr': 5e-5,
            'weight_decay': 1e-4,
            'patience': 20,
            'freeze_tst1': False,
            'freeze_tst2': False,
            'save_dir': '',
            'classifier': {
                'hidden_dims': [256, 64],
                'dropout': 0.3
            }
        },
        'training': {
            'device': 'cuda',
            'seed': 42,
            'num_workers': 4
        },
        'logging': {
            'log_dir': '',
            'tensorboard': True
        }
    }
    
    # 实验配置列表
    experiments = [
        # Group 1: Baseline融合实验
        {
            'name': 'sw_baseline_concat',
            'description': '滑动窗口+Concat融合',
            'fusion': {'type': 'concat'}
        },
        {
            'name': 'sw_baseline_gated',
            'description': '滑动窗口+Gated融合',
            'fusion': {'type': 'gated'}
        },
        {
            'name': 'sw_baseline_cross_attention',
            'description': '滑动窗口+Cross-Attention融合',
            'fusion': {'type': 'cross_attention'}
        },
        {
            'name': 'sw_baseline_bilinear',
            'description': '滑动窗口+Bilinear融合',
            'fusion': {'type': 'bilinear'}
        },
        {
            'name': 'sw_baseline_attention_pooling',
            'description': '滑动窗口+Attention Pooling融合',
            'fusion': {'type': 'attention_pooling'}
        },
        
        # Group 2: 对比学习实验
        {
            'name': 'sw_contrastive_both_train',
            'description': '滑动窗口+对比学习(双流训练)',
            'contrastive': {'enabled': True, 'freeze_tst1': False, 'freeze_tst2': False}
        },
        {
            'name': 'sw_contrastive_freeze_tst1',
            'description': '滑动窗口+对比学习(冻结TST1)',
            'contrastive': {'enabled': True, 'freeze_tst1': True, 'freeze_tst2': False}
        },
        {
            'name': 'sw_contrastive_freeze_tst2',
            'description': '滑动窗口+对比学习(冻结TST2)',
            'contrastive': {'enabled': True, 'freeze_tst1': False, 'freeze_tst2': True}
        },
        {
            'name': 'sw_contrastive_freeze_both',
            'description': '滑动窗口+对比学习(双流冻结)',
            'contrastive': {'enabled': True, 'freeze_tst1': True, 'freeze_tst2': True}
        },
        
        # Group 3: 微调冻结实验
        {
            'name': 'sw_finetune_freeze_tst1',
            'description': '滑动窗口+微调冻结TST1',
            'finetune': {'freeze_tst1': True, 'freeze_tst2': False}
        },
        {
            'name': 'sw_finetune_freeze_tst2',
            'description': '滑动窗口+微调冻结TST2',
            'finetune': {'freeze_tst1': False, 'freeze_tst2': True}
        },
        {
            'name': 'sw_finetune_freeze_both',
            'description': '滑动窗口+微调双冻结',
            'finetune': {'freeze_tst1': True, 'freeze_tst2': True}
        },
        
        # Group 4: 消融实验
        {
            'name': 'sw_single_tst1',
            'description': '滑动窗口+仅TST1',
            'fusion': {'use_tst1': True, 'use_tst2': False}
        },
        {
            'name': 'sw_single_tst2',
            'description': '滑动窗口+仅TST2',
            'fusion': {'use_tst1': False, 'use_tst2': True}
        },
        
        # Group 5: 组合实验
        {
            'name': 'sw_contrastive_finetune_freeze',
            'description': '滑动窗口+对比学习+微调双冻结',
            'contrastive': {'enabled': True, 'freeze_tst1': True, 'freeze_tst2': True},
            'finetune': {'freeze_tst1': True, 'freeze_tst2': True}
        },
    ]
    
    config_files = []
    for exp in experiments:
        config = yaml.safe_load(yaml.dump(base_config))  # deep copy
        
        # 更新实验名称和描述
        config['experiment']['name'] = exp['name']
        config['experiment']['description'] = exp['description']
        
        # 更新特定配置
        if 'fusion' in exp:
            config['fusion'].update(exp['fusion'])
        if 'contrastive' in exp:
            config['contrastive'].update(exp['contrastive'])
        if 'finetune' in exp:
            config['finetune'].update(exp['finetune'])
        
        # 设置保存目录
        config['finetune']['save_dir'] = f"{PROJECT_ROOT}/checkpoints_sw/finetune/{exp['name']}"
        config['logging']['log_dir'] = f"{PROJECT_ROOT}/logs_sw/{exp['name']}"
        
        # 保存配置
        config_path = os.path.join(sw_config_dir, f"{exp['name']}.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        config_files.append(config_path)
        print(f"Created: {config_path}")
    
    return config_files


def pretrain_models(window_size=32):
    """预训练TST1和TST2"""
    print("\n" + "="*80)
    print("STEP 3: 预训练TST1和TST2模型")
    print("="*80)
    
    data_path = os.path.join(PROJECT_ROOT, 'data', 'processed_sw', 'processed_data.pkl')
    save_dir = os.path.join(PROJECT_ROOT, 'checkpoints_sw')
    log_dir = os.path.join(PROJECT_ROOT, 'logs_sw', 'pretrain')
    
    cmd = f"""cd {PROJECT_ROOT} && python scripts/train_pretrain.py \
        --data_path {data_path} \
        --save_dir {save_dir} \
        --log_dir {log_dir} \
        --tst1_epochs 100 \
        --tst2_epochs 100 \
        --batch_size 64 \
        --lr 1e-4"""
    
    return run_command(cmd, "预训练TST1和TST2")


def run_all_experiments(config_files):
    """运行所有实验"""
    print("\n" + "="*80)
    print("STEP 4: 运行所有实验")
    print("="*80)
    
    results = []
    for i, config_path in enumerate(config_files, 1):
        exp_name = Path(config_path).stem
        print(f"\n[{i}/{len(config_files)}] Running: {exp_name}")
        
        cmd = f"cd {PROJECT_ROOT} && python scripts/run_experiment.py --config {config_path}"
        ret = run_command(cmd, f"实验 {exp_name}")
        
        # 读取结果
        save_dir = os.path.join(PROJECT_ROOT, 'checkpoints_sw', 'finetune', exp_name)
        results_file = os.path.join(save_dir, 'results.json')
        
        if os.path.exists(results_file):
            with open(results_file) as f:
                result = json.load(f)
                results.append(result)
        
    return results


def generate_report(results, window_size=32, stride=16):
    """生成实验报告"""
    print("\n" + "="*80)
    print("STEP 5: 生成实验报告")
    print("="*80)
    
    results_dir = os.path.join(PROJECT_ROOT, 'results_sw')
    os.makedirs(results_dir, exist_ok=True)
    
    # 计算统计数据
    aucs = [r['auc'] for r in results]
    accs = [r['accuracy'] for r in results]
    best_epochs = [r.get('best_epoch', 0) for r in results]
    
    stats = {
        'total_experiments': len(results),
        'window_size': window_size,
        'stride': stride,
        'auc': {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs)),
            'max': float(np.max(aucs)),
            'min': float(np.min(aucs)),
            'best_exp': results[np.argmax(aucs)]['experiment'],
            'best_exp_epoch': results[np.argmax(aucs)].get('best_epoch', 0)
        },
        'accuracy': {
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)),
            'max': float(np.max(accs)),
            'min': float(np.min(accs)),
            'best_exp': results[np.argmax(accs)]['experiment'],
            'best_exp_epoch': results[np.argmax(accs)].get('best_epoch', 0)
        },
        'best_epoch': {
            'mean': float(np.mean(best_epochs)) if best_epochs else 0,
            'std': float(np.std(best_epochs)) if best_epochs else 0,
            'min': int(np.min(best_epochs)) if best_epochs else 0,
            'max': int(np.max(best_epochs)) if best_epochs else 0
        }
    }
    
    # 保存JSON结果
    with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    with open(os.path.join(results_dir, 'statistics.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    # 生成Markdown报告
    report = generate_markdown_report(results, stats, window_size, stride)
    
    report_path = os.path.join(results_dir, 'SlidingWindowReport.md')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n报告已保存到: {report_path}")
    return stats


def generate_markdown_report(results, stats, window_size, stride):
    """生成Markdown格式报告"""
    
    # 按AUC排序
    sorted_results = sorted(results, key=lambda x: x['auc'], reverse=True)
    
    # 获取best_epoch统计
    best_epoch_mean = stats['best_epoch']['mean'] if 'best_epoch' in stats else 0
    best_epoch_std = stats['best_epoch']['std'] if 'best_epoch' in stats else 0
    
    report = f"""# TwoTST 滑动窗口增强实验报告

## Sliding Window Data Augmentation Experiment Report

---

## 1. 实验配置

### 1.1 滑动窗口参数

| 参数 | 值 |
|------|-----|
| 窗口大小 (Window Size) | {window_size} |
| 滑动步长 (Stride) | {stride} |
| 原始时间点数 | 100 |
| 每个样本窗口数 | {(100 - window_size) // stride + 1} |

### 1.2 数据增强效果

原始数据经过滑动窗口处理后，数据量显著增加：
- 原始样本数：963
- 增强后样本数：963 × {(100 - window_size) // stride + 1} = {963 * ((100 - window_size) // stride + 1)}

---

## 2. 实验结果汇总

### 2.1 统计指标

| 指标 | 平均值 | 标准差 | 最高值 | 最低值 |
|------|--------|--------|--------|--------|
| **AUC** | {stats['auc']['mean']:.4f} | {stats['auc']['std']:.4f} | {stats['auc']['max']:.4f} | {stats['auc']['min']:.4f} |
| **Accuracy** | {stats['accuracy']['mean']:.4f} | {stats['accuracy']['std']:.4f} | {stats['accuracy']['max']:.4f} | {stats['accuracy']['min']:.4f} |
| **Best Epoch** | {best_epoch_mean:.1f} | {best_epoch_std:.1f} | {stats['best_epoch']['max']} | {stats['best_epoch']['min']} |

### 2.2 最佳配置

| 指标 | 最佳实验 | 数值 | 最佳Epoch |
|------|----------|------|-----------|
| 最高AUC | {stats['auc']['best_exp']} | {stats['auc']['max']:.4f} | {stats['auc']['best_exp_epoch']} |
| 最高ACC | {stats['accuracy']['best_exp']} | {stats['accuracy']['max']:.4f} | {stats['accuracy']['best_exp_epoch']} |

---

## 3. 完整实验结果排名（按AUC排序）

| 排名 | 实验名称 | Test AUC | Accuracy | Sensitivity | Specificity | F1 | Best Epoch |
|------|---------|----------|----------|-------------|-------------|-----|------------|
"""
    
    for i, r in enumerate(sorted_results, 1):
        medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)
        best_epoch = r.get('best_epoch', 'N/A')
        report += f"| {medal} | {r['experiment']} | {r['auc']:.4f} | {r['accuracy']:.4f} | {r['sensitivity']:.4f} | {r['specificity']:.4f} | {r['f1']:.4f} | {best_epoch} |\n"
    
    report += f"""

---

## 4. 分组分析

### 4.1 Group 1: 基线融合实验

"""
    
    baseline_results = [r for r in sorted_results if 'baseline' in r['experiment']]
    if baseline_results:
        report += "| 融合方式 | Test AUC | Accuracy |\n|---------|----------|----------|\n"
        for r in sorted(baseline_results, key=lambda x: x['auc'], reverse=True):
            fusion_type = r['experiment'].replace('sw_baseline_', '')
            report += f"| {fusion_type} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n"
    
    report += f"""

### 4.2 Group 2: 对比学习实验

"""
    
    contrastive_results = [r for r in sorted_results if 'contrastive' in r['experiment'] and 'finetune' not in r['experiment']]
    if contrastive_results:
        report += "| 策略 | Test AUC | Accuracy |\n|------|----------|----------|\n"
        for r in sorted(contrastive_results, key=lambda x: x['auc'], reverse=True):
            strategy = r['experiment'].replace('sw_contrastive_', '')
            report += f"| {strategy} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n"
    
    report += f"""

### 4.3 Group 3: 微调冻结实验

"""
    
    finetune_results = [r for r in sorted_results if 'finetune_freeze' in r['experiment'] and 'contrastive' not in r['experiment']]
    if finetune_results:
        report += "| 冻结策略 | Test AUC | Accuracy |\n|----------|----------|----------|\n"
        for r in sorted(finetune_results, key=lambda x: x['auc'], reverse=True):
            strategy = r['experiment'].replace('sw_finetune_freeze_', '')
            report += f"| {strategy} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n"
    
    report += f"""

### 4.4 Group 4: 消融实验（单流）

"""
    
    single_results = [r for r in sorted_results if 'single' in r['experiment']]
    if single_results:
        report += "| 模型 | Test AUC | Accuracy |\n|------|----------|----------|\n"
        for r in sorted(single_results, key=lambda x: x['auc'], reverse=True):
            model = r['experiment'].replace('sw_', '')
            report += f"| {model} | {r['auc']:.4f} | {r['accuracy']:.4f} |\n"
    
    report += f"""

---

## 5. 与无滑动窗口对比

待补充：对比无滑动窗口的实验结果

---

## 6. 结论

### 6.1 滑动窗口增强效果

- 数据量增加至原来的 {(100 - window_size) // stride + 1} 倍
- 平均AUC: {stats['auc']['mean']:.4f} ± {stats['auc']['std']:.4f}
- 最佳AUC: {stats['auc']['max']:.4f} ({stats['auc']['best_exp']})
- 平均收敛Epoch: {stats['best_epoch']['mean']:.1f} ± {stats['best_epoch']['std']:.1f}

### 6.2 最佳配置推荐

| 项目 | 推荐配置 |
|------|---------|
| 窗口大小 | {window_size} |
| 滑动步长 | {stride} |
| 最佳方法 | {stats['auc']['best_exp']} |
| 最佳AUC | {stats['auc']['max']:.4f} |
| 达到最佳的Epoch | {stats['auc']['best_exp_epoch']} |

### 6.3 训练收敛分析

| 统计量 | Best Epoch |
|--------|------------|
| 平均值 | {stats['best_epoch']['mean']:.1f} |
| 标准差 | {stats['best_epoch']['std']:.1f} |
| 最小值 | {stats['best_epoch']['min']} |
| 最大值 | {stats['best_epoch']['max']} |

---

**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**总实验数量**: {len(results)} 个

**最佳结果**: {stats['auc']['best_exp']} (AUC: {stats['auc']['max']:.4f}, Best Epoch: {stats['auc']['best_exp_epoch']})
"""
    
    return report


def main():
    """主函数"""
    import argparse
    parser = argparse.ArgumentParser(description='滑动窗口增强实验')
    parser.add_argument('--window_size', type=int, default=32, help='窗口大小')
    parser.add_argument('--stride', type=int, default=16, help='滑动步长')
    parser.add_argument('--skip_data_prep', action='store_true', help='跳过数据准备')
    parser.add_argument('--skip_pretrain', action='store_true', help='跳过预训练')
    parser.add_argument('--only_report', action='store_true', help='仅生成报告')
    args = parser.parse_args()
    
    print(f"""
{'='*80}
TwoTST 滑动窗口增强实验
{'='*80}
Window Size: {args.window_size}
Stride: {args.stride}
{'='*80}
    """)
    
    if args.only_report:
        # 仅生成报告
        results = []
        finetune_dir = os.path.join(PROJECT_ROOT, 'checkpoints_sw', 'finetune')
        if os.path.exists(finetune_dir):
            for exp_dir in Path(finetune_dir).iterdir():
                if exp_dir.is_dir():
                    results_file = exp_dir / 'results.json'
                    if results_file.exists():
                        with open(results_file) as f:
                            results.append(json.load(f))
        
        if results:
            generate_report(results, args.window_size, args.stride)
        else:
            print("没有找到实验结果")
        return
    
    # Step 1: 准备数据
    if not args.skip_data_prep:
        ret = prepare_sliding_window_data(args.window_size, args.stride)
        if ret != 0:
            print("数据准备失败！")
            return
    
    # Step 2: 创建配置文件
    config_files = create_sw_experiment_configs(args.window_size, args.stride)
    
    # Step 3: 预训练
    if not args.skip_pretrain:
        ret = pretrain_models(args.window_size)
        if ret != 0:
            print("预训练失败！")
            return
    
    # Step 4: 运行实验
    results = run_all_experiments(config_files)
    
    # Step 5: 生成报告
    if results:
        generate_report(results, args.window_size, args.stride)
    
    print("\n" + "="*80)
    print("所有实验完成！")
    print("="*80)


if __name__ == '__main__':
    main()
