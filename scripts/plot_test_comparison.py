"""
Plot TEST SET performance comparison for different fusion methods
Generates bar charts showing final test AUC and Accuracy
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set up matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 10


def load_test_results(base_dir, experiments):
    """Load test results from results.json files"""
    data = {}
    for exp_name, display_name in experiments:
        path = os.path.join(base_dir, exp_name, 'results.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                result = json.load(f)
            data[display_name] = {
                'auc': result.get('auc', 0),
                'accuracy': result.get('accuracy', 0),
                'sensitivity': result.get('sensitivity', 0),
                'specificity': result.get('specificity', 0),
                'f1': result.get('f1', 0)
            }
    return data


def main():
    output_dir = '/root/workplace/exp/TwoTST/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiments
    sw_experiments = [
        ('sw_baseline_cross_attention', 'Cross-Attention'),
        ('sw_baseline_bilinear', 'Bilinear'),
        ('sw_baseline_attention_pooling', 'Attn Pooling'),
        ('sw_baseline_gated', 'Gated'),
        ('sw_baseline_concat', 'Concat'),
    ]
    
    nosw_experiments = [
        ('baseline_concat', 'Concat'),
        ('baseline_bilinear', 'Bilinear'),
        ('baseline_gated', 'Gated'),
        ('baseline_cross_attention', 'Cross-Attention'),
        ('baseline_attention_pooling', 'Attn Pooling'),
    ]
    
    # Load data
    print("Loading test results...")
    sw_data = load_test_results('/root/workplace/exp/TwoTST/checkpoints_sw/finetune', sw_experiments)
    nosw_data = load_test_results('/root/workplace/exp/TwoTST/checkpoints/finetune', nosw_experiments)
    
    # ============================================================
    # Figure 1: Sliding Window - Test Set Performance
    # ============================================================
    print("\nGenerating sliding window test performance figure...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    methods = list(sw_data.keys())
    aucs = [sw_data[m]['auc'] for m in methods]
    accs = [sw_data[m]['accuracy'] for m in methods]
    f1s = [sw_data[m]['f1'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.25
    
    bars1 = ax.bar(x - width, aucs, width, label='AUC', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, accs, width, label='Accuracy', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label='F1 Score', color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Fusion Method')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance: Sliding Window (Sorted by Test AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim([0.8, 0.95])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    # Highlight best
    best_idx = np.argmax(aucs)
    ax.annotate('Best', xy=(best_idx - width, aucs[best_idx] + 0.01), 
               fontsize=10, fontweight='bold', color='#3498db', ha='center')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sw_test_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/sw_test_performance.pdf', bbox_inches='tight')
    print('Saved: sw_test_performance.png')
    plt.close()
    
    # ============================================================
    # Figure 2: Non-Sliding Window - Test Set Performance
    # ============================================================
    print("\nGenerating non-sliding window test performance figure...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Sort by AUC
    sorted_methods = sorted(nosw_data.keys(), key=lambda m: nosw_data[m]['auc'], reverse=True)
    aucs = [nosw_data[m]['auc'] for m in sorted_methods]
    accs = [nosw_data[m]['accuracy'] for m in sorted_methods]
    f1s = [nosw_data[m]['f1'] for m in sorted_methods]
    
    x = np.arange(len(sorted_methods))
    
    bars1 = ax.bar(x - width, aucs, width, label='AUC', color='#3498db', alpha=0.85)
    bars2 = ax.bar(x, accs, width, label='Accuracy', color='#2ecc71', alpha=0.85)
    bars3 = ax.bar(x + width, f1s, width, label='F1 Score', color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Fusion Method')
    ax.set_ylabel('Score')
    ax.set_title('Test Set Performance: Without Sliding Window (Sorted by Test AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(sorted_methods, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim([0.3, 0.8])
    ax.grid(True, alpha=0.3, axis='y')
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nosw_test_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/nosw_test_performance.pdf', bbox_inches='tight')
    print('Saved: nosw_test_performance.png')
    plt.close()
    
    # ============================================================
    # Figure 3: Combined Comparison (SW vs No-SW)
    # ============================================================
    print("\nGenerating combined comparison figure...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Use same order for both
    method_order = ['Cross-Attention', 'Bilinear', 'Attn Pooling', 'Gated', 'Concat']
    
    sw_aucs = [sw_data.get(m, {}).get('auc', 0) for m in method_order]
    nosw_aucs = [nosw_data.get(m, {}).get('auc', 0) for m in method_order]
    
    x = np.arange(len(method_order))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sw_aucs, width, label='With Sliding Window', color='#2ecc71', alpha=0.85)
    bars2 = ax.bar(x + width/2, nosw_aucs, width, label='Without Sliding Window', color='#e74c3c', alpha=0.85)
    
    ax.set_xlabel('Fusion Method')
    ax.set_ylabel('Test AUC')
    ax.set_title('Test AUC Comparison: With vs Without Sliding Window Augmentation')
    ax.set_xticks(x)
    ax.set_xticklabels(method_order, fontsize=11)
    ax.legend(loc='lower right')
    ax.set_ylim([0.5, 1.0])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                   xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Add improvement percentages
    for i, (sw, nosw) in enumerate(zip(sw_aucs, nosw_aucs)):
        if nosw > 0:
            improvement = (sw - nosw) / nosw * 100
            ax.annotate(f'+{improvement:.1f}%', xy=(i, sw + 0.02), 
                       fontsize=9, ha='center', color='#27ae60', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/test_auc_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/test_auc_comparison.pdf', bbox_inches='tight')
    print('Saved: test_auc_comparison.png')
    plt.close()
    
    # Print summary
    print("\n" + "=" * 70)
    print("TEST SET PERFORMANCE SUMMARY (Sorted by AUC)")
    print("=" * 70)
    print("\nWith Sliding Window:")
    for m in methods:
        d = sw_data[m]
        print(f"  {m:<18} AUC={d['auc']:.4f}  ACC={d['accuracy']:.4f}  F1={d['f1']:.4f}")
    
    print("\nWithout Sliding Window:")
    for m in sorted_methods:
        d = nosw_data[m]
        print(f"  {m:<18} AUC={d['auc']:.4f}  ACC={d['accuracy']:.4f}  F1={d['f1']:.4f}")


if __name__ == '__main__':
    main()
