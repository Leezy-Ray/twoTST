"""
Plot fusion method comparison for sliding window experiments
Generates training loss and validation AUC curves for different fusion methods
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# Set up matplotlib
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9


def load_experiment_data(base_dir, experiments):
    """Load training history from results.json files"""
    data = {}
    for exp_name, display_name in experiments:
        path = os.path.join(base_dir, exp_name, 'results.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                result = json.load(f)
            data[display_name] = {
                'train_losses': result.get('train_losses', []),
                'val_aucs': result.get('val_aucs', []),
                'val_accs': result.get('val_accs', []),
                'best_epoch': result.get('best_epoch', 0),
                'auc': result.get('auc', 0),
                'accuracy': result.get('accuracy', 0)
            }
    return data


def main():
    output_dir = '/root/workplace/exp/TwoTST/results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Define experiments
    sw_experiments = [
        ('sw_baseline_concat', 'Concat'),
        ('sw_baseline_gated', 'Gated'),
        ('sw_baseline_cross_attention', 'Cross-Attention'),
        ('sw_baseline_bilinear', 'Bilinear'),
        ('sw_baseline_attention_pooling', 'Attention Pooling'),
    ]
    
    # Color scheme
    colors = {
        'Concat': '#e74c3c',
        'Gated': '#3498db',
        'Cross-Attention': '#2ecc71',
        'Bilinear': '#9b59b6',
        'Attention Pooling': '#f39c12'
    }
    
    # ============================================================
    # Load sliding window data
    # ============================================================
    print("Loading sliding window experiment data...")
    sw_data = load_experiment_data(
        '/root/workplace/exp/TwoTST/checkpoints_sw/finetune',
        sw_experiments
    )
    
    for name, d in sw_data.items():
        print(f"  {name}: {len(d['train_losses'])} epochs, AUC={d['auc']:.4f}")
    
    # ============================================================
    # Figure 1: Sliding Window - Training Loss & Validation AUC
    # ============================================================
    print("\nGenerating sliding window fusion comparison figure...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Training Loss
    ax = axes[0]
    for name, d in sw_data.items():
        if len(d['train_losses']) > 0:
            epochs = np.arange(1, len(d['train_losses']) + 1)
            ax.plot(epochs, d['train_losses'], label=f"{name} (AUC={d['auc']:.3f})", 
                    color=colors[name], linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Training Loss')
    ax.set_title('(a) Training Loss Comparison')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    
    # Plot 2: Validation AUC
    ax = axes[1]
    for name, d in sw_data.items():
        if len(d['val_aucs']) > 0:
            epochs = np.arange(1, len(d['val_aucs']) + 1)
            # Show Test AUC in legend (final performance)
            ax.plot(epochs, d['val_aucs'], label=f"{name} (Test={d['auc']:.3f})", 
                    color=colors[name], linewidth=2, alpha=0.8)
            # Mark best epoch
            best_idx = np.argmax(d['val_aucs'])
            ax.scatter([best_idx + 1], [d['val_aucs'][best_idx]], 
                      color=colors[name], s=100, zorder=5, marker='*')
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation AUC')
    ax.set_title('(b) Validation AUC (Legend shows Test AUC)')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.65, 0.96])
    
    plt.suptitle('Sliding Window: Fusion Method Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sw_fusion_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/sw_fusion_comparison.pdf', bbox_inches='tight')
    print('Saved: sw_fusion_comparison.png')
    plt.close()
    
    # ============================================================
    # Figure 2: Non-Sliding Window (if data available)
    # ============================================================
    nosw_experiments = [
        ('baseline_concat', 'Concat'),
        ('baseline_gated', 'Gated'),
        ('baseline_cross_attention', 'Cross-Attention'),
        ('baseline_bilinear', 'Bilinear'),
        ('baseline_attention_pooling', 'Attention Pooling'),
    ]
    
    print("\nLoading non-sliding window experiment data...")
    nosw_data = load_experiment_data(
        '/root/workplace/exp/TwoTST/checkpoints/finetune',
        nosw_experiments
    )
    
    # Check if we have training history
    has_history = any(len(d['train_losses']) > 0 for d in nosw_data.values())
    
    if has_history:
        print("Generating non-sliding window fusion comparison figure...")
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot Training Loss
        ax = axes[0]
        for name, d in nosw_data.items():
            if len(d['train_losses']) > 0:
                epochs = np.arange(1, len(d['train_losses']) + 1)
                ax.plot(epochs, d['train_losses'], label=f"{name} (AUC={d['auc']:.3f})", 
                        color=colors[name], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('(a) Training Loss Comparison')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        # Plot Validation AUC
        ax = axes[1]
        for name, d in nosw_data.items():
            if len(d['val_aucs']) > 0:
                epochs = np.arange(1, len(d['val_aucs']) + 1)
                ax.plot(epochs, d['val_aucs'], label=f"{name} (AUC={d['auc']:.3f})", 
                        color=colors[name], linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation AUC')
        ax.set_title('(b) Validation AUC Comparison')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Without Sliding Window: Fusion Method Comparison', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/nosw_fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/nosw_fusion_comparison.pdf', bbox_inches='tight')
        print('Saved: nosw_fusion_comparison.png')
        plt.close()
    else:
        print("Non-sliding window experiments don't have training history data.")
        print("Creating bar chart comparison instead...")
        
        # Create bar chart for final results
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(nosw_data.keys())
        aucs = [nosw_data[m]['auc'] for m in methods]
        accs = [nosw_data[m]['accuracy'] for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, aucs, width, label='AUC', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='#2ecc71', alpha=0.8)
        
        ax.set_xlabel('Fusion Method')
        ax.set_ylabel('Score')
        ax.set_title('Without Sliding Window: Fusion Method Performance')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=15, ha='right')
        ax.legend()
        ax.set_ylim([0.4, 0.85])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/nosw_fusion_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/nosw_fusion_comparison.pdf', bbox_inches='tight')
        print('Saved: nosw_fusion_comparison.png (bar chart)')
        plt.close()


if __name__ == '__main__':
    main()
