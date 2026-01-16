"""
Plot pretraining loss comparison between sliding window and non-sliding window
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

# Set up matplotlib for better quality
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

def load_history(path):
    """Load training history from .pt file"""
    if os.path.exists(path):
        data = torch.load(path, map_location='cpu', weights_only=False)
        return data
    return None

def main():
    # Paths
    sw_tst1_history = '/root/workplace/exp/TwoTST/checkpoints_sw/tst1/tst1_history.pt'
    sw_tst2_history = '/root/workplace/exp/TwoTST/checkpoints_sw/tst2/tst2_history.pt'
    
    # Load sliding window histories
    sw_tst1 = load_history(sw_tst1_history)
    sw_tst2 = load_history(sw_tst2_history)
    
    # Non-sliding window data from logs (parsed manually)
    # TST1 non-sliding window
    nosw_tst1_train = [
        1.3277, 0.9501, 0.8800, 0.8310, 0.7909, 0.7621, 0.7220, 0.7020, 0.6821, 0.6577,
        0.6447, 0.6424, 0.6234, 0.6127, 0.6138, 0.6066, 0.6025, 0.5969, 0.5869, 0.5847,
        0.5697, 0.5719, 0.5634, 0.5666, 0.5638, 0.5538, 0.5491, 0.5503, 0.5419, 0.5454,
        0.5452, 0.5315, 0.5365, 0.5418, 0.5299, 0.5387, 0.5298, 0.5291, 0.5279, 0.5259,
        0.5110, 0.5126, 0.5237, 0.5215, 0.5210, 0.5146, 0.5219, 0.5166, 0.5099, 0.5068,
        0.5008, 0.5001, 0.5061, 0.5040, 0.4986, 0.5030, 0.5006, 0.5018, 0.4955, 0.4946,
        0.5054, 0.4998, 0.5063, 0.4981, 0.5020, 0.4998, 0.4961, 0.4989, 0.5028, 0.4966,
        0.4925, 0.4901, 0.4963, 0.4906, 0.4827, 0.4989, 0.5003, 0.4855, 0.4964, 0.4926,
        0.4948, 0.4807, 0.4977, 0.4826, 0.5023, 0.4902, 0.4822, 0.4974, 0.4978, 0.4924,
        0.4964, 0.4948, 0.4957, 0.4858, 0.4865, 0.4935, 0.4935, 0.4917, 0.4877, 0.4974
    ]
    
    nosw_tst1_val = [
        1.0027, 0.8522, 0.7962, 0.7633, 0.7199, 0.7207, 0.6875, 0.6454, 0.6199, 0.6246,
        0.6259, 0.5913, 0.5644, 0.5315, 0.5747, 0.5317, 0.5426, 0.5594, 0.5333, 0.5428,
        0.5172, 0.5186, 0.5258, 0.5156, 0.5037, 0.4730, 0.5316, 0.4887, 0.4948, 0.5150,
        0.4829, 0.4827, 0.5139, 0.4979, 0.4558, 0.4713, 0.4649, 0.4625, 0.4774, 0.4979,
        0.4881, 0.4679, 0.4659, 0.4836, 0.4598, 0.4970, 0.4912, 0.4824, 0.4737, 0.4627,
        0.4724, 0.4759, 0.4687, 0.4425, 0.4623, 0.4777, 0.4431, 0.4577, 0.4581, 0.4860,
        0.4540, 0.4659, 0.4567, 0.4680, 0.4669, 0.4507, 0.4449, 0.4630, 0.4584, 0.4467,
        0.4362, 0.4596, 0.4826, 0.4683, 0.4524, 0.4565, 0.4582, 0.4755, 0.4444, 0.4412,
        0.4629, 0.4527, 0.4743, 0.4366, 0.4510, 0.4541, 0.4519, 0.4173, 0.4547, 0.4345,
        0.4686, 0.4306, 0.4741, 0.4668, 0.4725, 0.4372, 0.4486, 0.4662, 0.4846, 0.4230
    ]
    
    # TST2 non-sliding window (from log - only partial, need to extend)
    nosw_tst2_train = [
        0.9614, 0.7884, 0.6722, 0.6411, 0.6327, 0.6288, 0.6248, 0.6236
    ]
    nosw_tst2_val = [
        0.8661, 0.6747, 0.6109, 0.5891, 0.5851, 0.5802, 0.5749, 0.5778
    ]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    epochs = np.arange(1, 101)
    
    # Plot 1: TST1 Training Loss Comparison
    ax = axes[0, 0]
    ax.plot(epochs, sw_tst1['train_losses'], label='With Sliding Window', color='#2ecc71', linewidth=1.5)
    ax.plot(epochs, nosw_tst1_train, label='Without Sliding Window', color='#e74c3c', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('TST1 (ROI Transformer) Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    
    # Plot 2: TST1 Validation Loss Comparison
    ax = axes[0, 1]
    ax.plot(epochs, sw_tst1['val_losses'], label='With Sliding Window', color='#2ecc71', linewidth=1.5)
    ax.plot(epochs, nosw_tst1_val, label='Without Sliding Window', color='#e74c3c', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('TST1 (ROI Transformer) Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    
    # Plot 3: TST2 Training Loss Comparison
    ax = axes[1, 0]
    ax.plot(epochs, sw_tst2['train_losses'], label='With Sliding Window', color='#3498db', linewidth=1.5)
    # For non-SW TST2, we only have partial data, so we'll use what we have
    epochs_nosw_tst2 = np.arange(1, len(nosw_tst2_train) + 1)
    ax.plot(epochs_nosw_tst2, nosw_tst2_train, label='Without Sliding Window', color='#9b59b6', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('TST2 (PCC Transformer) Training Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.annotate('Data incomplete\nfor non-SW', xy=(50, 0.7), fontsize=8, color='gray')
    
    # Plot 4: TST2 Validation Loss Comparison
    ax = axes[1, 1]
    ax.plot(epochs, sw_tst2['val_losses'], label='With Sliding Window', color='#3498db', linewidth=1.5)
    ax.plot(epochs_nosw_tst2, nosw_tst2_val, label='Without Sliding Window', color='#9b59b6', linewidth=1.5, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('TST2 (PCC Transformer) Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.annotate('Data incomplete\nfor non-SW', xy=(50, 0.65), fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig('/root/workplace/exp/TwoTST/results/pretrain_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/root/workplace/exp/TwoTST/results/pretrain_loss_comparison.pdf', bbox_inches='tight')
    print('Saved to /root/workplace/exp/TwoTST/results/pretrain_loss_comparison.png')
    
    # Also create a simplified version with just TST1 (complete data)
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # TST1 Training Loss
    ax = axes2[0]
    ax.plot(epochs, sw_tst1['train_losses'], label='With Sliding Window (4,815 samples)', color='#2ecc71', linewidth=2)
    ax.plot(epochs, nosw_tst1_train, label='Without Sliding Window (963 samples)', color='#e74c3c', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('(a) TST1 Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.4, 1.4])
    
    # TST1 Validation Loss
    ax = axes2[1]
    ax.plot(epochs, sw_tst1['val_losses'], label='With Sliding Window (4,815 samples)', color='#2ecc71', linewidth=2)
    ax.plot(epochs, nosw_tst1_val, label='Without Sliding Window (963 samples)', color='#e74c3c', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('(b) TST1 Validation Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.35, 1.1])
    
    plt.tight_layout()
    plt.savefig('/root/workplace/exp/TwoTST/results/tst1_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/root/workplace/exp/TwoTST/results/tst1_loss_comparison.pdf', bbox_inches='tight')
    print('Saved to /root/workplace/exp/TwoTST/results/tst1_loss_comparison.png')
    
    print("\n=== Summary Statistics ===")
    print(f"TST1 with SW - Final Train Loss: {sw_tst1['train_losses'][-1]:.4f}, Final Val Loss: {sw_tst1['val_losses'][-1]:.4f}")
    print(f"TST1 without SW - Final Train Loss: {nosw_tst1_train[-1]:.4f}, Final Val Loss: {nosw_tst1_val[-1]:.4f}")
    print(f"TST2 with SW - Final Train Loss: {sw_tst2['train_losses'][-1]:.4f}, Final Val Loss: {sw_tst2['val_losses'][-1]:.4f}")

if __name__ == '__main__':
    main()
