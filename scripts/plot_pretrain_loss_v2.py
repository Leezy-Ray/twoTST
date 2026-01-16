"""
Plot pretraining loss comparison between sliding window and non-sliding window
Generates two separate figures: tst1_loss_comparison and tst2_loss_comparison
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
    # ============================================================
    # Load Sliding Window Data from history files
    # ============================================================
    sw_tst1_history = '/root/workplace/exp/TwoTST/checkpoints_sw/tst1/tst1_history.pt'
    sw_tst2_history = '/root/workplace/exp/TwoTST/checkpoints_sw/tst2/tst2_history.pt'
    
    sw_tst1 = load_history(sw_tst1_history)
    sw_tst2 = load_history(sw_tst2_history)
    
    # ============================================================
    # Non-Sliding Window TST1 Data (from pretrain_new.log)
    # ============================================================
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
    
    # ============================================================
    # Non-Sliding Window TST2 Data (from pretrain_new.log)
    # ============================================================
    nosw_tst2_train = [
        0.9614, 0.7884, 0.6722, 0.6411, 0.6327, 0.6288, 0.6248, 0.6236, 0.6212, 0.6194,
        0.6190, 0.6156, 0.6134, 0.6101, 0.6046, 0.6001, 0.5961, 0.5914, 0.5871, 0.5825,
        0.5779, 0.5745, 0.5719, 0.5679, 0.5647, 0.5628, 0.5601, 0.5586, 0.5572, 0.5549,
        0.5529, 0.5512, 0.5504, 0.5480, 0.5474, 0.5437, 0.5435, 0.5425, 0.5416, 0.5391,
        0.5385, 0.5363, 0.5347, 0.5358, 0.5325, 0.5311, 0.5303, 0.5305, 0.5290, 0.5272,
        0.5277, 0.5265, 0.5244, 0.5231, 0.5218, 0.5216, 0.5197, 0.5212, 0.5180, 0.5180,
        0.5172, 0.5165, 0.5163, 0.5156, 0.5147, 0.5150, 0.5140, 0.5120, 0.5125, 0.5120,
        0.5110, 0.5111, 0.5105, 0.5097, 0.5100, 0.5094, 0.5089, 0.5091, 0.5095, 0.5085,
        0.5086, 0.5072, 0.5088, 0.5075, 0.5081, 0.5069, 0.5064, 0.5062, 0.5072, 0.5067,
        0.5079, 0.5067, 0.5066, 0.5069, 0.5061, 0.5067, 0.5060, 0.5057, 0.5060, 0.5054
    ]
    
    nosw_tst2_val = [
        0.8661, 0.6747, 0.6109, 0.5891, 0.5851, 0.5802, 0.5749, 0.5778, 0.5758, 0.5749,
        0.5740, 0.5727, 0.5699, 0.5661, 0.5584, 0.5560, 0.5567, 0.5526, 0.5462, 0.5466,
        0.5442, 0.5377, 0.5422, 0.5356, 0.5350, 0.5326, 0.5281, 0.5294, 0.5299, 0.5283,
        0.5278, 0.5229, 0.5240, 0.5252, 0.5239, 0.5235, 0.5204, 0.5202, 0.5184, 0.5167,
        0.5180, 0.5160, 0.5181, 0.5178, 0.5126, 0.5154, 0.5142, 0.5103, 0.5125, 0.5140,
        0.5132, 0.5116, 0.5106, 0.5073, 0.5094, 0.5064, 0.5111, 0.5054, 0.5056, 0.5071,
        0.5085, 0.5081, 0.5039, 0.5067, 0.5047, 0.5052, 0.5044, 0.5046, 0.5062, 0.5055,
        0.5044, 0.5049, 0.5008, 0.5040, 0.5043, 0.5031, 0.5035, 0.5025, 0.5039, 0.5021,
        0.5034, 0.5011, 0.4996, 0.5044, 0.5033, 0.5001, 0.5033, 0.5007, 0.4999, 0.4981,
        0.5004, 0.4988, 0.5023, 0.5011, 0.5021, 0.5017, 0.5015, 0.5030, 0.5017, 0.5033
    ]
    
    epochs = np.arange(1, 101)
    
    # ============================================================
    # Figure 1: TST1 Loss Comparison
    # ============================================================
    fig1, axes1 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # TST1 Training Loss
    ax = axes1[0]
    ax.plot(epochs, sw_tst1['train_losses'], label='With Sliding Window (4,815 samples)', 
            color='#2ecc71', linewidth=2)
    ax.plot(epochs, nosw_tst1_train, label='Without Sliding Window (963 samples)', 
            color='#e74c3c', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('(a) TST1 Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.4, 1.4])
    
    # TST1 Validation Loss
    ax = axes1[1]
    ax.plot(epochs, sw_tst1['val_losses'], label='With Sliding Window (4,815 samples)', 
            color='#2ecc71', linewidth=2)
    ax.plot(epochs, nosw_tst1_val, label='Without Sliding Window (963 samples)', 
            color='#e74c3c', linewidth=2, linestyle='--')
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
    print('Saved: tst1_loss_comparison.png')
    plt.close()
    
    # ============================================================
    # Figure 2: TST2 Loss Comparison
    # ============================================================
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4.5))
    
    # TST2 Training Loss
    ax = axes2[0]
    ax.plot(epochs, sw_tst2['train_losses'], label='With Sliding Window (4,815 samples)', 
            color='#3498db', linewidth=2)
    ax.plot(epochs, nosw_tst2_train, label='Without Sliding Window (963 samples)', 
            color='#9b59b6', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('(a) TST2 Training Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.5, 1.0])
    
    # TST2 Validation Loss
    ax = axes2[1]
    ax.plot(epochs, sw_tst2['val_losses'], label='With Sliding Window (4,815 samples)', 
            color='#3498db', linewidth=2)
    ax.plot(epochs, nosw_tst2_val, label='Without Sliding Window (963 samples)', 
            color='#9b59b6', linewidth=2, linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('(b) TST2 Validation Loss')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([1, 100])
    ax.set_ylim([0.5, 0.9])
    
    plt.tight_layout()
    plt.savefig('/root/workplace/exp/TwoTST/results/tst2_loss_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig('/root/workplace/exp/TwoTST/results/tst2_loss_comparison.pdf', bbox_inches='tight')
    print('Saved: tst2_loss_comparison.png')
    plt.close()
    
    # ============================================================
    # Print Summary Statistics
    # ============================================================
    print("\n" + "=" * 60)
    print("TST1 Pre-training Summary")
    print("=" * 60)
    print(f"With Sliding Window:")
    print(f"  - Final Train Loss: {sw_tst1['train_losses'][-1]:.4f}")
    print(f"  - Final Val Loss: {sw_tst1['val_losses'][-1]:.4f}")
    print(f"  - Best Val Loss: {min(sw_tst1['val_losses']):.4f} (Epoch {np.argmin(sw_tst1['val_losses'])+1})")
    print(f"Without Sliding Window:")
    print(f"  - Final Train Loss: {nosw_tst1_train[-1]:.4f}")
    print(f"  - Final Val Loss: {nosw_tst1_val[-1]:.4f}")
    print(f"  - Best Val Loss: {min(nosw_tst1_val):.4f} (Epoch {np.argmin(nosw_tst1_val)+1})")
    
    print("\n" + "=" * 60)
    print("TST2 Pre-training Summary")
    print("=" * 60)
    print(f"With Sliding Window:")
    print(f"  - Final Train Loss: {sw_tst2['train_losses'][-1]:.4f}")
    print(f"  - Final Val Loss: {sw_tst2['val_losses'][-1]:.4f}")
    print(f"  - Best Val Loss: {min(sw_tst2['val_losses']):.4f} (Epoch {np.argmin(sw_tst2['val_losses'])+1})")
    print(f"Without Sliding Window:")
    print(f"  - Final Train Loss: {nosw_tst2_train[-1]:.4f}")
    print(f"  - Final Val Loss: {nosw_tst2_val[-1]:.4f}")
    print(f"  - Best Val Loss: {min(nosw_tst2_val):.4f} (Epoch {np.argmin(nosw_tst2_val)+1})")

if __name__ == '__main__':
    main()
