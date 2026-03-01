# TwoTST: Dual-Stream Self-Supervised Pretraining for ASD Diagnosis

[![GitHub](https://img.shields.io/badge/GitHub-Leezy--Ray/twoTST-blue)](https://github.com/Leezy-Ray/twoTST)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org)

**TwoTST** is a dual-stream self-supervised pretraining framework for fMRI-based autism spectrum disorder (ASD) diagnosis. It fuses temporal dynamics (BOLD time series) and functional connectivity (PCC matrix) via two independent Transformer encoders, contrastive learning alignment, and flexible fusion modules.

---

## Overview

TwoTST jointly models two complementary views of fMRI data:

- **TST1 (Transformer-TS)**: encodes raw BOLD time series using an ROI-level masking pretraining strategy (~19M parameters, `emb_dim=512`, 6 layers)
- **TST2 (Transformer-FC)**: encodes the upper-triangle PCC connectivity vector using element-level masking (~16M parameters, `d_model=256`, 2 layers)

After pretraining, an optional **contrastive learning** stage aligns the two representation spaces via InfoNCE loss. A **projection-based attention-pooling fusion** module then merges the two streams for downstream ASD/TC classification.

### Best Configuration (AUC ≈ 0.744 single-split, AUC = 0.7043 ± 0.040 over 5 seeds)

| Dimension | Choice |
|-----------|--------|
| Sliding window | Disabled |
| Pretraining | Non-sliding-window TST1 + TST2 |
| Contrastive learning | Enabled (freeze TST1, unfreeze TST2) |
| Fusion | `attention_pooling` |
| Finetuning | TST1 + TST2 unfrozen, projection head frozen |
| Total parameters | ~36.5M |

### LOSO Cross-site Generalization (19 sites, ABIDE I, subject-level majority_vote)

| Metric | Mean ± Std | 95% CI |
|--------|------------|--------|
| AUC | 0.6897 ± 0.1358 | [0.6254, 0.7460] |
| Accuracy | 0.6189 ± 0.1291 | [0.5586, 0.6790] |
| Sensitivity | 0.6184 ± 0.1990 | — |
| Specificity | 0.6293 ± 0.1929 | — |
| F1 | 0.6141 ± 0.1508 | — |

---

## Training Pipeline

```
fMRI Data (N × T × R)
        │
        ▼
┌───────────────────────┐
│  Phase 1: Pretrain    │
│  TST1  (time series)  │  ROI-level mask → reconstruction
│  TST2  (PCC vector)   │  element-level mask → reconstruction
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Phase 2: Contrastive │  InfoNCE, projection heads
│  (optional)           │  aligns TST1 & TST2 embedding spaces
└───────────────────────┘
        │
        ▼
┌───────────────────────┐
│  Phase 3: Finetune    │  projection + attention_pooling fusion
│  ASD / TC             │  → MLP classifier → AUC / ACC / F1
└───────────────────────┘
```

---

## Project Structure

```
TwoTST/
├── models/                          # Model definitions
│   ├── transformer_ts.py            # TST1: time-series Transformer
│   ├── transformer_fc.py            # TST2: connectivity Transformer
│   ├── fusion.py                    # 5 fusion strategies
│   ├── dual_stream.py               # DualStreamModel wrapper
│   └── __init__.py
│
├── pretrain/                        # Pretraining modules
│   ├── pretrain_ts.py               # TST1 pretraining
│   ├── pretrain_fc.py               # TST2 pretraining
│   ├── contrastive.py               # InfoNCE contrastive learning
│   ├── mask_utils.py                # Masking strategies
│   └── __init__.py
│
├── scripts/
│   ├── train/
│   │   ├── train_pretrain.py        # Unified pretraining entry
│   │   ├── train_finetune.py        # Finetuning entry (k-fold / LOSO)
│   │   └── generate_contrastive_checkpoints.py
│   ├── experiments/
│   │   └── run_experiment.py        # Single-config experiment runner
│   ├── ablation/                    # Ablation study scripts
│   │   ├── run_ablation_no_pretrain.py
│   │   ├── run_ablation_baseline_and_freeze.py
│   │   └── run_ablation_contrastive_freeze.py
│   ├── analysis/                    # Visualization & interpretability
│   │   ├── interpretability_gradients.py   # Gradient-based importance
│   │   ├── plot_interpretability.py        # Heatmap / bar plots
│   │   ├── plot_pretrain_loss_v3.py
│   │   ├── plot_fusion_comparison.py
│   │   └── analyze_attention.py
│   ├── validation/
│   │   └── collect_statistical_results.py
│   ├── data/
│   │   └── prepare_data.py
│   ├── run_best_config_5fold_loso.py   # Best config: 5-fold CV + LOSO
│   ├── run_best_config_5fold_loso.sh
│   ├── run_best_config_5x.py           # Best config: 5-seed repeatability
│   ├── run_fusion_5x.py                # Fusion ablation × 5 seeds
│   ├── run_statistical_validation.sh   # Full statistical validation
│   ├── run_ablation_*.sh
│   └── run_full_training_autodl.sh     # One-shot full training entry
│
├── utils/
│   ├── data_loader.py               # Dataset & DataLoader
│   ├── metrics.py                   # AUC, ACC, bootstrap CI
│   ├── splitters.py                 # Subject-level k-fold & LOSO splits
│   └── __init__.py
│
├── configs/
│   ├── experiments/                 # Per-experiment YAML configs
│   │   └── group7_projection_fusion_attention_pooling.yaml  # best config
│   └── default.yaml
│
├── api/                             # REST API for model serving
│   ├── app.py
│   ├── models/
│   ├── services/
│   └── utils/
│
├── docs/                            # Experiment docs & analysis
│   ├── OPTIMAL_CONFIGURATION.md
│   ├── STATISTICAL_VALIDATION_RESULTS.md
│   ├── ABLATION_RESULTS.md
│   ├── INTERPRETABILITY_RESULTS_SUMMARY.md
│   └── ...
│
├── requirements.txt
└── README.md
```

> **Note**: `data/`, `checkpoints/`, `logs/`, `results/` are excluded from git (see `.gitignore`). Place data on `/root/autodl-tmp/` on AutoDL instances.

---

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/Leezy-Ray/twoTST.git
cd twoTST
pip install -r requirements.txt
```

Requirements: Python ≥ 3.8, PyTorch ≥ 2.0, CUDA ≥ 11.8 (recommended: RTX 4090).

### 2. Data Preparation

Prepare `processed_data.pkl` from raw ABIDE fMRI data (CC200 atlas, 963 subjects):

```bash
python scripts/data/prepare_data.py \
    --data_path /path/to/fmri.npy \
    --output_dir data/processed \
    --n_rois 200 --time_points 100
```

Output fields in `processed_data.pkl`:

| Field | Shape | Description |
|-------|-------|-------------|
| `timeseries` | `(N, T, R)` | BOLD time series |
| `pcc_vectors` | `(N, R*(R-1)/2)` | PCC upper-triangle |
| `labels` | `(N,)` | 0=ASD, 1=TC |
| `subject_indices` | `(N,)` | For subject-level split |
| `site_ids` | `(N,)` | For LOSO evaluation |

### 3. Pretraining

```bash
# Pretrain both TST1 and TST2 sequentially
python scripts/train/train_pretrain.py \
    --data_path data/processed/processed_data.pkl \
    --pretrain_tst1 --pretrain_tst2 \
    --tst1_epochs 100 --tst2_epochs 100 \
    --batch_size 32 --lr 1e-4 \
    --save_dir checkpoints
```

Or run individually:

```bash
# TST1: ROI-level masking on BOLD time series
python pretrain/pretrain_ts.py \
    --data_path data/processed/processed_data.pkl \
    --epochs 100 --save_dir checkpoints/tst1

# TST2: element-level masking on PCC vector
python pretrain/pretrain_fc.py \
    --data_path data/processed/processed_data.pkl \
    --epochs 100 --mask_ratio 0.15 --save_dir checkpoints/tst2
```

### 4. Finetuning (Best Config)

```bash
python scripts/experiments/run_experiment.py \
    --config configs/experiments/group7_projection_fusion_attention_pooling.yaml
```

Or with the general finetuning script:

```bash
python scripts/train/train_finetune.py \
    --data_path data/processed/processed_data.pkl \
    --tst1_checkpoint checkpoints/tst1/tst1_best.pt \
    --tst2_checkpoint checkpoints/tst2/tst2_best.pt \
    --fusion_type attention_pooling \
    --eval_protocol kfold --n_folds 5 \
    --subject_agg_strategy majority_vote \
    --save_dir results/best_config
```

### 5. Statistical Validation (5-fold CV + LOSO)

```bash
# Run best config over 5-fold CV and LOSO in one shot
bash scripts/run_best_config_5fold_loso.sh

# Or separately:
python scripts/run_best_config_5fold_loso.py \
    --config configs/experiments/group7_projection_fusion_attention_pooling.yaml \
    --eval_protocol loso \
    --subject_agg_strategy majority_vote \
    --save_dir results/best_config_loso
```

### 6. Interpretability Analysis

```bash
# Compute gradient-based connection & ROI importance
python scripts/analysis/interpretability_gradients.py \
    --data_path data/processed/processed_data.pkl \
    --checkpoint checkpoints/finetune/projection_fusion_attention_pooling_unfrozen/best_model.pt \
    --config configs/experiments/group7_projection_fusion_attention_pooling.yaml \
    --output_dir results/interpretability \
    --target_class 1 --max_samples 500

# Plot heatmaps and bar charts
python scripts/analysis/plot_interpretability.py \
    --result_dir results/interpretability \
    --labels data/labels/cc200_coordinates.json \
    --output_dir results/interpretability
```

---

## Fusion Strategies

| Strategy | Description |
|----------|-------------|
| `concat` | Concatenate `[h_ts; h_fc]` |
| `gated` | Learnable gate: `g ⊙ h_ts + (1−g) ⊙ h_fc` |
| `cross_attention` | Cross-attention between the two streams |
| `bilinear` | Bilinear interaction |
| `attention_pooling` | Attention-weighted pooling (**best**) |

Fusion ablation results (5 seeds each, fixed best pretraining strategy):

| Fusion | AUC | ACC |
|--------|-----|-----|
| attention_pooling | 0.7043 ± 0.040 | 0.6352 ± 0.030 |
| gated | 0.7217 ± 0.045 | 0.6591 ± 0.027 |
| cross_attention | 0.7094 ± 0.043 | 0.6601 ± 0.042 |
| concat | 0.7132 ± 0.055 | 0.6497 ± 0.042 |
| bilinear | 0.6002 ± 0.034 | 0.5782 ± 0.028 |

---

## Evaluation Protocol

- **Subject-level split**: all windows from the same subject are placed in the same fold (`StratifiedGroupKFold`), preventing sliding-window data leakage.
- **LOSO**: `--eval_protocol loso` performs Leave-One-Site-Out across 19 ABIDE sites.
- **Subject-level aggregation**: test predictions are aggregated per subject via `majority_vote` or `prob_mean` before computing metrics.
- **Statistical reporting**: CV outputs `mean ± std` and bootstrap 95% CI; `summary.json` includes reproducibility metadata (PyTorch/CUDA/seed).

---

## Ablation Summary

| Condition | Test AUC |
|-----------|----------|
| Full model (best config) | **0.744** |
| No pretraining | ~0.630 |
| Freeze both TST1 & TST2 | ~0.633 |
| Freeze TST1 only | ~0.716 |
| No contrastive learning | ~0.732 |
| Sliding-window pretraining | ~0.730 |

See `docs/ABLATION_RESULTS.md` and `docs/OPTIMAL_CONFIGURATION.md` for detailed tables.

---

## API Server

A REST API is provided under `api/` for model-based prediction and connection analysis:

```bash
cd api
pip install -r requirements.txt
python app.py
```

Endpoints: `POST /predict`, `POST /analyze_connections`. See `api/README.md` for details.

---

## Common Issues

**Q: CUDA out of memory?**  
Reduce `batch_size` to 16 or enable gradient checkpointing.

**Q: LOSO fails with `LOSO requires site_ids`?**  
Your `processed_data.pkl` must include a `site_ids` field. Re-run `prepare_data.py` on ABIDE data which contains site metadata.

**Q: `git push` fails with "Password authentication is not supported"?**  
GitHub disabled password auth in 2021. Use a Personal Access Token (PAT) as the password:  
`echo "https://Leezy-Ray:<your_token>@github.com" > ~/.git-credentials`

---

## License

This project is for research use only.

---

**TwoTST** · [GitHub](https://github.com/Leezy-Ray/twoTST) · Dual-Stream Self-Supervised Pretraining for fMRI-based ASD Diagnosis
