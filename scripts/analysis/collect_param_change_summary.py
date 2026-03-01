#!/usr/bin/env python3
"""从已有的 param_change_* 目录收集并汇总 Mean Relative Change"""
import json
import numpy as np
import torch
from pathlib import Path

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.analysis.param_change_metrics import compute_module_changes

FUSIONS = ["concat", "gated", "cross_attention", "bilinear", "attention_pooling"]
SEEDS = [42, 43, 44, 45, 46]
MODULES = ["tst1", "tst2", "proj_head1", "proj_head2"]

out_root = Path("/root/autodl-tmp/TwoTST")
if not out_root.exists():
    out_root = PROJECT_ROOT
ckpt_dir = out_root / "checkpoints" / "finetune"
cont_path = out_root / "checkpoints" / "contrastive_checkpoint.pt"
results_root = out_root / "results" if (out_root / "results").exists() else PROJECT_ROOT / "results"
out_dir = results_root / "param_change"
out_dir.mkdir(parents=True, exist_ok=True)

cont_ckpt = torch.load(cont_path, map_location="cpu", weights_only=False)
all_data = {}

for fusion in FUSIONS:
    runs = []
    for seed in SEEDS:
        d = ckpt_dir / f"param_change_{fusion}_seed{seed}"
        if not (d / "results.json").exists() or not (d / "best_model.pt").exists():
            continue
        with open(d / "results.json") as f:
            res = json.load(f)
        best = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
        changes = compute_module_changes(cont_ckpt, best["model_state_dict"])
        runs.append({"seed": seed, "auc": res["auc"], "param_changes": changes})
    if not runs:
        continue
    stats = {}
    for m in MODULES:
        vals = [r["param_changes"][m] for r in runs if not np.isnan(r["param_changes"].get(m, np.nan))]
        if vals:
            stats[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    all_data[fusion] = {"runs": runs, "module_stats": stats}

print("Module Mean Relative Change (Mean ± Std)")
print("=" * 80)
headers = ["Fusion"] + MODULES
rows = []
for fn, d in all_data.items():
    s = d.get("module_stats", {})
    row = [fn]
    for m in MODULES:
        row.append("%.4f±%.4f" % (s[m]["mean"], s[m]["std"]) if m in s else "-")
    rows.append(row)
if rows:
    col_w = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    fmt = "  ".join("{%d:%d}" % (i, w) for i, w in enumerate(col_w))
    print(fmt.format(*headers))
    print("-" * 80)
    for row in rows:
        print(fmt.format(*row))

with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
    json.dump(all_data, f, indent=2, ensure_ascii=False)
print("\nSaved:", out_dir / "summary.json")
