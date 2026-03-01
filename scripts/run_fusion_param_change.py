#!/usr/bin/env python3
"""
5 种融合方式 × 5 seeds，微调阶段全 unfrozen（TST1/TST2/proj1/proj2）
计算各模块的 Mean Relative Change（指标 C），统计各融合方式下模块参数平均变化程度
"""

import os
import sys
import json
import yaml
import subprocess
import tempfile
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
FUSION_CONFIGS = [
    "group7_projection_fusion_concat.yaml",
    "group7_projection_fusion_gated.yaml",
    "group7_projection_fusion_cross_attention.yaml",
    "group7_projection_fusion_bilinear.yaml",
    "group7_projection_fusion_attention_pooling.yaml",
]
SEEDS = [42, 43, 44, 45, 46]
MODULES = ["tst1", "tst2", "proj_head1", "proj_head2"]

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.analysis.param_change_metrics import compute_module_changes

_results = Path(os.environ.get("RESULTS_ROOT", "/root/autodl-tmp/TwoTST/results"))
RESULTS_ROOT = _results if _results.parent.exists() else PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "param_change"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_once(config_path: Path, fusion_name: str, seed: int) -> dict:
    """运行单次实验，返回 results 和 param_changes"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["training"]["seed"] = seed
    config["finetune"]["freeze_tst1"] = False
    config["finetune"]["freeze_tst2"] = False
    config["finetune"]["freeze_projection"] = False

    ckpt_root = Path("/root/autodl-tmp/TwoTST")
    if not (ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt").exists():
        ckpt_root = PROJECT_ROOT
    config["pretrain"]["tst1"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")
    cont_ckpt_path = ckpt_root / "checkpoints" / "contrastive_checkpoint.pt"
    if cont_ckpt_path.exists():
        config["contrastive"]["load_checkpoint"] = str(cont_ckpt_path)
    else:
        config["contrastive"]["load_checkpoint"] = None

    run_name = f"param_change_{fusion_name}_seed{seed}"
    out_root = Path("/root/autodl-tmp/TwoTST") if Path("/root/autodl-tmp/TwoTST").exists() else PROJECT_ROOT
    save_dir = out_root / "checkpoints" / "finetune" / run_name
    config["finetune"]["save_dir"] = str(save_dir)
    config["logging"]["log_dir"] = str(out_root / "logs" / run_name)
    config["experiment"]["name"] = run_name

    if (save_dir / "results.json").exists() and (save_dir / "best_model.pt").exists():
        with open(save_dir / "results.json") as f:
            results = json.load(f)
        cont_ckpt = torch.load(cont_ckpt_path, map_location="cpu", weights_only=False)
        best_model = torch.load(save_dir / "best_model.pt", map_location="cpu", weights_only=False)
        param_changes = compute_module_changes(cont_ckpt, best_model["model_state_dict"])
        return {"results": results, "param_changes": param_changes}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"), "--config", tmp_path],
            cwd=str(PROJECT_ROOT), check=True,
        )
    finally:
        os.unlink(tmp_path)

    with open(save_dir / "results.json") as f:
        results = json.load(f)

    cont_ckpt = torch.load(cont_ckpt_path, map_location="cpu", weights_only=False)
    best_model = torch.load(save_dir / "best_model.pt", map_location="cpu", weights_only=False)
    param_changes = compute_module_changes(cont_ckpt, best_model["model_state_dict"])

    return {"results": results, "param_changes": param_changes}


def main():
    fusion_names = [c.replace("group7_projection_fusion_", "").replace(".yaml", "") for c in FUSION_CONFIGS]
    print("=" * 70)
    print("5 Fusion Types × 5 Seeds, All Unfrozen, Mean Relative Change (Metric C)")
    print("=" * 70)

    all_data = {}
    for i, (cfg_file, fusion_name) in enumerate(zip(FUSION_CONFIGS, fusion_names)):
        config_path = CONFIG_DIR / cfg_file
        if not config_path.exists():
            print("Skip", cfg_file)
            continue
        print("\n>>> Fusion:", fusion_name)
        runs = []
        for j, seed in enumerate(SEEDS):
            print("    Run %d/5 (seed=%d)" % (j + 1, seed), end=" ")
            try:
                out = run_once(config_path, fusion_name, seed)
                runs.append({
                    "seed": seed,
                    "auc": out["results"]["auc"],
                    "param_changes": out["param_changes"],
                })
                print("AUC=%.4f  tst1=%.4f tst2=%.4f proj1=%.4f proj2=%.4f" % (
                    out["results"]["auc"],
                    out["param_changes"]["tst1"],
                    out["param_changes"]["tst2"],
                    out["param_changes"]["proj_head1"],
                    out["param_changes"]["proj_head2"],
                ))
            except Exception as e:
                print("FAILED:", e)
                runs.append({"seed": seed, "error": str(e)})

        stats = {}
        for mod in MODULES:
            vals = [r["param_changes"][mod] for r in runs if "param_changes" in r and mod in r["param_changes"] and not np.isnan(r["param_changes"][mod])]
            if vals:
                stats[mod] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        all_data[fusion_name] = {"runs": runs, "module_stats": stats}

    print("\n" + "=" * 70)
    print("Module Mean Relative Change (Mean ± Std over 5 seeds)")
    print("=" * 70)
    headers = ["Fusion"] + MODULES
    rows = []
    for fn, d in all_data.items():
        s = d.get("module_stats", {})
        row = [fn]
        for m in MODULES:
            row.append("%.4f±%.4f" % (s[m]["mean"], s[m]["std"]) if m in s else "-")
        rows.append(row)
    col_w = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    fmt = "  ".join("{%d:%d}" % (i, w) for i, w in enumerate(col_w))
    print(fmt.format(*headers))
    print("-" * 70)
    for row in rows:
        print(fmt.format(*row))

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)
    print("\nSaved:", OUTPUT_DIR / "summary.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
