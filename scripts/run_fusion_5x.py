#!/usr/bin/env python3
"""
5 种融合方式 × 5 次随机实验
配置：非滑窗、对比学习 freeze TST1、微调 unfrozen、projection 冻结
输出：各融合方式的 Mean±Std (AUC, ACC, Sensitivity, Specificity, F1)
"""

import os
import sys
import json
import yaml
import subprocess
import tempfile
from pathlib import Path
import numpy as np

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
METRICS = ["auc", "accuracy", "sensitivity", "specificity", "f1"]
# 结果保存到数据盘 autodl-tmp（若不存在则用项目目录）
_results = Path(os.environ.get("RESULTS_ROOT", "/root/autodl-tmp/TwoTST/results"))
RESULTS_ROOT = _results if _results.parent.exists() else PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "fusion_5x"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_once(config_path: Path, fusion_name: str, seed: int) -> dict:
    """运行单次实验，返回 results"""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    config["training"]["seed"] = seed

    ckpt_root = Path("/root/autodl-tmp/TwoTST")
    if not (ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt").exists():
        ckpt_root = PROJECT_ROOT
    config["pretrain"]["tst1"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")
    cont_ckpt = ckpt_root / "checkpoints" / "contrastive_checkpoint.pt"
    if cont_ckpt.exists():
        config["contrastive"]["load_checkpoint"] = str(cont_ckpt)
    else:
        config["contrastive"]["load_checkpoint"] = None

    run_name = f"projection_fusion_{fusion_name}_seed{seed}"
    # 保存到数据盘 autodl-tmp（若存在）
    out_root = Path("/root/autodl-tmp/TwoTST") if Path("/root/autodl-tmp/TwoTST").exists() else PROJECT_ROOT
    save_dir = out_root / "checkpoints" / "finetune" / run_name
    config["finetune"]["save_dir"] = str(save_dir)
    config["logging"]["log_dir"] = str(out_root / "logs" / run_name)
    config["experiment"]["name"] = run_name

    res_path = save_dir / "results.json"
    if res_path.exists():
        with open(res_path) as f:
            return json.load(f)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"),
            "--config", tmp_path,
        ]
        # 不捕获输出，便于在终端实时查看训练进度（epoch、loss、AUC 等）
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        res_path = Path(config["finetune"]["save_dir"]) / "results.json"
        with open(res_path) as f:
            return json.load(f)
    finally:
        os.unlink(tmp_path)


def main():
    # fusion_name 从配置文件名提取，如 projection_fusion_concat -> concat
    fusion_names = []
    for cfg in FUSION_CONFIGS:
        # group7_projection_fusion_concat.yaml -> concat
        name = cfg.replace("group7_projection_fusion_", "").replace(".yaml", "")
        fusion_names.append(name)

    print("=" * 70)
    print("5 Fusion Types × 5 Seeds (concat, gated, cross_attention, bilinear, attention_pooling)")
    print("=" * 70)

    all_results = {}
    for i, (cfg_file, fusion_name) in enumerate(zip(FUSION_CONFIGS, fusion_names)):
        config_path = CONFIG_DIR / cfg_file
        if not config_path.exists():
            print(f"Skip {cfg_file} (not found)")
            continue

        print(f"\n>>> Fusion: {fusion_name} ({i+1}/{len(FUSION_CONFIGS)})")
        runs = []
        for j, seed in enumerate(SEEDS):
            print(f"    Run {j+1}/5 (seed={seed})", end=" ")
            try:
                res = run_once(config_path, fusion_name, seed)
                runs.append({
                    "seed": seed,
                    "auc": res.get("auc"),
                    "accuracy": res.get("accuracy"),
                    "sensitivity": res.get("sensitivity"),
                    "specificity": res.get("specificity"),
                    "f1": res.get("f1"),
                })
                print(f"AUC={res['auc']:.4f}")
            except Exception as e:
                print(f"FAILED: {e}")
                runs.append({"seed": seed, "error": str(e)})

        # 统计
        stats = {}
        for m in METRICS:
            vals = [r.get(m) for r in runs if r.get(m) is not None]
            if vals:
                stats[m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        all_results[fusion_name] = {"runs": runs, "stats": stats}

    # 汇总表
    print("\n" + "=" * 70)
    print("Summary: Mean ± Std")
    print("=" * 70)

    headers = ["Fusion", "AUC", "Accuracy", "Sensitivity", "Specificity", "F1"]
    rows = []
    for fusion_name, data in all_results.items():
        s = data.get("stats", {})
        row = [fusion_name]
        for m in METRICS:
            if m in s:
                row.append(f"{s[m]['mean']:.4f}±{s[m]['std']:.4f}")
            else:
                row.append("-")
        rows.append(row)

    col_widths = [max(len(str(c)) for c in col) for col in zip(headers, *[r for r in rows])]
    fmt = "  ".join(f"{{:{w}}}" for w in col_widths)
    print(fmt.format(*headers))
    print("-" * 70)
    for row in rows:
        print(fmt.format(*row))

    with open(OUTPUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
