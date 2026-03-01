#!/usr/bin/env python3
"""
最佳配置重复 5 次运行，验证 Test AUC 是否稳定在 ~0.74
使用不同 seed (42, 43, 44, 45, 46) 获得 5 组结果
"""

import os
import sys
import json
import yaml
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiments" / "group7_projection_fusion_attention_pooling.yaml"
SEEDS = [42, 43, 44, 45, 46]
# 结果保存到数据盘 autodl-tmp（若不存在则用项目目录）
_results = Path(os.environ.get("RESULTS_ROOT", "/root/autodl-tmp/TwoTST/results"))
RESULTS_ROOT = _results if _results.parent.exists() else PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_ROOT / "best_config_5x"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_once(seed: int) -> dict:
    """运行单次实验，返回 results"""
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    # 覆盖 seed
    config["training"]["seed"] = seed

    # 使用项目路径（若 autodl 不存在）
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

    # 每次使用独立保存目录，优先保存到数据盘 autodl-tmp
    run_name = f"projection_fusion_attention_pooling_seed{seed}"
    out_root = Path("/root/autodl-tmp/TwoTST") if Path("/root/autodl-tmp/TwoTST").exists() else PROJECT_ROOT
    config["finetune"]["save_dir"] = str(out_root / "checkpoints" / "finetune" / run_name)
    config["logging"]["log_dir"] = str(out_root / "logs" / run_name)
    config["experiment"]["name"] = run_name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"),
            "--config", tmp_path,
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, capture_output=True, text=True)
        res_path = Path(config["finetune"]["save_dir"]) / "results.json"
        with open(res_path) as f:
            return json.load(f)
    finally:
        os.unlink(tmp_path)


def main():
    print("=" * 60)
    print("Best Config x 5 Runs (seeds: 42, 43, 44, 45, 46)")
    print("=" * 60)

    results = []
    for i, seed in enumerate(SEEDS):
        print(f"\n>>> Run {i+1}/5 (seed={seed})")
        try:
            res = run_once(seed)
            auc = res["auc"]
            results.append({"seed": seed, "auc": auc, "accuracy": res.get("accuracy"), "f1": res.get("f1")})
            print(f"    Test AUC: {auc:.4f}")
        except Exception as e:
            print(f"    FAILED: {e}")
            results.append({"seed": seed, "auc": None, "error": str(e)})

    # 汇总
    aucs = [r["auc"] for r in results if r["auc"] is not None]
    if aucs:
        import numpy as np
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        print("\n" + "=" * 60)
        print("Summary")
        print("=" * 60)
        for r in results:
            print(f"  seed {r['seed']}: AUC {r['auc']:.4f}" if r['auc'] else f"  seed {r['seed']}: FAILED")
        print(f"\n  Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
        print(f"  Range: [{min(aucs):.4f}, {max(aucs):.4f}]")
        with open(OUTPUT_DIR / "summary.json", "w") as f:
            json.dump({"runs": results, "mean_auc": mean_auc, "std_auc": float(std_auc), "aucs": aucs}, f, indent=2)
        print(f"\nSaved to {OUTPUT_DIR / 'summary.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
