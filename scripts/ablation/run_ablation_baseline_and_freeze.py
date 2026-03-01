#!/usr/bin/env python3
"""
消融实验：baseline（无对比学习）+ 冻结策略对比
1. group1_baseline_* (5种融合) x 4种冻结 = 20 实验
2. group7_projection_* (7种) x 4种冻结 = 28 实验
总计 48 实验，结果保存到 /root/autodl-tmp/TwoTST
"""

import os
import sys
import yaml
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT_ROOT = Path("/root/autodl-tmp/TwoTST")
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
LOG_DIR = OUTPUT_ROOT / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 冻结策略: (freeze_tst1, freeze_tst2)
FREEZE_VARIANTS = [
    ("unfrozen", False, False),
    ("freeze_both", True, True),
    ("freeze_tst1", True, False),
    ("freeze_tst2", False, True),
]

BASELINE_CONFIGS = [
    "group1_baseline_concat.yaml",
    "group1_baseline_gated.yaml",
    "group1_baseline_cross_attention.yaml",
    "group1_baseline_bilinear.yaml",
    "group1_baseline_attention_pooling.yaml",
]

PROJECTION_CONFIGS = [
    "group7_projection_tst1_only.yaml",
    "group7_projection_tst2_only.yaml",
    "group7_projection_fusion_concat.yaml",
    "group7_projection_fusion_gated.yaml",
    "group7_projection_fusion_cross_attention.yaml",
    "group7_projection_fusion_bilinear.yaml",
    "group7_projection_fusion_attention_pooling.yaml",
]


def _replace_paths(obj, output_root):
    """递归替换路径"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and "checkpoints" in v:
                obj[k] = v.replace("/root/workplace/exp/TwoTST/checkpoints", str(output_root / "checkpoints"))
                obj[k] = obj[k].replace("/root/autodl-tmp/TwoTST/checkpoints", str(output_root / "checkpoints"))
            else:
                _replace_paths(v, output_root)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str) and "checkpoints" in v:
                obj[i] = v.replace("/root/workplace/exp/TwoTST/checkpoints", str(output_root / "checkpoints"))
                obj[i] = obj[i].replace("/root/autodl-tmp/TwoTST/checkpoints", str(output_root / "checkpoints"))
            else:
                _replace_paths(v, output_root)


def run_experiment(base_config_path, freeze_name, freeze_tst1, freeze_tst2, output_root):
    """加载配置，修改冻结策略和路径，运行实验"""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    base_name = config.get("experiment", {}).get("name", Path(base_config_path).stem)
    exp_name = f"{base_name}_{freeze_name}"

    # 更新路径为 autodl-tmp
    _replace_paths(config, output_root)
    config["pretrain"]["tst1"]["checkpoint"] = str(output_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(output_root / "checkpoints" / "tst2" / "tst2_best.pt")

    # 对比学习 checkpoint
    if config.get("contrastive", {}).get("enabled"):
        config["contrastive"]["load_checkpoint"] = str(output_root / "checkpoints" / "contrastive_checkpoint.pt")

    # 更新 finetune 冻结
    config["finetune"]["freeze_tst1"] = freeze_tst1
    config["finetune"]["freeze_tst2"] = freeze_tst2
    config["finetune"]["save_dir"] = str(output_root / "checkpoints" / "finetune" / exp_name)

    # 更新 experiment 和 logging
    if "experiment" not in config:
        config["experiment"] = {}
    config["experiment"]["name"] = exp_name
    config["logging"]["log_dir"] = str(output_root / "logs" / exp_name)

    # 写入临时配置
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        log_path = LOG_DIR / f"{exp_name}.log"
        cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"), "--config", tmp_path]
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd, cwd=str(PROJECT_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1
            )
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)
                logf.flush()
            proc.wait()
        return proc.returncode == 0
    finally:
        os.unlink(tmp_path)


def main(limit=0):
    os.chdir(PROJECT_ROOT)
    print("=" * 60)
    print("Ablation: Baseline + Freeze Variants")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    total = 0
    failed = []

    # 1. Baseline
    print("\n=== [1/2] group1_baseline (5 fusion x 4 freeze) ===")
    for cfg_name in BASELINE_CONFIGS:
        if limit > 0 and total >= limit:
            break
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.exists():
            print(f"  Skip (not found): {cfg_name}")
            continue
        for freeze_name, f1, f2 in FREEZE_VARIANTS:
            if limit > 0 and total >= limit:
                break
            total += 1
            print(f"\n>>> [{total}] {Path(cfg_name).stem}_{freeze_name}")
            ok = run_experiment(cfg_path, freeze_name, f1, f2, OUTPUT_ROOT)
            if not ok:
                failed.append(f"{Path(cfg_name).stem}_{freeze_name}")
            print(f"    {'OK' if ok else 'FAILED'}")

    # 2. Projection (contrastive)
    print("\n=== [2/2] group7_projection (7 types x 4 freeze) ===")
    for cfg_name in PROJECTION_CONFIGS:
        if limit > 0 and total >= limit:
            break
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.exists():
            print(f"  Skip (not found): {cfg_name}")
            continue
        for freeze_name, f1, f2 in FREEZE_VARIANTS:
            if limit > 0 and total >= limit:
                break
            total += 1
            print(f"\n>>> [{total}] {Path(cfg_name).stem}_{freeze_name}")
            ok = run_experiment(cfg_path, freeze_name, f1, f2, OUTPUT_ROOT)
            if not ok:
                failed.append(f"{Path(cfg_name).stem}_{freeze_name}")
            print(f"    {'OK' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print(f"Completed {total} experiments. Failed: {len(failed)}")
    if failed:
        print("Failed:", failed)
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 个实验（0=全部）")
    args = parser.parse_args()
    main(limit=args.limit)
