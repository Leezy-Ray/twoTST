#!/usr/bin/env python3
"""
对比学习消融：仅运行对比阶段，保存 train_loss/val_loss/train_alignment/val_alignment 曲线
用于方案 C 的分层评价：对比阶段用 alignment 和 loss 衡量

4 种策略：freeze_tst1+unfreeze_tst2, unfreeze_tst1+freeze_tst2, freeze_both, unfreeze_both
输出：各策略的 contrastive_history.json
"""

import os
import sys
import yaml
import subprocess
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs" / "experiments"
BASE_CONFIG = CONFIG_DIR / "group7_projection_fusion_attention_pooling.yaml"

CONTRASTIVE_FREEZE_VARIANTS = [
    ("freeze_tst1", True, False),
    ("freeze_tst2", False, True),
    ("freeze_both", True, True),
    ("unfreeze_both", False, False),
]


def run_contrastive_only(variant_name, freeze_tst1, freeze_tst2, output_root):
    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)

    exp_name = "contrastive_curves_" + variant_name
    ckpt_root = Path("/root/autodl-tmp/TwoTST")
    if not (ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt").exists():
        ckpt_root = PROJECT_ROOT

    config["pretrain"]["tst1"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")
    config["contrastive"]["load_checkpoint"] = None
    config["contrastive"]["freeze_tst1"] = freeze_tst1
    config["contrastive"]["freeze_tst2"] = freeze_tst2

    out_dir = output_root / "contrastive_curves" / exp_name
    config["finetune"]["save_dir"] = str(out_dir)
    config["experiment"]["name"] = exp_name
    config["logging"]["log_dir"] = str(out_dir / "logs")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"),
            "--config", tmp_path,
            "--contrastive_only",
        ]
        ret = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
        return ret.returncode == 0
    finally:
        os.unlink(tmp_path)


def main():
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/TwoTST"))
    if not output_root.exists():
        output_root = PROJECT_ROOT

    print("=" * 60)
    print("Contrastive Curves Ablation (contrastive-only)")
    print("=" * 60)
    print("Output:", str(output_root / "contrastive_curves"))
    print()

    for name, f1, f2 in CONTRASTIVE_FREEZE_VARIANTS:
        print(">>>", name, "(freeze_tst1=%s, freeze_tst2=%s)" % (f1, f2))
        ok = run_contrastive_only(name, f1, f2, output_root)
        print("    OK" if ok else "    FAILED")

    print()
    print("=" * 60)
    print("Run: python scripts/analysis/plot_contrastive_curves.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
