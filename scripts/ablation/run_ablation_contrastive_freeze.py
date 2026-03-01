#!/usr/bin/env python3
"""
消融实验：对比学习阶段的 TST freeze 策略对微调阶段融合的影响

在 attention_pooling + 非滑窗 条件下，对比 4 种对比学习 freeze 策略：
1. freeze_tst1, unfreeze_tst2 (当前默认)
2. unfreeze_tst1, freeze_tst2
3. freeze_both
4. unfreeze_both

每个实验：不加载已有 contrastive checkpoint，从 pretrained TST 重新运行对比学习，
        再以 unfrozen 进行微调。
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
LOG_DIR = PROJECT_ROOT / "logs" / "ablation_contrastive_freeze"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# (名称, contrastive freeze_tst1, contrastive freeze_tst2)
CONTRASTIVE_FREEZE_VARIANTS = [
    ("cont_freeze_tst1", True, False),   # 当前默认
    ("cont_freeze_tst2", False, True),
    ("cont_freeze_both", True, True),
    ("cont_unfreeze_both", False, False),
]


def _replace_paths(obj, base_paths):
    """替换 checkpoint 路径为项目内路径"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                if "/root/autodl-tmp/TwoTST" in v:
                    obj[k] = v.replace("/root/autodl-tmp/TwoTST", str(base_paths["root"]))
                if "/root/workplace/exp/TwoTST" in v and base_paths.get("workspace"):
                    obj[k] = v.replace("/root/workplace/exp/TwoTST", str(base_paths["workspace"]))
            else:
                _replace_paths(v, base_paths)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, str):
                if "/root/autodl-tmp/TwoTST" in v:
                    obj[i] = v.replace("/root/autodl-tmp/TwoTST", str(base_paths["root"]))
            else:
                _replace_paths(v, base_paths)


def run_experiment(variant_name, cont_freeze_tst1, cont_freeze_tst2, output_root):
    """运行单次消融实验"""
    with open(BASE_CONFIG) as f:
        config = yaml.safe_load(f)

    exp_name = f"projection_fusion_attention_pooling_{variant_name}"
    base_paths = {
        "root": output_root,
        "workspace": PROJECT_ROOT,
    }

    # 路径：优先使用 output_root（autodl），否则用项目路径
    autodl_tst1 = output_root / "checkpoints" / "tst1" / "tst1_best.pt"
    proj_tst1 = PROJECT_ROOT / "checkpoints" / "tst1" / "tst1_best.pt"
    ckpt_root = output_root if autodl_tst1.exists() else PROJECT_ROOT
    config["pretrain"]["tst1"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst1" / "tst1_best.pt")
    config["pretrain"]["tst2"]["checkpoint"] = str(ckpt_root / "checkpoints" / "tst2" / "tst2_best.pt")

    # 关键：不加载已有 contrastive checkpoint，强制重新训练对比学习
    config["contrastive"]["load_checkpoint"] = None

    # 对比学习 freeze 策略
    config["contrastive"]["freeze_tst1"] = cont_freeze_tst1
    config["contrastive"]["freeze_tst2"] = cont_freeze_tst2

    # 微调阶段保持 unfrozen
    config["finetune"]["freeze_tst1"] = False
    config["finetune"]["freeze_tst2"] = False

    config["finetune"]["save_dir"] = str(PROJECT_ROOT / "checkpoints" / "finetune" / exp_name)
    config["experiment"]["name"] = exp_name
    config["experiment"]["description"] = (
        f"对比学习消融: cont freeze_tst1={cont_freeze_tst1}, freeze_tst2={cont_freeze_tst2}, "
        f"finetune unfrozen"
    )
    config["logging"]["log_dir"] = str(PROJECT_ROOT / "logs" / exp_name)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tf:
        yaml.dump(config, tf, default_flow_style=False, allow_unicode=True)
        tmp_path = tf.name

    try:
        log_path = LOG_DIR / f"{exp_name}.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "experiments" / "run_experiment.py"),
            "--config", tmp_path,
        ]
        with open(log_path, "w") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            for line in proc.stdout:
                print(line, end="")
                logf.write(line)
                logf.flush()
            proc.wait()
        return proc.returncode == 0
    finally:
        os.unlink(tmp_path)


def main():
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/TwoTST"))
    if not output_root.exists():
        output_root = PROJECT_ROOT

    print("=" * 60)
    print("Ablation: Contrastive Freeze Strategy")
    print("attention_pooling + non-sliding-window")
    print("=" * 60)
    print(f"Base config: {BASE_CONFIG}")
    print(f"Output root: {output_root}")
    print()

    failed = []
    for variant_name, f1, f2 in CONTRASTIVE_FREEZE_VARIANTS:
        print(f"\n>>> Running: {variant_name} (cont freeze_tst1={f1}, freeze_tst2={f2})")
        ok = run_experiment(variant_name, f1, f2, output_root)
        if not ok:
            failed.append(variant_name)
        print(f"    {'OK' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print(f"Completed {len(CONTRASTIVE_FREEZE_VARIANTS)} experiments. Failed: {len(failed)}")
    if failed:
        print("Failed:", failed)
    print("Results: checkpoints/finetune/projection_fusion_attention_pooling_cont_*")
    print("=" * 60)


if __name__ == "__main__":
    main()
