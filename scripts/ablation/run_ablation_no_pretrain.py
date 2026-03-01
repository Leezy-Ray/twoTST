#!/usr/bin/env python3
"""
消融实验：验证 TST1/TST2 预训练是否有效

对比实验设计：
- 有预训练：加载 pretrained TST1 + TST2，然后微调（即 baseline_*_unfrozen）
- 无预训练：TST1/TST2 随机初始化，与融合层一起端到端训练

运行 5 种融合方式的无预训练实验，与已有 baseline_*_unfrozen 结果对比。
结果保存到 /root/autodl-tmp/TwoTST
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

# 5 种融合方式（与 baseline 对应）
FUSION_CONFIGS = [
    "group1_baseline_concat.yaml",
    "group1_baseline_gated.yaml",
    "group1_baseline_cross_attention.yaml",
    "group1_baseline_bilinear.yaml",
    "group1_baseline_attention_pooling.yaml",
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


def run_experiment(base_config_path, output_root):
    """加载配置，设置 use_pretrained=false，运行实验"""
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    base_name = Path(base_config_path).stem.replace("group1_baseline_", "")
    exp_name = f"no_pretrain_{base_name}"

    # 更新路径
    _replace_paths(config, output_root)

    # 核心：禁用预训练，TST1/TST2 随机初始化
    config["pretrain"]["use_pretrained"] = False

    # 端到端训练：使用较高学习率（从零训练需要更大学习率）
    config["finetune"]["lr"] = 1e-4
    config["finetune"]["freeze_tst1"] = False
    config["finetune"]["freeze_tst2"] = False
    config["finetune"]["save_dir"] = str(output_root / "checkpoints" / "finetune" / exp_name)

    if "experiment" not in config:
        config["experiment"] = {}
    config["experiment"]["name"] = exp_name
    config["experiment"]["description"] = f"无预训练：TST1/TST2 随机初始化 + {base_name} 融合，端到端训练"
    config["logging"]["log_dir"] = str(output_root / "logs" / exp_name)

    # 数据路径：若 autodl 下有数据则使用，否则保持配置
    data_path_autodl = output_root / "data" / "processed" / "processed_data.pkl"
    if data_path_autodl.exists():
        config["data"]["data_path"] = str(data_path_autodl)

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
    print("Ablation: No Pretrain (TST1/TST2 from scratch)")
    print("Validates whether pretraining TST1 and TST2 is effective")
    print(f"Output: {OUTPUT_ROOT}")
    print("=" * 60)

    total = 0
    failed = []

    for cfg_name in FUSION_CONFIGS:
        if limit > 0 and total >= limit:
            break
        cfg_path = CONFIG_DIR / cfg_name
        if not cfg_path.exists():
            print(f"  Skip (not found): {cfg_name}")
            continue
        total += 1
        fusion_name = Path(cfg_name).stem.replace("group1_baseline_", "")
        print(f"\n>>> [{total}] no_pretrain_{fusion_name}")
        ok = run_experiment(cfg_path, OUTPUT_ROOT)
        if not ok:
            failed.append(f"no_pretrain_{fusion_name}")
        print(f"    {'OK' if ok else 'FAILED'}")

    print("\n" + "=" * 60)
    print(f"Completed {total} experiments. Failed: {len(failed)}")
    if failed:
        print("Failed:", failed)
    print("Compare with: baseline_*_unfrozen (same fusion, with pretrain)")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="仅运行前 N 个实验（0=全部）")
    args = parser.parse_args()
    main(limit=args.limit)
