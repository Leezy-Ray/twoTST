#!/usr/bin/env python3
"""
绘制对比学习阶段的训练曲线：Loss 和 Alignment vs Epoch
支持从 contrastive_history.json 读取，或从已有消融实验目录收集
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from pathlib import Path

plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

# 策略名称映射（显示用）
LABELS = {
    "freeze_tst1": "Freeze TST1, Unfreeze TST2",
    "freeze_tst2": "Unfreeze TST1, Freeze TST2",
    "freeze_both": "Freeze Both (Projection Only)",
    "unfreeze_both": "Unfreeze Both",
    "cont_freeze_tst1": "Freeze TST1, Unfreeze TST2",
    "cont_freeze_tst2": "Unfreeze TST1, Freeze TST2",
    "cont_freeze_both": "Freeze Both (Projection Only)",
    "cont_unfreeze_both": "Unfreeze Both",
}


def collect_histories(search_dirs):
    """从多个目录收集 contrastive_history.json"""
    data = {}
    for d in search_dirs:
        d = Path(d)
        if not d.exists():
            continue
        if d.is_file() and d.name == "contrastive_history.json":
            # 直接传入的 JSON 路径
            try:
                with open(d) as f:
                    obj = json.load(f)
                name = d.parent.name
                if "contrastive_curves_" in name:
                    name = name.replace("contrastive_curves_", "")
                elif "projection_fusion_attention_pooling_" in name:
                    name = name.replace("projection_fusion_attention_pooling_", "")
                data[name] = obj["history"]
            except Exception as e:
                print(f"Skip {d}: {e}")
            continue
        for p in d.rglob("contrastive_history.json"):
            try:
                with open(p) as f:
                    obj = json.load(f)
                name = p.parent.name
                if "contrastive_curves_" in name:
                    name = name.replace("contrastive_curves_", "")
                elif "projection_fusion_attention_pooling_" in name:
                    name = name.replace("projection_fusion_attention_pooling_", "")
                if name not in data:
                    data[name] = obj["history"]
                    print(f"Loaded: {p}")
            except Exception as e:
                print(f"Skip {p}: {e}")
    return data


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    output_root = Path(os.environ.get("OUTPUT_ROOT", "/root/autodl-tmp/TwoTST"))
    if not output_root.exists():
        output_root = project_root

    search_dirs = [
        output_root / "contrastive_curves",
        output_root / "checkpoints" / "finetune",
        project_root / "checkpoints" / "finetune",
    ]
    data = collect_histories(search_dirs)

    if not data:
        print("No contrastive_history.json found.")
        print("Run: python scripts/analysis/run_contrastive_curves_ablation.py")
        return

    # 排序：freeze_tst1, freeze_tst2, freeze_both, unfreeze_both
    order = ["freeze_tst1", "freeze_tst2", "freeze_both", "unfreeze_both",
             "cont_freeze_tst1", "cont_freeze_tst2", "cont_freeze_both", "cont_unfreeze_both"]
    names = [n for n in order if n in data] + [n for n in data if n not in order]

    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    out_dir = output_root / "results" if (output_root / "results").exists() else project_root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2x2: train_loss, val_loss, train_alignment, val_alignment
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    for i, name in enumerate(names):
        hist = data[name]
        if not hist:
            continue
        epochs = [h["epoch"] for h in hist]
        label = LABELS.get(name, name)
        c = colors[i % len(colors)]

        axes[0, 0].plot(epochs, [h["train_loss"] for h in hist], label=label, color=c)
        axes[0, 1].plot(epochs, [h["val_loss"] for h in hist], label=label, color=c)
        axes[1, 0].plot(epochs, [h["train_alignment"] for h in hist], label=label, color=c)
        axes[1, 1].plot(epochs, [h["val_alignment"] for h in hist], label=label, color=c)

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("(a) Contrastive Train Loss")
    axes[0, 0].legend(loc="upper right", fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_title("(b) Contrastive Val Loss")
    axes[0, 1].legend(loc="upper right", fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Alignment")
    axes[1, 0].set_title("(c) Train Alignment (Positive Pair Cosine Sim)")
    axes[1, 0].legend(loc="lower right", fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Alignment")
    axes[1, 1].set_title("(d) Val Alignment (Positive Pair Cosine Sim)")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = out_dir / "contrastive_curves.png"
    pdf_path = out_dir / "contrastive_curves.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {png_path}, {pdf_path}")


if __name__ == "__main__":
    main()
